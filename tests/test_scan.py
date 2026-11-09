import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

from scan_and_index import scan_and_index
from db.operations import get_backlog, get_or_add_root, images_by_path
from pipeline.discovery import discover
from tests.support import IsolatedHome, add_indexed

CAN_HIDE_FOLDERS = os.name == "posix" and (not hasattr(os, "geteuid") or os.geteuid() != 0)  # root ignores permissions


class ScanBase(IsolatedHome):
    def _index(self, root: Path, names: list[str]) -> None:
        """Create the images, scan them in, and mark them as indexed."""
        for number, name in enumerate(names):
            self.make_image(root / name, shade=number)
        scan_and_index.scan(root)
        conn = self.connect()
        root_id = get_or_add_root(conn, str(root))
        for item in get_backlog(conn, 99):
            add_indexed(conn, root_id, Path(item["file_path"]))
            with conn:
                conn.execute("DELETE FROM backlog WHERE file_path = ?", (item["file_path"],))

    def _stored(self, root: Path) -> int:
        conn = self.connect()
        with conn:  # get_or_add_root writes, and an open write transaction would block the next scan
            root_id = get_or_add_root(conn, str(root))
        return len(images_by_path(conn, root_id))


class ScanSafetyTests(ScanBase):
    @unittest.skipUnless(CAN_HIDE_FOLDERS, "needs folder permissions that root ignores")
    def test_a_folder_that_cannot_be_read_keeps_its_images(self):
        root = self.work / "pictures"
        self._index(root, ["a/1.png", "a/2.png", "b/3.png"])
        (root / "a").chmod(0)
        try:
            counts = scan_and_index.scan(root)
        finally:
            (root / "a").chmod(0o755)
        self.assertEqual(counts["deleted"], 0)
        self.assertEqual(counts["unreadable_dirs"], 1)
        self.assertEqual(self._stored(root), 3)

    def test_losing_most_of_a_folder_asks_before_forgetting(self):
        root = self.work / "pictures"
        names = [f"{n}.png" for n in range(12)]
        self._index(root, names)
        for name in names[:10]:
            (root / name).unlink()
        counts = scan_and_index.scan(root)
        self.assertEqual((counts["deleted"], counts["delete_blocked"]), (0, 10))
        self.assertEqual(self._stored(root), 12)
        counts = scan_and_index.scan(root, allow_mass_delete=True)
        self.assertEqual(counts["deleted"], 10)
        self.assertEqual(self._stored(root), 2)

    def test_a_few_deletions_are_applied_without_asking(self):
        root = self.work / "pictures"
        names = [f"{n}.png" for n in range(12)]
        self._index(root, names)
        (root / names[0]).unlink()
        (root / names[1]).unlink()
        counts = scan_and_index.scan(root)
        self.assertEqual((counts["deleted"], counts["delete_blocked"]), (2, 0))

    def test_a_folder_that_shows_nothing_is_never_wiped_unasked(self):
        root = self.work / "pictures"
        self._index(root, ["only.png"])
        (root / "only.png").unlink()
        counts = scan_and_index.scan(root)
        self.assertEqual((counts["deleted"], counts["delete_blocked"]), (0, 1))

    def test_a_moved_file_keeps_its_record(self):
        root = self.work / "pictures"
        self._index(root, ["old/1.png", "2.png"])
        (root / "new").mkdir()
        shutil.move(root / "old" / "1.png", root / "new" / "renamed.png")
        counts = scan_and_index.scan(root)
        self.assertEqual((counts["moved"], counts["deleted"], counts["queued_new"]), (1, 0, 0))

    def test_empty_files_are_left_out_and_counted(self):
        root = self.work / "pictures"
        self.make_image(root / "ok.png")
        (root / "empty.png").write_bytes(b"")
        counts = scan_and_index.scan(root)
        self.assertEqual((counts["queued_new"], counts["empty"]), (1, 1))


class DiscoveryTests(IsolatedHome):
    def test_junk_is_skipped_and_pictures_are_found(self):
        root = self.work / "pictures"
        keep = [self.make_image(root / "holiday" / "beach.JPG"), self.make_image(root / "top.png", 1)]
        for junk in ("._beach.jpg", ".hidden/x.jpg", "node_modules/pkg/x.png", "Photos.photoslibrary/resources/x.jpg",
                     "@eaDir/x.jpg", "notes.txt"):
            (root / junk).parent.mkdir(parents=True, exist_ok=True)
            (root / junk).write_bytes(b"x")
        self.assertEqual(discover(root).images, sorted(keep))

    def test_a_name_that_is_not_valid_text_is_reported_not_fatal(self):
        bad = ("/pictures", [], ["bad\udce9.jpg", "good.jpg"])
        with mock.patch("pipeline.discovery.os.walk", return_value=[bad]), \
                mock.patch("pipeline.discovery.Path.is_file", return_value=True):
            found = discover(Path("/pictures"))
        self.assertEqual([path.name for path in found.images], ["good.jpg"])
        self.assertEqual(len(found.bad_names), 1)
        self.assertNotIn("\udce9", found.bad_names[0])  # safe to print and to store

    @unittest.skipUnless(CAN_HIDE_FOLDERS, "needs folder permissions that root ignores")
    def test_unreadable_folders_are_reported(self):
        root = self.work / "pictures"
        self.make_image(root / "open" / "1.png")
        self.make_image(root / "locked" / "2.png", 1)
        (root / "locked").chmod(0)
        try:
            found = discover(root)
        finally:
            (root / "locked").chmod(0o755)
        self.assertEqual([path.name for path in found.unreadable_dirs], ["locked"])
        self.assertEqual([path.name for path in found.images], ["1.png"])


if __name__ == "__main__":
    unittest.main()


class CloudAndHiddenTests(ScanBase):
    """Files kept in cloud storage are neither read nor forgotten; folders the system hides are not walked."""

    def _scan_with_away(self, root, away_names):
        """``is_not_downloaded`` says yes for the files with these names, as the operating system would for an evicted file."""
        away_inodes = {os.stat(root / name).st_ino for name in away_names}
        return mock.patch.object(scan_and_index, "is_not_downloaded", side_effect=lambda status: status.st_ino in away_inodes)

    def test_a_file_that_is_only_in_the_cloud_is_not_read_and_is_not_queued(self):
        root = self.work / "pictures"
        self.make_image(root / "here.png", 1)
        self.make_image(root / "cloud.png", 2)
        opened = []
        with self._scan_with_away(root, ["cloud.png"]), mock.patch.object(scan_and_index, "compute_hash", side_effect=opened.append):
            counts = scan_and_index.scan(root)
        self.assertNotIn("cloud.png", [Path(path).name for path in opened])  # a new file is hashed only to tell a move from a new file
        self.assertEqual((counts["scanned"], counts["queued_new"], counts["not_downloaded"]), (1, 1, 1))
        self.assertEqual([Path(item["file_path"]).name for item in get_backlog(self.connect(), 99)], ["here.png"])

    def test_an_indexed_file_that_goes_to_the_cloud_keeps_its_record(self):
        root = self.work / "pictures"
        self._index(root, ["a.png", "b.png", "c.png"])
        with self._scan_with_away(root, ["a.png", "b.png", "c.png"]):  # all of them: the folder looks empty, like an unmounted drive
            counts = scan_and_index.scan(root)
        self.assertEqual((counts["deleted"], counts["delete_blocked"], counts["not_downloaded"]), (0, 0, 3))
        self.assertEqual(self._stored(root), 3)

    def test_a_queued_file_that_goes_to_the_cloud_stays_queued(self):
        root = self.work / "pictures"
        self.make_image(root / "a.png", 1)
        scan_and_index.scan(root)
        with self._scan_with_away(root, ["a.png"]):
            scan_and_index.scan(root)
        self.assertEqual(len(get_backlog(self.connect(), 99)), 1)

    def test_the_user_is_told_how_many_files_were_left_alone(self):
        import contextlib
        import io
        root = self.work / "pictures"
        self.make_image(root / "a.png", 1)
        out = io.StringIO()
        with self._scan_with_away(root, ["a.png"]), contextlib.redirect_stdout(out):
            scan_and_index._scan_and_report(root)
        self.assertIn("cloud storage and not downloaded", out.getvalue())
        self.assertIn("1 image(s) in cloud storage", out.getvalue())

    def test_a_file_that_went_to_the_cloud_after_the_scan_is_set_aside_with_what_to_do_and_not_downloaded(self):
        from image_processing.problems import NOT_DOWNLOADED, ImageProblem
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(self.work))
        path = self.make_image(self.work / "a.png")
        settings = {"vlm_model": "v", "embedding_model": "e", "embedding_dim": 2, "versions": {}}
        with mock.patch.object(scan_and_index, "is_not_downloaded", return_value=True), \
                mock.patch.object(scan_and_index, "compute_hash", side_effect=AssertionError("downloaded it")), \
                self.assertRaises(ImageProblem) as raised:
            scan_and_index.index_image(conn, object(), settings, root_id, path)
        self.assertEqual((raised.exception.kind, raised.exception.park, raised.exception.report), (NOT_DOWNLOADED, False, True))
        self.assertIn("available offline", str(raised.exception))

    def test_a_folder_the_system_hides_is_not_walked(self):
        root = self.work / "pictures"
        self.make_image(root / "shown" / "a.png", 1)
        self.make_image(root / "AppDataLike" / "b.png", 2)
        with mock.patch("pipeline.discovery.is_hidden_folder", side_effect=lambda path: os.path.basename(path) == "AppDataLike"):
            found = discover(root)
        self.assertEqual([path.name for path in found.images], ["a.png"])

    def test_a_scan_keeps_one_small_tuple_per_stored_image(self):
        root = self.work / "pictures"
        self._index(root, ["a.png"])
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(root))
        (record,) = images_by_path(conn, root_id).values()
        self.assertEqual(sorted(record._fields), ["file_id", "hash", "modified_at"])
