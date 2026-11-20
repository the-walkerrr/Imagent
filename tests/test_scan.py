from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, PngImagePlugin

from imagent.commands import scan
from imagent.db.operations import get_backlog, get_or_add_root, images_by_path, list_roots, set_keep_deleted
from imagent.image_processing.hashing import compute_hash, compute_pixel_hash
from imagent.pipeline.discovery import discover
from tests.support import IsolatedHome, add_indexed

CAN_HIDE_FOLDERS = os.name == "posix" and (not hasattr(os, "geteuid") or os.geteuid() != 0)  # root ignores permissions


class ScanBase(IsolatedHome):
    def _index(self, root: Path, names: list[str]) -> None:
        """Create the images, scan them in, and mark them as indexed."""
        for number, name in enumerate(names):
            self.make_image(root / name, shade=number)
        scan.scan(root)
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
            counts = scan.scan(root)
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
        counts = scan.scan(root)
        self.assertEqual((counts["deleted"], counts["delete_blocked"]), (0, 10))
        self.assertEqual(self._stored(root), 12)
        counts = scan.scan(root, allow_mass_delete=True)
        self.assertEqual(counts["deleted"], 10)
        self.assertEqual(self._stored(root), 2)

    def test_a_few_deletions_are_applied_without_asking(self):
        root = self.work / "pictures"
        names = [f"{n}.png" for n in range(12)]
        self._index(root, names)
        (root / names[0]).unlink()
        (root / names[1]).unlink()
        counts = scan.scan(root)
        self.assertEqual((counts["deleted"], counts["delete_blocked"]), (2, 0))

    def test_a_folder_that_shows_nothing_is_never_wiped_unasked(self):
        root = self.work / "pictures"
        self._index(root, ["only.png"])
        (root / "only.png").unlink()
        counts = scan.scan(root)
        self.assertEqual((counts["deleted"], counts["delete_blocked"]), (0, 1))

    def test_a_moved_file_keeps_its_record(self):
        root = self.work / "pictures"
        self._index(root, ["old/1.png", "2.png"])
        (root / "new").mkdir()
        shutil.move(root / "old" / "1.png", root / "new" / "renamed.png")
        counts = scan.scan(root)
        self.assertEqual((counts["moved"], counts["deleted"], counts["queued_new"]), (1, 0, 0))

    def test_empty_files_are_left_out_and_counted(self):
        root = self.work / "pictures"
        self.make_image(root / "ok.png")
        (root / "empty.png").write_bytes(b"")
        counts = scan.scan(root)
        self.assertEqual((counts["queued_new"], counts["empty"]), (1, 1))


class PictureBase(ScanBase):
    """Scans of pictures whose pixels, EXIF facts and other embedded data can each be changed alone."""

    def _picture(self, path: Path, shade: int = 0, size=(8, 4), camera: str | None = None, taken: str | None = None,
                 orientation: int | None = None, note: str | None = None, when: float | None = None) -> Path:
        """Write a PNG: ``shade`` and ``size`` are its pixels, the rest is data around them. ``when`` is the modified time to give it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", size, (shade % 256, 0, 7))
        image.putpixel((0, 0), (255, 255, 255))  # not symmetric, so turning it changes it
        exif = Image.Exif()
        for tag, value in ((0x010F, camera), (0x0132, taken), (0x0112, orientation)):
            if value is not None:
                exif[tag] = value
        text = PngImagePlugin.PngInfo()
        if note:
            text.add_text("note", note)
        image.save(path, exif=exif, pnginfo=text)
        if when is not None:
            os.utime(path, (when, when))
        return path

    def _rewrite(self, path: Path, **changes) -> None:
        """Save the picture again with ``changes`` and a later modified time, as an editor would."""
        later = os.stat(path).st_mtime + 100
        self._picture(path, when=later, **changes)

    def _index_pictures(self, root: Path, pictures: dict[str, dict]) -> None:
        for name, options in pictures.items():
            self._picture(root / name, **options)
        scan.scan(root)
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(root))
        for item in get_backlog(conn, 99):
            add_indexed(conn, root_id, Path(item["file_path"]))
            with conn:
                conn.execute("DELETE FROM backlog WHERE file_path = ?", (item["file_path"],))

    def _record(self, path: Path) -> dict | None:
        row = self.connect().execute("SELECT * FROM image_data WHERE file_path = ?", (str(path),)).fetchone()
        return dict(row) if row else None

    def _queued(self) -> list[str]:
        return [item["file_path"] for item in get_backlog(self.connect(), 99)]


class ScanStateTests(PictureBase):
    """Every file is in one state after a scan: new, changed, details updated, moved, removed or up to date."""

    def test_a_file_not_in_the_index_is_new(self):
        root = self.work / "pictures"
        self._picture(root / "a.png")
        counts = scan.scan(root)
        self.assertEqual((counts["queued_new"], counts["queued_modified"], counts["up_to_date"]), (1, 0, 0))

    def test_a_file_nothing_happened_to_is_up_to_date(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {}, "b.png": {"shade": 1}})
        counts = scan.scan(root)
        self.assertEqual((counts["up_to_date"], counts["queued_new"], counts["queued_modified"], counts["details_updated"]), (2, 0, 0, 0))

    def test_a_touched_file_is_up_to_date_and_its_timestamp_is_kept(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {}})
        later = os.stat(root / "a.png").st_mtime + 500
        os.utime(root / "a.png", (later, later))
        counts = scan.scan(root)
        self.assertEqual((counts["up_to_date"], counts["queued_modified"], counts["details_updated"]), (1, 0, 0))
        self.assertEqual(self._queued(), [])
        self.assertEqual(scan.scan(root)["up_to_date"], 1)  # the new timestamp is stored: the next scan has nothing to look at

    def test_an_edited_picture_is_changed_and_queued(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {}})
        self._rewrite(root / "a.png", shade=99)
        counts = scan.scan(root)
        self.assertEqual((counts["queued_modified"], counts["details_updated"], counts["up_to_date"]), (1, 0, 0))
        self.assertEqual(self._queued(), [str(root / "a.png")])

    def test_turning_a_picture_by_its_orientation_is_a_change_of_the_picture(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {}})
        self._rewrite(root / "a.png", orientation=6)  # the model sees the picture turned
        counts = scan.scan(root)
        self.assertEqual((counts["queued_modified"], counts["details_updated"]), (1, 0))

    def test_changed_exif_facts_are_refreshed_without_analysing_again(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {"camera": "Acme"}})
        before = self._record(root / "a.png")
        self.assertEqual(before["camera_make"], "Acme")
        self._rewrite(root / "a.png", camera="Zenith", taken="2024:05:06 07:08:09")
        counts = scan.scan(root)
        self.assertEqual((counts["details_updated"], counts["queued_modified"], counts["up_to_date"]), (1, 0, 0))
        self.assertEqual(self._queued(), [])
        after = self._record(root / "a.png")
        self.assertEqual((after["camera_make"], after["taken_at"]), ("Zenith", "2024-05-06T07:08:09"))
        self.assertEqual((after["file_id"], after["hash"], after["modified_at"]),
                         (before["file_id"], compute_hash(str(root / "a.png")), after["modified_at"]))
        self.assertNotEqual(after["hash"], before["hash"])
        self.assertEqual(after["pixel_hash"], before["pixel_hash"])
        self.assertEqual(scan.scan(root)["up_to_date"], 1)  # and now it is up to date

    def test_other_embedded_data_changing_leaves_the_file_up_to_date(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {"camera": "Acme"}})
        self._rewrite(root / "a.png", camera="Acme", note="edited in some program")
        counts = scan.scan(root)
        self.assertEqual((counts["up_to_date"], counts["details_updated"], counts["queued_modified"]), (1, 0, 0))
        self.assertEqual(self._record(root / "a.png")["hash"], compute_hash(str(root / "a.png")))

    def test_a_picture_whose_pixel_hash_is_not_known_counts_every_change_as_an_edit(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {}})
        with self.connect() as conn:
            conn.execute("UPDATE image_data SET pixel_hash = NULL")  # analysed through a converted copy
        self._rewrite(root / "a.png", note="only data around the pixels changed")
        self.assertEqual(scan.scan(root)["queued_modified"], 1)

    def test_a_picture_too_large_to_compare_counts_as_edited(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {"camera": "Acme"}})
        self._rewrite(root / "a.png", camera="Zenith")
        with mock.patch.object(scan, "check_image_size", side_effect=RuntimeError("too large")):
            counts = scan.scan(root)
        self.assertEqual((counts["queued_modified"], counts["details_updated"]), (1, 0))

    def test_a_file_that_is_gone_is_removed(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {}, "b.png": {"shade": 1}})
        (root / "a.png").unlink()
        counts = scan.scan(root)
        self.assertEqual((counts["deleted"], counts["moved"], counts["up_to_date"]), (1, 0, 1))


class MovedFileTests(PictureBase):
    def test_a_moved_file_whose_exif_was_edited_is_still_the_same_picture(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"old/a.png": {"camera": "Acme"}, "b.png": {"shade": 1}})
        before = self._record(root / "old" / "a.png")
        (root / "new").mkdir()
        shutil.move(root / "old" / "a.png", root / "new" / "renamed.png")
        self._rewrite(root / "new" / "renamed.png", camera="Zenith")
        counts = scan.scan(root)
        self.assertEqual((counts["moved"], counts["deleted"], counts["queued_new"], counts["queued_modified"]), (1, 0, 0, 0))
        self.assertEqual(self._queued(), [])
        after = self._record(root / "new" / "renamed.png")
        self.assertEqual((after["file_id"], after["camera_make"]), (before["file_id"], "Zenith"))
        self.assertEqual(after["hash"], compute_hash(str(root / "new" / "renamed.png")))

    def test_a_moved_file_whose_picture_was_edited_is_removed_and_new(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {}, "b.png": {"shade": 1}})
        shutil.move(root / "a.png", root / "c.png")
        self._rewrite(root / "c.png", shade=77)
        counts = scan.scan(root)
        self.assertEqual((counts["moved"], counts["deleted"], counts["queued_new"]), (0, 1, 1))

    def test_a_moved_file_with_the_same_bytes_is_found_without_reading_the_pixels(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {}, "b.png": {"shade": 1}})
        shutil.move(root / "a.png", root / "c.png")
        with mock.patch.object(scan, "compute_pixel_hash", side_effect=AssertionError("pixels were read")):
            counts = scan.scan(root)
        self.assertEqual((counts["renamed"], counts["moved"], counts["deleted"]), (1, 0, 0))

    def test_a_rename_is_renamed_and_a_change_of_folder_is_moved(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {}, "b.png": {"shade": 1}, "c.png": {"shade": 2}, "d.png": {"shade": 3}})
        (root / "sub").mkdir()
        (root / "a.png").rename(root / "a2.png")  # same folder, other name
        shutil.move(root / "b.png", root / "sub" / "b.png")  # other folder, same name
        shutil.move(root / "c.png", root / "sub" / "c2.png")  # other folder and other name
        counts = scan.scan(root)
        self.assertEqual((counts["renamed"], counts["moved"], counts["deleted"], counts["queued_new"], counts["up_to_date"]), (1, 2, 0, 0, 1))

    def test_new_files_are_not_read_when_nothing_could_be_their_old_self(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"a.png": {}})
        (root / "a.png").unlink()
        for number in range(3):
            self._picture(root / f"new{number}.png", shade=10 + number, size=(30, 30 + number))  # other size, other sides
        with mock.patch.object(scan, "compute_hash", side_effect=AssertionError("a new file was read")), \
                mock.patch.object(scan, "compute_pixel_hash", side_effect=AssertionError("a new file was decoded")):
            counts = scan.scan(root)
        self.assertEqual((counts["deleted"], counts["queued_new"], counts["moved"]), (1, 3, 0))

    def test_of_identical_copies_the_one_that_was_renamed_is_the_one_that_moved(self):
        root = self.work / "pictures"
        self._index_pictures(root, {"one/x.png": {"when": 1_700_000_000.0}, "two/y.png": {"when": 1_700_005_000.0}})
        record_y = self._record(root / "two" / "y.png")
        record_x = self._record(root / "one" / "x.png")
        (root / "one" / "x.png").unlink()  # x was removed ...
        (root / "three").mkdir()
        shutil.move(root / "two" / "y.png", root / "three" / "y.png")  # ... and y was moved, keeping its name and time
        counts = scan.scan(root)
        self.assertEqual((counts["moved"], counts["deleted"], counts["queued_new"]), (1, 1, 0))
        self.assertEqual(self._record(root / "three" / "y.png")["file_id"], record_y["file_id"])
        self.assertNotIn(record_x["file_id"], [row["file_id"] for row in self.connect().execute("SELECT file_id FROM image_data")])

    def test_pairing_of_copies_takes_modified_time_and_name_before_order(self):
        stored = {"/old/x.png": scan.StoredImage(1, "h", "t1"), "/old/y.png": scan.StoredImage(2, "h", "t2"),
                  "/old/z.png": scan.StoredImage(3, "h", "t3")}
        on_disk = {"/new/a.png": "t3", "/new/y.png": "t9", "/new/q.png": "t8"}
        pairs = scan._pair_moves({"h": list(on_disk)}, {"h": list(stored)}, stored, on_disk)
        self.assertEqual(pairs, {"/new/a.png": "/old/z.png", "/new/y.png": "/old/y.png", "/new/q.png": "/old/x.png"})

    def test_a_file_moved_to_another_scanned_folder_keeps_its_record(self):
        one, two = self.work / "one", self.work / "two"
        self._index_pictures(one, {"a.png": {}, "b.png": {"shade": 1}})
        self._index_pictures(two, {"c.png": {"shade": 2}})
        before = self._record(one / "a.png")
        shutil.move(one / "a.png", two / "a.png")
        counts = scan.scan(two)
        self.assertEqual((counts["moved"], counts["queued_new"], counts["deleted"]), (1, 0, 0))
        after = self._record(two / "a.png")
        self.assertEqual(after["file_id"], before["file_id"])
        self.assertNotEqual(after["root_id"], before["root_id"])
        self.assertEqual(scan.scan(one)["deleted"], 0)  # the other folder has nothing left to forget
        self.assertEqual(self._queued(), [])

    def test_a_copy_into_another_scanned_folder_is_new_not_moved(self):
        one, two = self.work / "one", self.work / "two"
        self._index_pictures(one, {"a.png": {}})
        self._index_pictures(two, {"c.png": {"shade": 2}})
        shutil.copy2(one / "a.png", two / "a.png")  # the original is still there
        counts = scan.scan(two)
        self.assertEqual((counts["moved"], counts["queued_new"]), (0, 1))
        self.assertIsNotNone(self._record(one / "a.png"))

    def test_nothing_is_taken_from_a_folder_that_cannot_be_seen(self):
        one, two = self.work / "one", self.work / "two"
        self._index_pictures(one, {"a.png": {}})
        self._index_pictures(two, {"c.png": {"shade": 2}})
        shutil.move(one / "a.png", two / "a.png")
        shutil.move(one, self.work / "one-away")  # its folder is not there either: a drive that is not connected, perhaps
        counts = scan.scan(two)
        self.assertEqual((counts["moved"], counts["queued_new"]), (0, 1))
        self.assertIsNotNone(self._record(one / "a.png"))


class PixelHashTests(PictureBase):
    def test_it_ignores_what_surrounds_the_pixels(self):
        a = self._picture(self.work / "a.png", camera="Acme", note="one")
        b = self._picture(self.work / "b.png", camera="Zenith", taken="2024:01:02 03:04:05", note="two")
        self.assertNotEqual(compute_hash(str(a)), compute_hash(str(b)))
        self.assertEqual(compute_pixel_hash(str(a)), compute_pixel_hash(str(b)))

    def test_it_sees_the_pixels_and_how_they_are_turned(self):
        a = self._picture(self.work / "a.png")
        edited = self._picture(self.work / "edited.png", shade=5)
        turned = self._picture(self.work / "turned.png", orientation=6)
        self.assertEqual(len({compute_pixel_hash(str(path)) for path in (a, edited, turned)}), 3)

    def test_a_picture_baked_turned_equals_one_turned_by_its_tag(self):
        by_tag = self._picture(self.work / "tag.png", size=(8, 4), orientation=6)
        baked = self.work / "baked.png"
        Image.open(by_tag).transpose(Image.Transpose.ROTATE_270).save(baked)  # orientation 6: turn 90 degrees clockwise
        self.assertEqual(compute_pixel_hash(str(by_tag)), compute_pixel_hash(str(baked)))

    def test_a_palette_picture_is_hashed_by_its_colors(self):
        one, two = self.work / "one.gif", self.work / "two.gif"
        for path, color in ((one, (255, 0, 0)), (two, (0, 255, 0))):
            image = Image.new("P", (4, 4))
            image.putpalette(list(color) + [0, 0, 0] * 255)
            image.save(path)
        self.assertNotEqual(compute_pixel_hash(str(one)), compute_pixel_hash(str(two)))

    def test_a_tall_picture_is_hashed_in_bands_like_a_short_one(self):
        from imagent.image_processing import hashing
        path = self._picture(self.work / "tall.png", size=(40, 300))
        whole = compute_pixel_hash(str(path))
        with mock.patch.object(hashing, "_BAND_BYTES", 400):  # a few rows at a time
            self.assertEqual(compute_pixel_hash(str(path)), whole)


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
        with mock.patch("imagent.pipeline.discovery.os.walk", return_value=[bad]), \
                mock.patch("imagent.pipeline.discovery.Path.is_file", return_value=True):
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
        return mock.patch.object(scan, "is_not_downloaded", side_effect=lambda status: status.st_ino in away_inodes)

    def test_a_file_that_is_only_in_the_cloud_is_not_read_and_is_not_queued(self):
        root = self.work / "pictures"
        self.make_image(root / "here.png", 1)
        self.make_image(root / "cloud.png", 2)
        opened = []
        with self._scan_with_away(root, ["cloud.png"]), mock.patch.object(scan, "compute_hash", side_effect=opened.append):
            counts = scan.scan(root)
        self.assertNotIn("cloud.png", [Path(path).name for path in opened])  # a new file is hashed only to tell a move from a new file
        self.assertEqual((counts["scanned"], counts["queued_new"], counts["not_downloaded"]), (1, 1, 1))
        self.assertEqual([Path(item["file_path"]).name for item in get_backlog(self.connect(), 99)], ["here.png"])

    def test_an_indexed_file_that_goes_to_the_cloud_keeps_its_record(self):
        root = self.work / "pictures"
        self._index(root, ["a.png", "b.png", "c.png"])
        with self._scan_with_away(root, ["a.png", "b.png", "c.png"]):  # all of them: the folder looks empty, like an unmounted drive
            counts = scan.scan(root)
        self.assertEqual((counts["deleted"], counts["delete_blocked"], counts["not_downloaded"]), (0, 0, 3))
        self.assertEqual(self._stored(root), 3)

    def test_a_queued_file_that_goes_to_the_cloud_stays_queued(self):
        root = self.work / "pictures"
        self.make_image(root / "a.png", 1)
        scan.scan(root)
        with self._scan_with_away(root, ["a.png"]):
            scan.scan(root)
        self.assertEqual(len(get_backlog(self.connect(), 99)), 1)

    def test_the_user_is_told_how_many_files_were_left_alone(self):
        import contextlib
        import io
        root = self.work / "pictures"
        self.make_image(root / "a.png", 1)
        out = io.StringIO()
        with self._scan_with_away(root, ["a.png"]), contextlib.redirect_stdout(out):
            scan._scan_and_report(root)
        self.assertIn("cloud storage and not downloaded", out.getvalue())
        self.assertIn("Skipped         : 1 (1 not downloaded)\n", out.getvalue())

    def test_a_file_that_went_to_the_cloud_after_the_scan_is_set_aside_with_what_to_do_and_not_downloaded(self):
        from imagent.image_processing.problems import NOT_DOWNLOADED, ImageProblem
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(self.work))
        path = self.make_image(self.work / "a.png")
        settings = {"vlm_model": "v", "embedding_model": "e", "embedding_dim": 2, "versions": {}}
        with mock.patch.object(scan, "is_not_downloaded", return_value=True), \
                mock.patch.object(scan, "compute_hash", side_effect=AssertionError("downloaded it")), \
                self.assertRaises(ImageProblem) as raised:
            scan.index_image(conn, object(), settings, root_id, path)
        self.assertEqual((raised.exception.kind, raised.exception.park, raised.exception.report), (NOT_DOWNLOADED, False, True))
        self.assertIn("available offline", str(raised.exception))

    def test_a_folder_the_system_hides_is_not_walked(self):
        root = self.work / "pictures"
        self.make_image(root / "shown" / "a.png", 1)
        self.make_image(root / "AppDataLike" / "b.png", 2)
        with mock.patch("imagent.pipeline.discovery.is_hidden_folder", side_effect=lambda path: os.path.basename(path) == "AppDataLike"):
            found = discover(root)
        self.assertEqual([path.name for path in found.images], ["a.png"])

    def test_a_scan_keeps_one_small_tuple_per_stored_image(self):
        root = self.work / "pictures"
        self._index(root, ["a.png"])
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(root))
        (record,) = images_by_path(conn, root_id).values()
        self.assertEqual(sorted(record._fields), ["file_id", "file_size", "hash", "height", "modified_at", "pixel_hash", "width"])


class ScanCommandTests(ScanBase):
    """``scan`` scans every folder that was added, and ``scan add`` adds a folder and scans it; each then offers to index the backlog."""

    def _scan_two_folders(self):
        first, second = (self.work / "first").resolve(), (self.work / "second").resolve()
        self.make_image(first / "a.png", 1)
        self.make_image(first / "b.png", 2)
        self.make_image(second / "c.png", 3)
        scan.scan(first)
        scan.scan(second)
        return first, second

    def _call(self, function, *args, answer=False):
        import contextlib
        import io
        out = io.StringIO()
        with mock.patch.object(scan, "run") as run, mock.patch.object(scan, "_ask_yes_no", return_value=answer) as yes_no, \
                contextlib.redirect_stdout(out):
            function(*args)
        return out.getvalue(), run, yes_no

    def test_scan_with_no_folder_added_says_how_to_add_one(self):
        with self.assertRaises(SystemExit) as stop:
            self._call(scan.main)
        self.assertIn('imagent scan add "folder path"', str(stop.exception.code))

    def test_scan_reports_every_folder_without_listing_them_first(self):
        first, second = (self.work / "first").resolve(), (self.work / "second").resolve()
        self._index(first, ["a.png", "b.png"])
        self._index(second, ["c.png"])
        self.make_image(second / "d.png", 4)
        report, _, _ = self._call(scan.main)
        self.assertNotIn("Folders added", report)
        for folder in (first, second):
            self.assertNotIn(f"  {folder}", report)  # each folder is named once, as its report's heading
        self.assertEqual(report.count("Already indexed :"), 2)
        self.assertIn(f"\n{first}\n  Already indexed : 2\n  To index        : 0", report)
        self.assertIn(f"\n{second}\n  Already indexed : 1\n  To index        : 1", report)  # d.png

    def test_scan_skips_a_folder_that_cannot_be_read_and_scans_the_others(self):
        first, second = self._scan_two_folders()
        shutil.rmtree(second)
        report, _, _ = self._call(scan.main)
        self.assertIn(f"Skipping {second}", report)
        self.assertEqual(report.count("Already indexed :"), 1)
    def test_scan_indexes_the_whole_backlog_when_the_user_agrees(self):
        self._scan_two_folders()
        _, run, yes_no = self._call(scan.main, answer=True)
        yes_no.assert_called_once()
        self.assertIn("3 file(s)", yes_no.call_args.args[0])
        run.assert_called_once_with()

    def test_scan_does_not_index_when_the_user_declines(self):
        self._scan_two_folders()
        report, run, _ = self._call(scan.main)
        run.assert_not_called()
        self.assertIn("Skipping indexing", report)

    def test_scan_does_not_ask_when_nothing_is_pending(self):
        root = (self.work / "pictures").resolve()
        self._index(root, ["a.png"])
        _, run, yes_no = self._call(scan.main, answer=True)
        yes_no.assert_not_called()
        run.assert_not_called()

    def test_add_adds_the_folder_and_scans_only_it(self):
        first, second = self._scan_two_folders()
        third = (self.work / "third").resolve()
        self.make_image(third / "e.png", 5)
        report, _, _ = self._call(scan.add, str(third))
        self.assertEqual([row["path"] for row in list_roots(self.connect())].count(str(third)), 1)
        self.assertEqual(report.count("Already indexed :"), 1)
        self.assertIn(f"\n{third}\n  Already indexed :", report)

    def test_add_offers_to_index_the_backlog(self):
        root = (self.work / "pictures").resolve()
        self.make_image(root / "a.png", 1)
        _, run, yes_no = self._call(scan.add, str(root), answer=True)
        yes_no.assert_called_once()
        run.assert_called_once_with()

    def test_adding_a_folder_again_scans_it_again(self):
        root = (self.work / "pictures").resolve()
        self._index(root, ["a.png"])
        self.make_image(root / "b.png", 2)
        report, _, _ = self._call(scan.add, str(root))
        self.assertIn("Already indexed : 1\n", report)
        self.assertIn("To index        : 1\n", report)
        self.assertEqual(len(list_roots(self.connect())), 1)

    def test_adding_a_folder_that_overlaps_one_added_before_is_refused(self):
        root = (self.work / "pictures").resolve()
        self.make_image(root / "inner" / "a.png", 1)
        self._call(scan.add, str(root))
        with self.assertRaises(SystemExit) as stop:
            self._call(scan.add, str(root / "inner"))
        self.assertIn("overlaps", str(stop.exception.code))


class ScanReportTests(ScanBase):
    """The report lists every state, with its count even when it is 0, in a fixed order, and explains kept images in a note."""

    def _report(self, folder):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(scan, "_ask_yes_no", return_value=False):
            scan.add(str(folder))
        return out.getvalue()

    def test_a_first_scan_shows_every_state_in_order_and_no_status(self):
        root = (self.work / "pictures").resolve()
        self.make_image(root / "a.png", 1)
        self.make_image(root / "b.png", 2)
        report = self._report(root)
        for status in ("Indexed  :", "Failed   :", "Pending  :"):
            self.assertNotIn(status, report)
        self.assertIn(f"\n{root}\n  Already indexed : 0\n", report)  # a state with no files is shown too
        self.assertIn("  To index        : 2\n", report)
        self.assertIn("  Skipped         : 0\n", report)  # no reasons when nothing was skipped
        labels = ("Already indexed", "To index", "Deleted files", "Skipped")
        positions = [report.index(f"  {label:<15} :") for label in labels]
        self.assertEqual(positions, sorted(positions))
        for gone in ("Root", "Up to date", "Removed", "Not found", "Not in folder", "Unreadable", "Note:"):
            self.assertNotIn(gone, report)

    def test_the_folder_name_is_bold_on_a_terminal_only(self):
        import contextlib
        import io
        root = (self.work / "pictures").resolve()
        self.make_image(root / "a.png", 1)

        class Terminal(io.StringIO):
            def isatty(self):
                return True

        for no_color, bold in ((False, True), (True, False)):
            out = Terminal()
            with mock.patch.dict(os.environ, {"TERM": "xterm"}), contextlib.redirect_stdout(out), \
                    mock.patch.object(scan.os, "system"), mock.patch.object(scan, "_ask_yes_no", return_value=False):
                os.environ.pop("NO_COLOR", None)
                if no_color:
                    os.environ["NO_COLOR"] = "1"
                scan.add(str(root))
            self.assertEqual(f"\n\033[1m{root}\033[0m\n" in out.getvalue(), bold)
        self.assertIn(f"\n{root}\n  Already indexed", self._report(root))  # not a terminal: plain

    def test_deleted_files_are_reported_when_there_are_some(self):
        root = (self.work / "pictures").resolve()  # main() stores the resolved path
        self._index(root, ["a.png", "b.png", "c.png", "d.png"])
        (root / "d.png").unlink()
        report = self._report(root)
        self.assertIn("Deleted files   : 1\n", report)
        self.assertIn("Already indexed : 3\n", report)

    def test_a_renamed_and_a_moved_file_are_counted_as_already_indexed(self):
        root = (self.work / "pictures").resolve()
        self._index(root, ["a.png", "b.png", "c.png"])
        (root / "a.png").rename(root / "renamed.png")
        (root / "sub").mkdir()
        (root / "b.png").rename(root / "sub" / "b.png")
        report = self._report(root)
        self.assertIn("Already indexed : 3\n", report)
        self.assertIn("To index        : 0\n", report)
        self.assertIn("Deleted files   : 0\n", report)

    def test_skipped_files_show_only_the_reasons_that_happened(self):
        root = (self.work / "pictures").resolve()
        self.make_image(root / "a.png", 1)
        (root / "empty.png").write_bytes(b"")
        report = self._report(root)
        self.assertIn("Skipped         : 1 (1 empty)\n", report)

    def test_images_kept_after_a_declined_mass_deletion_are_counted_and_explained(self):
        root = (self.work / "pictures").resolve()
        names = [f"{n}.png" for n in range(12)]
        self._index(root, names)
        for name in names[:10]:
            (root / name).unlink()
        report = self._report(root)  # the user answers no to deleting them
        self.assertNotIn("Not in folder", report)  # no row: the note explains them
        self.assertIn("Deleted files   : 0\n", report)
        self.assertIn("Note: 10 image(s) are no longer in this folder.", report)
        for case in ("Moved them?", "Drive unplugged?", "Deleted them?"):
            self.assertIn(case, report)

    def test_no_note_when_the_user_deletes_the_missing_images(self):
        root = (self.work / "pictures").resolve()
        names = [f"{n}.png" for n in range(12)]
        self._index(root, names)
        for name in names[:10]:
            (root / name).unlink()
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(scan, "_ask_yes_no", return_value=True), \
                mock.patch.object(scan, "run"):
            scan.add(str(root))
        report = out.getvalue()
        self.assertIn("Deleted files   : 10\n", report)
        self.assertNotIn("Note:", report)


class KeepDeletedTests(ScanBase):
    """What a scan keeps of the images it finds deleted: their analyses, as many as the user chose at setup."""

    def _analysed(self) -> None:
        """Give every indexed image an analysis and an embedding, which a finished index run would have stored."""
        conn = self.connect()
        with conn:
            conn.execute("UPDATE image_data SET description = 'd', short_description = 's', metadata_version = 1, vlm_model = 'v', "
                         "vlm_digest = 'vd', prompt_hash = 'p', schema_hash = 's', search_text_version = 1, ocr_version = 'o'")
            conn.execute("INSERT INTO text_embeddings(file_id, embedding, embedding_model, embedding_digest, embedding_dim) "
                         "SELECT file_id, x'0000', 'e', 'ed', 2 FROM image_data")

    def _keep(self, number: int | None) -> None:
        if number is not None:
            conn = self.connect()
            with conn:
                set_keep_deleted(conn, number)

    def _kept(self) -> list[str]:
        return [row[0] for row in self.connect().execute("SELECT pixel_hash FROM deleted_analyses ORDER BY deleted_at, rowid")]

    def _pictures(self, keep: int | None):
        root = self.work / "pictures"
        self._index(root, ["a.png", "b.png", "c.png"])  # one is always left: a folder that shows nothing is not wiped unasked
        self._analysed()
        self._keep(keep)
        return root, {name: compute_pixel_hash(str(root / name)) for name in ("a.png", "b.png", "c.png")}

    def test_the_analysis_of_a_deleted_picture_is_kept_by_its_pixels(self):
        root, pixels = self._pictures(keep=5)
        (root / "a.png").unlink()
        self.assertEqual(scan.scan(root)["deleted"], 1)
        self.assertEqual(self._kept(), [pixels["a.png"]])
        row = self.connect().execute("SELECT description, vlm_digest, embedding_digest, embedding_dim FROM deleted_analyses").fetchone()
        self.assertEqual(tuple(row), ("d", "vd", "ed", 2))

    def test_nothing_is_kept_before_the_user_chose(self):
        root, _ = self._pictures(keep=None)
        (root / "a.png").unlink()
        scan.scan(root)
        self.assertEqual(self._kept(), [])

    def test_nothing_is_kept_when_the_user_chose_none(self):
        root, _ = self._pictures(keep=0)
        (root / "a.png").unlink()
        scan.scan(root)
        self.assertEqual(self._kept(), [])

    def test_only_the_newest_deletions_up_to_the_chosen_number_are_kept(self):
        root, pixels = self._pictures(keep=1)
        (root / "a.png").unlink()
        scan.scan(root)
        (root / "b.png").unlink()
        scan.scan(root)
        self.assertEqual(self._kept(), [pixels["b.png"]])

    def test_choosing_fewer_forgets_the_oldest_at_once(self):
        root, pixels = self._pictures(keep=5)
        (root / "a.png").unlink()
        scan.scan(root)
        (root / "b.png").unlink()
        scan.scan(root)
        self.assertEqual(self._kept(), [pixels["a.png"], pixels["b.png"]])
        self._keep(1)
        self.assertEqual(self._kept(), [pixels["b.png"]])
        self._keep(0)
        self.assertEqual(self._kept(), [])

    def test_a_picture_without_a_pixel_hash_is_not_kept(self):
        root, _ = self._pictures(keep=5)
        conn = self.connect()
        with conn:
            conn.execute("UPDATE image_data SET pixel_hash = NULL")
        (root / "a.png").unlink()
        scan.scan(root)
        self.assertEqual(self._kept(), [])

    def test_a_picture_that_was_moved_is_not_kept_as_deleted(self):
        root, _ = self._pictures(keep=5)
        (root / "a.png").rename(root / "renamed.png")
        counts = scan.scan(root)
        self.assertEqual((counts["renamed"], counts["deleted"]), (1, 0))
        self.assertEqual(self._kept(), [])
