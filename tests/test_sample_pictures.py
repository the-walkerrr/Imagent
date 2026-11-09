"""The test pictures: the zip that ships, the ones made on this computer, and the check that reads them all."""

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from PIL import Image

import config
from image_processing import sample_pictures
from install import install

SMALL = 1_500_000  # bytes for the largest picture made, so these tests are quick


class Folder(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name) / "pictures"
        patcher = mock.patch.object(config, "FILE_SETTLE_SECONDS", 0)  # a damaged file is not watched for being written
        patcher.start()
        self.addCleanup(patcher.stop)


class ShippedZipTests(Folder):
    def test_every_picture_in_the_manifest_is_in_the_zip_with_its_source_and_licence(self):
        with zipfile.ZipFile(sample_pictures.ZIP_PATH) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            names = set(archive.namelist())
            for entry in manifest:
                self.assertIn(entry["file"], names)
                self.assertEqual(archive.getinfo(entry["file"]).file_size, entry["bytes"])
                self.assertTrue(entry["source"].startswith("https://commons.wikimedia.org/"))
                self.assertIn(entry["license"], ("CC0", "Public domain"))  # the pictures ship to everyone
        self.assertEqual(names - {"manifest.json"}, {entry["file"] for entry in manifest})

    def test_the_pictures_cover_the_formats_and_the_sizes_people_have(self):
        with zipfile.ZipFile(sample_pictures.ZIP_PATH) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        suffixes = {Path(entry["file"]).suffix for entry in manifest}
        self.assertTrue({".jpg", ".png", ".webp", ".heic", ".bmp", ".tiff", ".gif"} <= suffixes)
        longest = [max(entry["width"], entry["height"]) for entry in manifest]
        self.assertLess(min(longest), 400)
        self.assertGreater(max(longest), 5000)

    def test_unpacking_gives_every_picture_back_and_every_one_can_be_read(self):
        count = sample_pictures.unpack(self.folder)
        self.assertGreaterEqual(count, 25)
        outcomes = sample_pictures.check_pictures(self.folder)
        self.assertEqual(set(outcomes) - {"read"}, set(), outcomes)

    def test_unpacking_twice_leaves_the_files_alone(self):
        sample_pictures.unpack(self.folder)
        mine = self.folder / "cat.jpg"
        stamp = mine.stat().st_mtime_ns
        sample_pictures.unpack(self.folder)
        self.assertEqual(mine.stat().st_mtime_ns, stamp)

    def test_a_zip_cannot_put_a_file_outside_its_folder(self):
        evil = Path(self.folder.parent) / "evil.zip"
        with zipfile.ZipFile(evil, "w") as archive:
            archive.writestr("../escaped.txt", "x")
            archive.writestr("sub/inside.txt", "x")
            archive.writestr("fine.jpg", "x")
        with mock.patch.object(sample_pictures, "ZIP_PATH", evil):
            self.assertEqual(sample_pictures.unpack(self.folder), 1)
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ["fine.jpg"])
        self.assertFalse((self.folder.parent / "escaped.txt").exists())


class MadePicturesTests(Folder):
    def test_pictures_up_to_the_target_size_are_made_and_the_damaged_ones_come_out_damaged(self):
        expected = sample_pictures.make_extremes(self.folder, target_bytes=SMALL)
        huge = self.folder / "extreme-huge.jpg"
        self.assertGreater(huge.stat().st_size, SMALL * 0.5)
        self.assertLess(huge.stat().st_size, SMALL * 2)
        self.assertLess((self.folder / "extreme-large.jpg").stat().st_size, huge.stat().st_size)
        said = []
        outcomes = sample_pictures.check_pictures(self.folder, expected, say=said.append)
        self.assertEqual(said, [])  # every picture came to what it should
        self.assertEqual(outcomes.get("damaged"), 3)
        self.assertNotIn("unexpected", outcomes)

    def test_a_picture_that_expands_to_more_pixels_than_the_memory_holds_is_too_large_not_an_error(self):
        sample_pictures.make_extremes(self.folder, target_bytes=SMALL)
        outcomes = sample_pictures.check_pictures(self.folder)
        self.assertGreaterEqual(outcomes.get("too large", 0), 1)

    def test_what_does_not_fit_in_this_computers_memory_is_left_out_and_said(self):
        said = []
        with mock.patch.object(sample_pictures, "available_memory", return_value=1_000_000):
            made = sample_pictures.make_extremes(self.folder, target_bytes=SMALL, say=said.append)
        self.assertFalse(any(name.startswith("extreme-") and name.endswith(".jpg") for name in made))
        self.assertEqual(len(said), 3)
        self.assertIn("not enough free memory", said[0])
        self.assertIn("damaged-empty.jpg", made)  # the small ones are still made

    def test_what_does_not_fit_on_this_computers_disk_is_left_out_and_said(self):
        said = []
        with mock.patch.object(sample_pictures.shutil, "disk_usage", return_value=mock.Mock(free=1000)):
            sample_pictures.make_extremes(self.folder, target_bytes=SMALL, say=said.append)
        self.assertTrue(all("not enough free disk space" in line for line in said))

    def test_a_picture_that_does_not_come_to_what_it_should_is_said(self):
        sample_pictures.make_extremes(self.folder, target_bytes=SMALL)
        said = []
        sample_pictures.check_pictures(self.folder, {"damaged-empty.jpg": "read"}, say=said.append)
        self.assertEqual(said, ["damaged-empty.jpg came to 'damaged', not 'read'."])

    def test_odd_ways_of_storing_colour_are_read(self):
        sample_pictures.make_extremes(self.folder, target_bytes=SMALL)
        for name in ("odd-transparent.png", "odd-sixteen-bit.png", "odd-cmyk.jpg", "odd-panorama.jpg", "odd-one-pixel.png"):
            with Image.open(self.folder / name) as picture:
                picture.load()  # they are real pictures
        outcomes = sample_pictures.check_pictures(self.folder, {n: "read" for n in ("odd-transparent.png", "odd-sixteen-bit.png",
                                                                                   "odd-cmyk.jpg", "odd-panorama.jpg", "odd-one-pixel.png")})
        self.assertNotIn("unexpected", outcomes)


class SetupStepTests(unittest.TestCase):
    def run_step(self, **patches):
        said = []
        with mock.patch.object(install, "say", lambda message, style="": said.append(message)), \
                mock.patch.object(config, "IMAGENT_HOME", Path(tempfile.mkdtemp())):
            for name, value in patches.items():
                patcher = mock.patch.object(sample_pictures, name, value)
                patcher.start()
                self.addCleanup(patcher.stop)
            install.check_test_pictures()
        return said

    def test_the_outcome_is_told_in_one_line(self):
        said = self.run_step(unpack=mock.Mock(return_value=3), make_extremes=mock.Mock(return_value={}),
                             check_pictures=mock.Mock(return_value={"read": 5, "damaged": 3}))
        self.assertIn("✓ Checked 8 test pictures: 3 damaged, 5 read.", said)

    def test_a_picture_that_came_to_the_wrong_outcome_is_shown_as_a_warning(self):
        def check(folder, expected, say):
            say("damaged-empty.jpg came to 'read', not 'damaged'.")
            return {"read": 1}
        said = self.run_step(unpack=mock.Mock(), make_extremes=mock.Mock(return_value={}), check_pictures=check)
        self.assertIn("⚠ damaged-empty.jpg came to 'read', not 'damaged'.", said)

    def test_a_problem_preparing_the_pictures_does_not_stop_setup(self):
        said = self.run_step(unpack=mock.Mock(side_effect=OSError("disk full")))
        self.assertTrue(any("could not be prepared" in line and "disk full" in line for line in said))


if __name__ == "__main__":
    unittest.main()
