"""Asking for the folder to add: which folders are suggested, in what order, and the prompt itself driven by typed keys."""

import contextlib
import io
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from imagent.pipeline import folder_picker
from imagent.pipeline.discovery import list_folders
from imagent.pipeline.folder_picker import Candidate, candidates, rank


def make(home: str, *folders: str) -> list[str]:
    paths = []
    for folder in folders:
        path = os.path.join(home, *folder.split("/"))
        os.makedirs(path, exist_ok=True)
        paths.append(path)
    return paths


class CandidatesTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = os.path.realpath(tmp.name)

    def shown(self, folders, added=()):
        return [c.shown.replace(os.sep, "/") for c in candidates(folders, self.home, list(added))]

    def test_folders_are_shown_below_the_home_folder(self):
        folders = make(self.home, "Pictures/Trips")
        self.assertEqual(self.shown(folders), ["~/Pictures/Trips"])

    def test_the_home_folder_itself_and_folders_outside_it_are_not_suggested(self):
        outside = os.path.join(os.path.dirname(self.home), "elsewhere")
        self.assertEqual(self.shown([self.home, outside]), [])

    def test_folders_the_scan_skips_and_what_is_inside_them_are_not_suggested(self):
        folders = make(self.home, "app/node_modules/pkg", "app/node_modules", ".hidden/photos", "app/src", "cache/__pycache__")
        self.assertEqual(sorted(self.shown(folders)), ["~/app/src"])

    def test_a_folder_the_system_hides_is_not_suggested_nor_what_is_inside_it(self):
        folders = make(self.home, "Library/Caches", "Pictures")
        with mock.patch.object(folder_picker, "is_hidden_folder", side_effect=lambda path: path.endswith("Library")):
            self.assertEqual(self.shown(folders), ["~/Pictures"])

    def test_folders_added_before_and_overlapping_ones_are_not_suggested(self):
        folders = make(self.home, "Pictures", "Pictures/Trips", "Documents", "Documents/Scans", "Music")
        added = [os.path.join(self.home, "Pictures"), os.path.join(self.home, "Documents/Scans")]
        self.assertEqual(sorted(self.shown(folders, added)), ["~/Music"])  # Pictures + inside it; Scans + what holds it


def candidate(path: str) -> Candidate:
    return Candidate("/home/u/" + path, "~/" + path, path.casefold())


class RankTests(unittest.TestCase):
    def ranked(self, query, *paths):
        return [c.shown[2:] for c in rank([candidate(p) for p in paths], query)]

    def test_the_folder_whose_own_name_matches_comes_before_one_whose_parent_matches(self):
        self.assertEqual(self.ranked("trips", "Trips/2019", "Pictures/Trips", "Old/Trips/Rome"),
                         ["Pictures/Trips", "Trips/2019", "Old/Trips/Rome"])

    def test_each_step_back_from_the_end_ranks_lower(self):
        self.assertEqual(self.ranked("rome", "rome/b/c", "x/rome/y", "a/rome", "rome/b"), ["a/rome", "rome/b", "x/rome/y", "rome/b/c"])

    def test_within_a_depth_the_whole_name_then_the_start_of_it_then_the_middle(self):
        self.assertEqual(self.ranked("pic", "a/my pictures", "a/pictures", "a/pic", "a/picnic"), ["a/pic", "a/picnic", "a/pictures", "a/my pictures"])

    def test_a_shorter_path_comes_first_among_equals(self):
        self.assertEqual(self.ranked("pic", "x/y/pic", "pic"), ["pic", "x/y/pic"])

    def test_upper_and_lower_case_are_the_same(self):
        self.assertEqual(self.ranked("TRIPS", "Pictures/trips"), ["Pictures/trips"])

    def test_more_than_one_word_of_a_path_can_be_typed(self):
        self.assertEqual(self.ranked("pictures/tri", "Pictures/Trips", "Other/Trips"), ["Pictures/Trips"])

    def test_fewer_than_three_characters_match_nothing(self):
        self.assertEqual(self.ranked("pi", "pictures", "pi"), [])
        self.assertEqual(self.ranked(" pi ", "pictures"), [])  # spaces around it do not count
        self.assertEqual(self.ranked("pic", "pictures"), ["pictures"])

    def test_nothing_typed_or_nothing_matching_gives_nothing(self):
        self.assertEqual(self.ranked("   ", "a"), [])
        self.assertEqual(self.ranked("zzz", "a"), [])


class ListFoldersTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = os.path.realpath(tmp.name)

    def listed(self):
        return sorted(os.path.relpath(path, self.home).replace(os.sep, "/") for path in list_folders(self.home))

    def test_every_folder_below_is_listed_at_any_depth(self):
        make(self.home, "Pictures/Trips/Rome", "Documents")
        self.assertEqual(self.listed(), ["Documents", "Pictures", "Pictures/Trips", "Pictures/Trips/Rome"])

    def test_folders_the_scan_skips_are_neither_listed_nor_entered(self):
        make(self.home, "app/src", "app/node_modules/pkg", ".hidden/photos", "Photos.photoslibrary/originals", "Safari.app/Contents/Resources",
             "Lightroom Previews.lrdata/a", "#recycle/old", "Program Files/tool", "keep/site-packages/x")
        self.assertEqual(self.listed(), ["app", "app/src", "keep"])

    def test_a_folder_the_system_hides_is_neither_listed_nor_entered(self):
        make(self.home, "Library/Caches", "Pictures")
        with mock.patch("imagent.pipeline.discovery.is_hidden_folder", side_effect=lambda path: path.endswith("Library")):
            self.assertEqual(self.listed(), ["Pictures"])

    @unittest.skipUnless(hasattr(os, "symlink") and os.name != "nt", "needs links that can be made without special rights")
    def test_a_link_to_a_folder_is_listed_but_not_followed(self):
        target = tempfile.TemporaryDirectory()
        self.addCleanup(target.cleanup)
        make(target.name, "inside")
        os.symlink(target.name, os.path.join(self.home, "link"))
        os.symlink(self.home, os.path.join(self.home, "loop"))
        self.assertEqual(self.listed(), ["link", "loop"])

    def test_a_folder_that_cannot_be_read_is_skipped_and_the_rest_is_listed(self):
        make(self.home, "a/b", "c")
        real_scandir = os.scandir

        def scandir(path):
            if path.endswith("a"):
                raise PermissionError(path)
            return real_scandir(path)

        with mock.patch("imagent.pipeline.discovery.os.scandir", side_effect=scandir):
            self.assertEqual(self.listed(), ["a", "c"])  # a is listed (it is a folder) but what is in it is unknown

    def test_a_folder_that_is_not_there_lists_nothing(self):
        self.assertEqual(list_folders(os.path.join(self.home, "missing")), [])


class ListingTests(unittest.TestCase):
    """The listing that starts with the command and is waited for by the picker."""

    def setUp(self):
        self.addCleanup(setattr, folder_picker, "_listing", None)
        folder_picker._listing = None

    def test_it_starts_at_once_and_the_picker_gets_its_result(self):
        with mock.patch.object(folder_picker, "list_folders", return_value=["/h/a"]) as listing, mock.patch.dict(os.environ, {"HOME": "/h", "USERPROFILE": "/h"}):
            folder_picker.start_listing()
            self.assertEqual(folder_picker._folders_below(os.path.expanduser("~")), ["/h/a"])
        listing.assert_called_once_with(os.path.expanduser("/h"))

    def test_the_listing_started_early_is_used_not_started_again(self):
        release = threading.Event()
        calls = []

        def slow(home):
            calls.append(home)
            release.wait(5)
            return ["/h/a"]

        with mock.patch.object(folder_picker, "list_folders", side_effect=slow):
            folder_picker._listing_of("/h")
            threading.Timer(0.05, release.set).start()
            self.assertEqual(folder_picker._folders_below("/h"), ["/h/a"])  # waits for the one under way
        self.assertEqual(calls, ["/h"])

    def test_a_listing_is_used_once_so_the_next_one_sees_new_folders(self):
        with mock.patch.object(folder_picker, "list_folders", side_effect=[["/h/a"], ["/h/a", "/h/b"]]):
            self.assertEqual(folder_picker._folders_below("/h"), ["/h/a"])
            self.assertEqual(folder_picker._folders_below("/h"), ["/h/a", "/h/b"])

    def test_a_listing_of_another_home_is_not_reused(self):
        with mock.patch.object(folder_picker, "list_folders", side_effect=lambda home: [home + "/x"]):
            folder_picker._listing_of("/h")
            self.assertEqual(folder_picker._folders_below("/other"), ["/other/x"])


class AskTests(unittest.TestCase):
    def test_the_picker_is_used_and_told_what_was_added(self):
        with mock.patch.object(folder_picker, "_Picker") as picker:
            picker.return_value.ask.return_value = "/pictures"
            self.assertEqual(folder_picker.ask_for_folder(["/a"], None), "/pictures")
        picker.assert_called_once_with(["/a"])


class PickerTests(unittest.TestCase):
    """The prompt run for real against a pipe that stands in for the keyboard."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = os.path.realpath(tmp.name)
        self.folders = make(self.home, "Pictures/Trips", "Pictures/Trips/Rome", "Trips", "Documents")

    def run_picker(self, keys: str, added=()):
        """Open the prompt, type ``keys`` once the folders are listed, and return what it gave."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        listed = threading.Event()
        real_load = folder_picker._Picker._load

        def load(picker):
            real_load(picker)
            listed.set()

        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), \
                mock.patch.dict(os.environ, {"HOME": self.home}), \
                mock.patch.object(folder_picker, "list_folders", return_value=self.folders), \
                mock.patch.object(folder_picker._Picker, "_load", load), mock.patch("prompt_toolkit.shortcuts.print_formatted_text"), \
                contextlib.redirect_stdout(io.StringIO()):
            threading.Thread(target=lambda: (listed.wait(5), pipe.send_text(keys)), daemon=True).start()
            return folder_picker._Picker(list(added)).ask(None)

    def test_typing_part_of_a_name_and_pressing_enter_picks_the_best_match(self):
        self.assertEqual(self.run_picker("trips\r"), os.path.join(self.home, "Trips"))  # the end folder named Trips, shorter path first

    def test_the_arrow_keys_move_down_the_list(self):
        self.assertEqual(self.run_picker("trips\x1b[B\r"), os.path.join(self.home, "Pictures", "Trips"))

    def test_a_typed_path_is_used_as_it_is(self):
        self.assertEqual(self.run_picker("~/somewhere\r"), "~/somewhere")

    def test_two_characters_suggest_nothing_so_enter_uses_them_as_a_path(self):
        self.assertEqual(self.run_picker("tr\r"), "tr")

    def test_ctrl_c_cancels(self):
        self.assertIsNone(self.run_picker("\x03"))


if __name__ == "__main__":
    unittest.main()
