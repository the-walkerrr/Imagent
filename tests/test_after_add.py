"""The menu shown after a folder was added: what it offers, and that it comes back after each choice until Exit."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

from imagent.commands import after_add

FOLDER = Path("/pictures/trips")


class MenuTests(unittest.TestCase):
    def run_menu(self, answers, added=(str(FOLDER),), indexed=True, **patches):
        """Run the menu with ``answers`` chosen in turn; returns the mocks it used, keyed by name, and its output. ``added`` is what is added at the start."""
        answers = list(answers)
        menus = []

        def select(message, **kwargs):
            menus.append([(choice.title, choice.value) for choice in kwargs["choices"] if not isinstance(choice, after_add.questionary.Separator)])
            return mock.Mock(ask=lambda: answers.pop(0))

        mocks = {name: mock.Mock() for name in ("add_folder", "scan_and_index", "main", "delete", "search")}
        for name, value in patches.items():
            mocks[name] = value
        with mock.patch.object(after_add.questionary, "select", side_effect=select), \
                mock.patch.object(after_add, "has_indexed_images", side_effect=lambda: indexed), \
                mock.patch.object(after_add.scan, "added_folders", return_value=list(added)), \
                mock.patch.object(after_add.scan, "add_folder", mocks["add_folder"]), \
                mock.patch.object(after_add.scan, "scan_and_index", mocks["scan_and_index"]), \
                mock.patch.object(after_add.scan, "main", mocks["main"]), \
                mock.patch.object(after_add.delete, "main", mocks["delete"]), \
                mock.patch.object(after_add.search, "main", mocks["search"]), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                after_add.run(FOLDER)
                stopped = None
            except SystemExit as stop:
                stopped = stop.code
        mocks["menus"], mocks["stopped"], mocks["out"], mocks["err"] = menus, stopped, out.getvalue(), err.getvalue()
        return mocks

    def test_it_offers_what_was_asked_for_in_that_order(self):
        menu = self.run_menu([after_add.EXIT])["menus"][0]
        self.assertEqual([value for _, value in menu],
                         [after_add.ADD, after_add.SCAN_THIS, after_add.SCAN_ALL, after_add.DELETE, after_add.SEARCH, after_add.EXIT])
        self.assertIn(str(FOLDER), dict((value, title) for title, value in menu)[after_add.SCAN_THIS])

    def test_search_is_not_offered_while_no_image_is_indexed(self):
        menu = self.run_menu([after_add.EXIT], indexed=False)["menus"][0]
        self.assertEqual([value for _, value in menu], [after_add.ADD, after_add.SCAN_THIS, after_add.SCAN_ALL, after_add.DELETE, after_add.EXIT])

    def test_exit_ends_the_menu_and_runs_nothing(self):
        ran = self.run_menu([after_add.EXIT])
        self.assertEqual(len(ran["menus"]), 1)
        for name in ("add_folder", "scan_and_index", "main", "delete", "search"):
            ran[name].assert_not_called()

    def test_each_choice_runs_its_command_and_the_menu_comes_back(self):
        ran = self.run_menu([after_add.SCAN_THIS, after_add.SCAN_ALL, after_add.DELETE, after_add.SEARCH, after_add.EXIT])
        ran["scan_and_index"].assert_called_once_with(FOLDER)
        ran["main"].assert_called_once_with()
        ran["delete"].assert_called_once_with([])
        ran["search"].assert_called_once_with()
        self.assertEqual(len(ran["menus"]), 5)

    def test_adding_another_folder_makes_it_the_one_scan_and_index_is_about(self):
        other = Path("/pictures/other")
        ran = self.run_menu([after_add.ADD, after_add.SCAN_THIS, after_add.EXIT], added=(str(FOLDER), str(other)),
                            add_folder=mock.Mock(return_value=other))
        ran["add_folder"].assert_called_once_with()
        ran["scan_and_index"].assert_called_once_with(other)
        self.assertIn(str(other), dict((value, title) for title, value in ran["menus"][1])[after_add.SCAN_THIS])

    def test_an_add_that_fails_shows_why_and_keeps_the_folder_added_before(self):
        ran = self.run_menu([after_add.ADD, after_add.SCAN_THIS, after_add.EXIT],
                            add_folder=mock.Mock(side_effect=SystemExit("Error: it overlaps")))
        self.assertIn("Error: it overlaps", ran["err"])
        ran["scan_and_index"].assert_called_once_with(FOLDER)
        self.assertEqual(len(ran["menus"]), 3)

    def test_a_folder_deleted_from_the_menu_is_no_longer_offered_to_scan(self):
        added = [str(FOLDER)]
        removed = mock.Mock(side_effect=lambda paths: added.clear())
        answers = iter([after_add.DELETE, after_add.EXIT])
        menus = []

        def select(message, **kwargs):
            menus.append([choice.value for choice in kwargs["choices"]])
            return mock.Mock(ask=lambda: next(answers))

        with mock.patch.object(after_add.questionary, "select", side_effect=select), \
                mock.patch.object(after_add.scan, "added_folders", side_effect=lambda: list(added)), \
                mock.patch.object(after_add.delete, "main", removed), contextlib.redirect_stdout(io.StringIO()):
            after_add.run(FOLDER)
        self.assertIn(after_add.SCAN_THIS, menus[0])
        self.assertNotIn(after_add.SCAN_THIS, menus[1])

    def test_scanning_everything_with_no_folder_added_says_so_instead_of_asking_for_one(self):
        ran = self.run_menu([after_add.SCAN_ALL, after_add.EXIT], added=())
        ran["main"].assert_not_called()
        self.assertIn("nothing to scan", ran["out"])

    def test_a_command_that_ends_with_a_message_or_ctrl_c_brings_the_menu_back(self):
        for error, shown in ((SystemExit("Nothing was deleted."), "Nothing was deleted."), (KeyboardInterrupt(), "Cancelled.")):
            ran = self.run_menu([after_add.DELETE, after_add.EXIT], delete=mock.Mock(side_effect=error))
            self.assertIsNone(ran["stopped"])
            self.assertEqual(len(ran["menus"]), 2)
            self.assertIn(shown, ran["err"])

    def test_ctrl_c_at_the_menu_ends_the_program(self):
        self.assertEqual(self.run_menu([None])["stopped"], 1)


if __name__ == "__main__":
    unittest.main()
