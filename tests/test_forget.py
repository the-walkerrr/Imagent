"""Forgetting the deleted pictures Imagent remembers (imagent forget) and the menu entry that offers it."""

import contextlib
import io
import sys
import unittest
from unittest import mock

from imagent.commands import forget, menu
from imagent.db.operations import count_deleted_analyses, get_keep_deleted, set_keep_deleted
from imagent.pipeline.locking import AlreadyRunning
from tests.support import IsolatedHome


def remember(conn, count):
    """Put ``count`` deleted pictures' analyses in the database, as scans that found pictures deleted would."""
    with conn:
        for number in range(count):
            conn.execute("INSERT INTO deleted_analyses (pixel_hash, vlm_metadata_json, metadata_version, embedding, deleted_at) "
                         "VALUES (?, '{}', 1, x'00', '2026-01-01T00:00:00')", (f"hash{number}",))


class ConfirmationTests(unittest.TestCase):
    def ask(self, answers, remembered=7):
        with mock.patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(io.StringIO()):
            return forget.confirm_twice(remembered)

    def test_two_right_answers_confirm(self):
        self.assertTrue(self.ask(["YES", "7"]))

    def test_each_wrong_answer_cancels(self):
        self.assertFalse(self.ask(["no"]))
        self.assertFalse(self.ask(["yes", "6"]))


class MainTests(IsolatedHome):
    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        stdin = mock.Mock()
        stdin.isatty.return_value = True
        patcher = mock.patch.object(sys, "stdin", stdin)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, answers=()):
        with mock.patch("builtins.input", side_effect=list(answers)), contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                forget.main()
                code = None
            except SystemExit as exit_info:
                code = exit_info.code
        return code, out.getvalue()

    def test_with_nothing_remembered_it_says_so_and_asks_nothing(self):
        code, out = self.run_main()
        self.assertIsNone(code)
        self.assertIn("does not remember any deleted pictures", out)

    def test_two_right_answers_forget_everything_but_keep_the_choice_of_how_many_to_remember(self):
        with self.conn:
            set_keep_deleted(self.conn, 500)
        remember(self.conn, 3)
        code, out = self.run_main(["yes", "3"])
        self.assertIsNone(code)
        self.assertEqual(count_deleted_analyses(self.conn), 0)
        self.assertEqual(get_keep_deleted(self.conn), 500)
        self.assertIn("Forgot 3 deleted picture(s)", out)

    def test_a_wrong_answer_deletes_nothing(self):
        remember(self.conn, 3)
        for answers in (["no"], ["yes", "2"]):
            code, _ = self.run_main(answers)
            self.assertNotEqual(code, 0)
            self.assertEqual(count_deleted_analyses(self.conn), 3)

    def test_nothing_is_deleted_while_indexing_runs(self):
        remember(self.conn, 3)
        with mock.patch.object(forget, "single_instance", side_effect=AlreadyRunning("Indexing is already running in another window.")):
            code, _ = self.run_main(["yes", "3"])
        self.assertIn("Nothing was deleted", str(code))
        self.assertEqual(count_deleted_analyses(self.conn), 3)

    def test_no_terminal_stops_before_anything(self):
        stdin = mock.Mock()
        stdin.isatty.return_value = False
        with mock.patch.object(sys, "stdin", stdin), self.assertRaisesRegex(SystemExit, "interactive terminal"):
            forget.main()


class MenuTests(IsolatedHome):
    def test_the_menu_offers_it_only_when_something_is_remembered(self):
        for remembered, offered in ((0, False), (4, True)):
            values = [choice.value for choice in menu.choices(0, remembered)]
            self.assertEqual(menu.FORGET_COMMAND in values, offered)
            self.assertEqual(values[-3:], [menu.HELP[1], menu.SETTINGS[1], menu.EXIT])
        self.assertIn("Forget the 4 deleted picture(s) Imagent remembers", [choice.title for choice in menu.choices(0, 4)])

    def test_the_menu_counts_what_is_remembered(self):
        self.assertEqual(menu.remembered_count(), 0)
        remember(self.connect(), 2)
        self.assertEqual(menu.remembered_count(), 2)


if __name__ == "__main__":
    unittest.main()
