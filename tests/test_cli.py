import contextlib
import io
import sqlite3
import unittest

from db.connection import DatabaseProblem
from pipeline.console import run_cli
from tests.support import IsolatedHome


class RunCliTests(IsolatedHome):
    def _ends_with(self, error: BaseException) -> SystemExit:
        def main():
            raise error
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stop:
                run_cli(main, cancelled="Search cancelled.")
        return stop.exception

    def test_ctrl_c(self):
        self.assertIn("Search cancelled.", str(self._ends_with(KeyboardInterrupt()).code))

    def test_a_question_with_no_terminal_to_answer_it(self):
        self.assertIn("no terminal", str(self._ends_with(EOFError()).code))

    def test_database_problems_are_shown_as_their_message(self):
        self.assertIn("is damaged", str(self._ends_with(DatabaseProblem("The Imagent database is damaged.")).code))

    def test_a_database_error_in_the_middle_of_work_is_explained(self):
        self.assertIn("disk is full", str(self._ends_with(sqlite3.OperationalError("database or disk is full")).code))

    def test_the_unexpected_ends_politely_and_keeps_the_details(self):
        try:
            raise ZeroDivisionError("oops")
        except ZeroDivisionError as exc:
            message = str(self._ends_with(exc).code)
        self.assertIn("Something unexpected went wrong", message)
        self.assertIn("last_error.log", message)
        self.assertIn("ZeroDivisionError", (self.home / "last_error.log").read_text())
        self.assertNotIn("Traceback", message)

    def test_a_clean_exit_passes_through(self):
        with self.assertRaises(SystemExit) as stop:
            run_cli(lambda: (_ for _ in ()).throw(SystemExit(0)))
        self.assertEqual(stop.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
