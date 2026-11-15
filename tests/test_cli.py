import contextlib
import io
import os
import sqlite3
import subprocess
import sys
import unittest
from unittest import mock

from imagent import cli
from imagent.commands import failed, menu, scan, search, setup, uninstall
from imagent.db.connection import DatabaseProblem
from imagent.pipeline.console import run_cli
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


class CommandLineTests(unittest.TestCase):
    def run_command(self, *argv):
        with mock.patch.object(search, "main") as run_search, mock.patch.object(scan, "main") as run_scan, \
                mock.patch.object(failed, "main") as run_failed, mock.patch.object(menu, "main") as run_menu, \
                mock.patch.object(setup, "main") as run_setup, mock.patch.object(uninstall, "main") as run_uninstall:
            cli.main(list(argv))
        return {"search": run_search, "scan": run_scan, "failed": run_failed, "menu": run_menu, "setup": run_setup,
                "uninstall": run_uninstall}

    def test_search_gets_the_words_typed_as_one_query(self):
        runs = self.run_command("search", "red", "car", "at", "night")
        runs["search"].assert_called_once_with("red car at night")
        self.assertEqual([name for name, run in runs.items() if run.called], ["search"])

    def test_search_with_no_words_gets_an_empty_query(self):
        self.run_command("search")["search"].assert_called_once_with("")

    def test_scan_gets_the_folder_or_none(self):
        self.run_command("scan", "/pictures")["scan"].assert_called_once_with("/pictures")
        self.run_command("scan")["scan"].assert_called_once_with(None)

    def test_failed_takes_no_arguments(self):
        runs = self.run_command("failed")
        runs["failed"].assert_called_once_with()
        self.assertEqual([name for name, run in runs.items() if run.called], ["failed"])

    def test_setup_and_uninstall_take_no_arguments(self):
        for command in ("setup", "uninstall"):
            runs = self.run_command(command)
            runs[command].assert_called_once_with()
            self.assertEqual([name for name, run in runs.items() if run.called], [command])

    def test_every_command_the_menu_offers_exists(self):
        for command in (menu.SEARCH[1], menu.SCAN[1], menu.FAILED_COMMAND, menu.UNINSTALL):
            self.assertEqual(cli.build_parser().parse_args([command]).command, command)

    def test_no_command_opens_the_menu(self):
        runs = self.run_command()
        runs["menu"].assert_called_once_with()
        self.assertEqual([name for name, run in runs.items() if run.called], ["menu"])

    def test_a_command_that_does_not_exist_is_refused_and_nothing_runs(self):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as stop:
            self.run_command("reembed")
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("invalid choice", err.getvalue())

    def test_a_command_ends_politely_when_it_is_cancelled(self):
        with mock.patch.object(search, "main", side_effect=KeyboardInterrupt), self.assertRaises(SystemExit) as stop:
            cli.main(["search", "x"])
        self.assertIn("Search cancelled.", str(stop.exception.code))

    def test_python_dash_m_imagent_is_the_same_command(self):
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run([sys.executable, "-m", "imagent", "--help"], capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("search", "scan", "failed", "setup", "uninstall"):
            self.assertIn(command, result.stdout)


class MenuTests(unittest.TestCase):
    def test_the_choices_start_with_search_and_end_with_uninstall(self):
        for failed in (0, 3):
            values = [choice.value for choice in menu.choices(failed)]
            self.assertEqual(values[0], menu.SEARCH[1])
            self.assertEqual(values[-1], menu.UNINSTALL)
            self.assertEqual(menu.FAILED_COMMAND in values, failed > 0)


if __name__ == "__main__":
    unittest.main()
