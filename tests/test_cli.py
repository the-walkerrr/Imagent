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
from imagent.db.operations import set_device_settings, set_model_settings
from tests.support import DEVICE, SETTINGS, IsolatedHome


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
    def run_command(self, *argv, needs_setup=False):
        with mock.patch.object(menu, "needs_setup", return_value=needs_setup), mock.patch.object(search, "main") as run_search, mock.patch.object(scan, "main") as run_scan, mock.patch.object(scan, "add") as run_scan_add, \
                mock.patch.object(failed, "main") as run_failed, mock.patch.object(menu, "main") as run_menu, \
                mock.patch.object(setup, "main") as run_setup, mock.patch.object(uninstall, "main") as run_uninstall:
            cli.main(list(argv))
        return {"search": run_search, "scan": run_scan, "scan add": run_scan_add, "failed": run_failed, "menu": run_menu, "setup": run_setup,
                "uninstall": run_uninstall}

    def test_search_gets_the_words_typed_as_one_query(self):
        runs = self.run_command("search", "red", "car", "at", "night")
        runs["search"].assert_called_once_with("red car at night", full=False)
        self.assertEqual([name for name, run in runs.items() if run.called], ["search"])

    def test_search_full_is_passed_on(self):
        self.run_command("search", "--full", "red", "car")["search"].assert_called_once_with("red car", full=True)

    def test_search_full_can_come_after_the_query(self):
        self.run_command("search", "bike in a cloudy weather", "--full")["search"].assert_called_once_with("bike in a cloudy weather", full=True)
        self.run_command("search", "bike", "in", "cloudy", "--full")["search"].assert_called_once_with("bike in cloudy", full=True)

    def test_search_with_no_words_gets_an_empty_query(self):
        self.run_command("search")["search"].assert_called_once_with("", full=False)

    def test_scan_takes_no_arguments(self):
        runs = self.run_command("scan")
        runs["scan"].assert_called_once_with()
        self.assertEqual([name for name, run in runs.items() if run.called], ["scan"])

    def test_scan_add_gets_the_folder(self):
        runs = self.run_command("scan", "add", "/pictures")
        runs["scan add"].assert_called_once_with("/pictures")
        self.assertEqual([name for name, run in runs.items() if run.called], ["scan add"])

    def test_scan_add_needs_a_folder(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
            self.run_command("scan", "add")
        self.assertEqual(stop.exception.code, 2)

    def test_scan_with_a_folder_is_gone(self):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as stop:
            self.run_command("scan", "/pictures")
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("invalid choice", err.getvalue())

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
        added = cli.build_parser().parse_args([*menu.ADD_FOLDER[1].split(), "/pictures"])
        self.assertEqual((added.command, added.scan_command, added.folder), ("scan", "add", "/pictures"))

    def test_no_command_opens_the_menu(self):
        runs = self.run_command()
        runs["menu"].assert_called_once_with()
        self.assertEqual([name for name, run in runs.items() if run.called], ["menu"])

    def test_no_command_runs_setup_first_when_it_has_not_been_done(self):
        runs = self.run_command(needs_setup=True)
        runs["setup"].assert_called_once_with()
        runs["menu"].assert_called_once_with()
        self.assertEqual([name for name, run in runs.items() if run.called], ["menu", "setup"])

    def test_the_menu_does_not_open_when_setup_is_cancelled_or_fails(self):
        for error in (KeyboardInterrupt, SystemExit("Ollama is required.")):
            with mock.patch.object(menu, "needs_setup", return_value=True), \
                    mock.patch.object(setup, "main", side_effect=error), mock.patch.object(menu, "main") as run_menu, \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                cli.main([])
            run_menu.assert_not_called()

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


class NeedsSetupTests(IsolatedHome):
    def test_a_new_install_needs_setup(self):
        self.assertTrue(menu.needs_setup())

    def test_models_without_the_measurements_still_need_setup(self):
        conn = self.connect()
        with conn:
            set_model_settings(conn, SETTINGS)
        self.assertTrue(menu.needs_setup())

    def test_a_finished_setup_does_not(self):
        conn = self.connect()
        with conn:
            set_device_settings(conn, DEVICE, None)
            set_model_settings(conn, SETTINGS)
        self.assertFalse(menu.needs_setup())

    def test_a_database_that_cannot_be_read_still_gets_the_menu(self):
        with mock.patch.object(menu, "init_db", side_effect=DatabaseProblem("damaged")):
            self.assertFalse(menu.needs_setup())


class MenuTests(unittest.TestCase):
    def test_the_choices_start_with_search_and_end_with_uninstall(self):
        for failed in (0, 3):
            values = [choice.value for choice in menu.choices(failed)]
            self.assertEqual(values[0], menu.SEARCH[1])
            self.assertEqual(values[-1], menu.UNINSTALL)
            self.assertEqual(menu.FAILED_COMMAND in values, failed > 0)

    def _choose(self, choice, typed):
        """Run the menu with ``choice`` picked and ``typed`` entered at the folder prompt; returns the arguments the command got."""
        with mock.patch.object(menu.questionary, "select") as select, mock.patch.object(menu.questionary, "text") as text, \
                mock.patch.object(menu, "failed_count", return_value=0), mock.patch.object(menu.subprocess, "Popen") as popen:
            select.return_value.ask.return_value = choice
            text.return_value.ask.return_value = typed
            popen.return_value.wait.return_value = 0
            with self.assertRaises(SystemExit):
                menu.main()
        return popen.call_args.args[0][3:] if popen.called else None

    def test_adding_a_folder_asks_for_it_and_runs_scan_add(self):
        self.assertEqual(self._choose(menu.ADD_FOLDER[1], "  /pictures "), ["scan", "add", "/pictures"])

    def test_other_choices_run_without_asking_for_a_folder(self):
        self.assertEqual(self._choose(menu.SCAN[1], None), ["scan"])

    def test_no_folder_typed_runs_nothing(self):
        for typed in (None, "   "):
            self.assertIsNone(self._choose(menu.ADD_FOLDER[1], typed))


if __name__ == "__main__":
    unittest.main()
