import contextlib
import io
import os
import re
import sqlite3
import subprocess
import sys
import unittest
from unittest import mock

from imagent import cli
from imagent.commands import delete, failed, forget, menu, scan, search, setup, uninstall
from imagent.commands import help as help_command
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


class ExitGapTests(unittest.TestCase):
    """On a terminal, every way the program ends leaves a blank line before the shell's prompt, once."""

    def end(self, run, tty=True):
        out, err = io.StringIO(), io.StringIO()
        out.isatty = lambda: tty
        code = None
        with mock.patch.object(cli, "_run", side_effect=run), mock.patch("imagent.pipeline.console.gap_shown", False), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                cli.main([])
            except SystemExit as stop:
                code = stop.code
        return out.getvalue(), err.getvalue(), code

    def test_a_normal_end(self):
        self.assertEqual(self.end(lambda argv: None)[0], "\n")

    def test_a_message_is_shown_before_the_gap(self):
        out, err, code = self.end(lambda argv: sys.exit("Nothing to do."))
        self.assertEqual((out, err, code), ("\n", "Nothing to do.\n", 1))

    def test_an_exit_code_is_kept(self):
        self.assertEqual(self.end(lambda argv: sys.exit(2)), ("\n", "", 2))

    def test_no_gap_when_output_is_redirected(self):
        self.assertEqual(self.end(lambda argv: sys.exit("Nothing to do."), tty=False), ("", "", "Nothing to do."))


class CommandLineTests(unittest.TestCase):
    def run_command(self, *argv, needs_setup=False):
        with mock.patch.object(menu, "needs_setup", return_value=needs_setup), mock.patch.object(search, "main") as run_search, mock.patch.object(scan, "main") as run_scan, mock.patch.object(scan, "add") as run_scan_add, \
                mock.patch.object(failed, "main") as run_failed, mock.patch.object(menu, "main") as run_menu, \
                mock.patch.object(setup, "main") as run_setup, mock.patch.object(uninstall, "main") as run_uninstall, \
                mock.patch.object(delete, "main") as run_delete, mock.patch("imagent.pipeline.folder_picker.start_listing"):
            cli.main(list(argv))
        return {"search": run_search, "scan": run_scan, "add": run_scan_add, "failed": run_failed, "menu": run_menu, "setup": run_setup,
                "uninstall": run_uninstall, "delete": run_delete}

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

    def test_add_gets_the_folder(self):
        runs = self.run_command("add", "/pictures")
        runs["add"].assert_called_once_with("/pictures")
        self.assertEqual([name for name, run in runs.items() if run.called], ["add"])

    def test_add_without_a_folder_starts_listing_the_folders_before_setup(self):
        from imagent.pipeline import folder_picker
        order = []
        with mock.patch.object(folder_picker, "start_listing", side_effect=lambda: order.append("list")) as listing, \
                mock.patch.object(menu, "needs_setup", return_value=True), mock.patch.object(setup, "main", side_effect=lambda: order.append("setup")), \
                mock.patch.object(scan, "add"), mock.patch.object(sys.stdin, "isatty", return_value=True), contextlib.redirect_stdout(io.StringIO()):
            cli.main(["add"])
        listing.assert_called_once_with()
        self.assertEqual(order, ["list", "setup"])

    def test_the_folders_are_not_listed_when_a_folder_is_given_or_there_is_no_terminal(self):
        from imagent.pipeline import folder_picker
        for argv, tty in ((["add", "/pictures"], True), (["add"], False), (["scan"], True), (["search", "cat"], True)):
            with mock.patch.object(folder_picker, "start_listing") as listing, mock.patch.object(menu, "needs_setup", return_value=False), \
                    mock.patch.object(scan, "add"), mock.patch.object(scan, "main"), mock.patch.object(search, "main"), \
                    mock.patch.object(sys.stdin, "isatty", return_value=tty):
                cli.main(argv)
            listing.assert_not_called()

    def test_add_without_a_folder_asks_for_one_itself(self):
        runs = self.run_command("add")
        runs["add"].assert_called_once_with(None)

    def test_delete_gets_every_folder_given(self):
        runs = self.run_command("delete", "/pictures", "/my photos")
        runs["delete"].assert_called_once_with(["/pictures", "/my photos"])
        self.assertEqual([name for name, run in runs.items() if run.called], ["delete"])

    def test_delete_without_a_folder_chooses_from_the_added_ones(self):
        self.run_command("delete")["delete"].assert_called_once_with([])

    def test_scan_with_a_folder_is_gone(self):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as stop:
            self.run_command("scan", "/pictures")
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("unrecognized arguments", err.getvalue())

    def test_scan_add_is_gone(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
            self.run_command("scan", "add", "/pictures")
        self.assertEqual(stop.exception.code, 2)

    def test_failed_takes_no_arguments(self):
        runs = self.run_command("failed")
        runs["failed"].assert_called_once_with()
        self.assertEqual([name for name, run in runs.items() if run.called], ["failed"])

    def test_setup_and_uninstall_take_no_arguments(self):
        for command in ("setup", "uninstall"):
            runs = self.run_command(command)
            runs[command].assert_called_once_with()
            self.assertEqual(sorted(name for name, run in runs.items() if run.called),
                             sorted([command, "menu"] if command == "setup" else [command]))  # setup opens the menu once it is done

    def test_help_runs_without_setup_and_takes_no_arguments(self):
        with mock.patch.object(menu, "needs_setup", return_value=True), mock.patch.object(setup, "main") as run_setup, \
                mock.patch.object(help_command, "main") as run_help:
            cli.main(["help"])
        run_help.assert_called_once_with()
        run_setup.assert_not_called()

    def test_help_explains_every_command(self):
        commands = set(cli.build_parser()._subparsers._group_actions[0].choices)
        self.assertEqual({name for name, _, _ in help_command.COMMANDS}, commands)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            help_command.main()
        for command in commands:
            self.assertIn(f"imagent {command}", out.getvalue())

    def test_help_is_styled_only_on_a_terminal_and_says_the_same_either_way(self):
        plain = help_command.render()
        self.assertNotIn("\033[", plain)
        with mock.patch.object(sys.stdout, "isatty", return_value=True), mock.patch.dict(os.environ, {"TERM": "xterm"}):
            os.environ.pop("NO_COLOR", None)
            styled = help_command.render()
            os.environ["NO_COLOR"] = "1"
            self.assertEqual(help_command.render(), plain)
        self.assertIn("\033[", styled)
        self.assertEqual(re.sub(r"\033\[[0-9;]*m", "", styled), plain)
        self.assertIn(f"\033[{help_command.BOLD}mimagent search \"red car at night\" --full\033[0m", styled)

    def test_help_table_fits_a_narrow_terminal_and_keeps_the_descriptions_in_the_right_column(self):
        right = " " * help_command.RIGHT
        for columns in (60, 80, 200):
            with mock.patch.object(help_command.shutil, "get_terminal_size", return_value=os.terminal_size((columns, 24))):
                table = [line for line in help_command.render().splitlines() if line.startswith(" ")]
            self.assertLessEqual(max(len(line) for line in table), min(columns - 1, help_command.WIDTH), columns)
            for line in table:  # a row of a table starts with a name, every other line of it sits in the right column
                self.assertTrue(line.startswith(right) or line[2] != " ", (columns, line))
            self.assertGreater(sum(line.startswith(right) for line in table), len(help_command.COMMANDS))

    def test_every_command_the_menu_offers_exists(self):
        for command in (menu.SEARCH[1], menu.SCAN[1], menu.FAILED_COMMAND, menu.HELP[1], menu.UNINSTALL):
            self.assertEqual(cli.build_parser().parse_args([command]).command, command)
        added = cli.build_parser().parse_args([*menu.ADD_FOLDER[1].split(), "/pictures"])
        self.assertEqual((added.command, added.folder), ("add", "/pictures"))

    def test_no_command_opens_the_menu(self):
        runs = self.run_command()
        runs["menu"].assert_called_once_with()
        self.assertEqual([name for name, run in runs.items() if run.called], ["menu"])

    def test_the_menu_opens_only_after_setup_when_it_has_not_been_done(self):
        with mock.patch.object(menu, "needs_setup", return_value=True), mock.patch.object(setup, "main") as run_setup, \
                mock.patch.object(menu, "main") as run_menu, contextlib.redirect_stdout(io.StringIO()):
            order = []
            run_setup.side_effect = lambda: order.append("setup")
            run_menu.side_effect = lambda: order.append("menu")
            cli.main([])
        self.assertEqual(order, ["setup", "menu"])

    def test_the_menu_does_not_open_when_setup_is_cancelled_or_fails(self):
        for error in (KeyboardInterrupt, SystemExit("Ollama is required.")):
            with mock.patch.object(menu, "needs_setup", return_value=True), mock.patch.object(setup, "main", side_effect=error), \
                    mock.patch.object(menu, "main") as run_menu, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                cli.main([])
            run_menu.assert_not_called()

    def test_every_command_but_setup_and_uninstall_runs_setup_first_when_it_has_not_been_done(self):
        for argv in (["search", "red car"], ["scan"], ["add", "/pictures"], ["failed"], ["forget"]):
            with mock.patch.object(menu, "needs_setup", return_value=True), mock.patch.object(setup, "main") as run_setup, \
                    mock.patch.object(search, "main") as run_search, mock.patch.object(scan, "main") as run_scan, \
                    mock.patch.object(scan, "add") as run_scan_add, mock.patch.object(failed, "main") as run_failed, \
                    mock.patch.object(forget, "main") as run_forget, contextlib.redirect_stdout(io.StringIO()):
                order = []
                run_setup.side_effect = lambda: order.append("setup")
                for run in (run_search, run_scan, run_scan_add, run_failed, run_forget):
                    run.side_effect = lambda *a, **k: order.append("command")
                cli.main(argv)
            self.assertEqual(order, ["setup", "command"], argv)

    def test_setup_and_uninstall_do_not_run_setup_first(self):
        for command in ("setup", "uninstall"):
            runs = self.run_command(command, needs_setup=True)
            self.assertEqual(runs["setup"].call_count, command == "setup")

    def test_the_command_does_not_run_when_setup_is_cancelled_or_fails(self):
        for error in (KeyboardInterrupt, SystemExit("Ollama is required.")):
            with mock.patch.object(menu, "needs_setup", return_value=True), \
                    mock.patch.object(setup, "main", side_effect=error), mock.patch.object(scan, "add") as run_scan_add, \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(["add", "/pictures"])
            run_scan_add.assert_not_called()

    def test_a_command_that_does_not_exist_is_refused_and_nothing_runs(self):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as stop:
            self.run_command("reembed")
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("invalid choice", err.getvalue())

    def test_a_command_ends_politely_when_it_is_cancelled(self):
        with mock.patch.object(menu, "needs_setup", return_value=False), mock.patch.object(search, "main", side_effect=KeyboardInterrupt), \
                self.assertRaises(SystemExit) as stop:
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

    def test_search_and_scan_are_only_offered_once_a_folder_has_been_added(self):
        without = [choice.value for choice in menu.choices(folders=False)]
        self.assertNotIn(menu.SEARCH[1], without)
        self.assertNotIn(menu.SCAN[1], without)
        self.assertNotIn(menu.DELETE_FOLDER[1], without)
        self.assertEqual(without[0], menu.ADD_FOLDER[1])
        with_folder = [choice.value for choice in menu.choices(folders=True)]
        self.assertEqual(with_folder[:4], [menu.SEARCH[1], menu.ADD_FOLDER[1], menu.SCAN[1], menu.DELETE_FOLDER[1]])

    def test_search_is_only_offered_once_an_image_is_indexed(self):
        self.assertNotIn(menu.SEARCH[1], [choice.value for choice in menu.choices(folders=True, indexed=False)])
        self.assertIn(menu.SEARCH[1], [choice.value for choice in menu.choices(folders=True, indexed=True)])

    def test_an_unreadable_database_still_gets_search_in_the_menu(self):
        with mock.patch.object(menu, "init_db", side_effect=DatabaseProblem("damaged")):
            self.assertTrue(menu.has_indexed_images())

    def test_an_unreadable_database_still_gets_the_full_menu(self):
        with mock.patch.object(menu, "init_db", side_effect=DatabaseProblem("damaged")):
            self.assertTrue(menu.has_folders())

    def _choose(self, choice):
        """Run the menu with ``choice`` picked; returns the arguments the command got."""
        with mock.patch.object(menu.questionary, "select") as select, \
                mock.patch.object(menu, "failed_count", return_value=0), mock.patch.object(menu.subprocess, "Popen") as popen:
            select.return_value.ask.return_value = choice
            popen.return_value.wait.return_value = 0
            with self.assertRaises(SystemExit):
                menu.main()
        return popen.call_args.args[0][3:] if popen.called else None

    def test_adding_a_folder_runs_add_which_asks_for_it(self):
        self.assertEqual(self._choose(menu.ADD_FOLDER[1]), ["add"])

    def test_other_choices_run_as_they_are(self):
        self.assertEqual(self._choose(menu.SCAN[1]), ["scan"])

    def test_help_is_always_offered_just_before_exit(self):
        for failed in (0, 3):
            values = [choice.value for choice in menu.choices(failed, remembered=failed)]
            self.assertEqual(values[-3:], [menu.HELP[1], menu.EXIT, menu.UNINSTALL])
        self.assertEqual(self._choose(menu.HELP[1]), ["help"])

    def test_exit_closes_the_menu_without_running_anything(self):
        with mock.patch.object(menu.questionary, "select") as select, mock.patch.object(menu, "failed_count", return_value=0), \
                mock.patch.object(menu.subprocess, "Popen") as popen:
            select.return_value.ask.return_value = menu.EXIT
            menu.main()
        popen.assert_not_called()

    def test_exit_is_offered_just_above_uninstall(self):
        for failed in (0, 3):
            values = [choice.value for choice in menu.choices(failed)]
            self.assertEqual(values[-2:], [menu.EXIT, menu.UNINSTALL])


if __name__ == "__main__":
    unittest.main()
