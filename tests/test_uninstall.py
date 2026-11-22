import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from imagent import config
from imagent.commands import setup
from imagent.db.operations import get_or_add_root
from imagent.commands import uninstall
from tests.support import IsolatedHome, add_indexed


def _model(name):
    return SimpleNamespace(model=name)


class InstallRecordTests(IsolatedHome):
    def test_a_model_is_recorded_once(self):
        setup._remember_pulled("vision")
        setup._remember_pulled("vision:latest")
        self.assertEqual(config.load_install_record(), {"ollama_models": ["vision:latest"]})

    def test_only_models_that_were_downloaded_are_recorded(self):
        client = mock.Mock()
        client.list.return_value = SimpleNamespace(models=[_model("had-before:latest")])
        client.pull.return_value = iter([SimpleNamespace(status="done", total=None, completed=None)])
        with contextlib.redirect_stdout(io.StringIO()):
            setup._pull_models(client, ["had-before", "new-one:1b"])
        self.assertEqual(config.load_install_record(), {"ollama_models": ["new-one:1b"]})

    def test_installing_ollama_is_recorded_beside_the_models(self):
        setup._remember_pulled("vision")
        config.record_ollama_installed()
        self.assertEqual(config.load_install_record(), {"ollama_models": ["vision:latest"], "ollama_installed": True})

    def test_setup_records_ollama_only_when_it_installed_it_itself(self):
        with mock.patch.object(setup, "say"), mock.patch.object(setup, "ensure_ollama_current"), \
                mock.patch.object(setup.shutil, "which", return_value="/usr/bin/ollama"):
            setup.ensure_ollama_installed()  # already there
        self.assertEqual(config.load_install_record(), {})
        with mock.patch.object(setup, "say"), mock.patch.object(setup, "ask", return_value=True), \
                mock.patch.object(setup.questionary, "confirm"), mock.patch.object(setup, "install_command", return_value=["brew", "install", "ollama"]), \
                mock.patch.object(setup, "_succeeds", return_value=True), mock.patch.object(setup.shutil, "which", side_effect=[None, "/usr/bin/ollama"]):
            setup.ensure_ollama_installed()
        self.assertEqual(config.load_install_record(), {"ollama_installed": True})

    def test_a_failed_ollama_install_is_not_recorded(self):
        with mock.patch.object(setup, "say"), mock.patch.object(setup, "ask", return_value=True), \
                mock.patch.object(setup.questionary, "confirm"), mock.patch.object(setup, "install_command", return_value=["brew", "install", "ollama"]), \
                mock.patch.object(setup, "_succeeds", return_value=False), mock.patch.object(setup.shutil, "which", return_value=None), \
                mock.patch.object(setup, "fail", side_effect=SystemExit):
            with self.assertRaises(SystemExit):
                setup.ensure_ollama_installed()
        self.assertEqual(config.load_install_record(), {})

    def test_a_missing_or_damaged_record_reads_as_empty(self):
        self.assertEqual(config.load_install_record(), {})
        self.home.mkdir(parents=True)
        (self.home / "install_record.json").write_text("[1,")
        self.assertEqual(config.load_install_record(), {})


class StatsTests(IsolatedHome):
    def test_no_database_gives_unknowns_not_a_crash(self):
        stats = uninstall.read_stats()
        self.assertIsNone(stats["images"])
        self.assertEqual(stats["folders"], [])

    def test_counts_come_from_the_database(self):
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(self.work))
        add_indexed(conn, root_id, self.make_image(self.work / "a.png", 1))
        add_indexed(conn, root_id, self.make_image(self.work / "b.png", 2))
        stats = uninstall.read_stats()
        self.assertEqual((stats["images"], stats["queued"], stats["errors"]), (2, 0, 0))
        self.assertEqual(stats["folders"], [str(self.work)])


class PlanTests(IsolatedHome):
    def test_only_recorded_and_present_models_are_planned(self):
        config.save_install_record({"ollama_models": ["ours:latest", "removed-already:latest"]})
        stats = {"locked_models": ["ours:latest", "theirs:latest"]}
        with mock.patch.object(uninstall, "_installed_models",
                               return_value={"ours:latest": 5, "theirs:latest": 9, "other:latest": 1}):
            plan = uninstall.make_plan("http://x", stats)
        self.assertEqual(plan["models"], [("ours:latest", 5)])
        self.assertEqual(plan["models_gone"], ["removed-already:latest"])
        self.assertEqual(plan["not_recorded"], ["theirs:latest"])

    def test_an_unreachable_ollama_removes_no_model(self):
        config.save_install_record({"ollama_models": ["ours:latest"]})
        with mock.patch.object(uninstall, "_installed_models", return_value=None):
            plan = uninstall.make_plan(None, {"locked_models": []})
        self.assertEqual(plan["models"], [])
        self.assertEqual(plan["models_unreachable"], ["ours:latest"])


CODE = "aB3dE5gH7j"


class OllamaRemovalCommandTests(IsolatedHome):
    def command(self, system, which, service_file=True, euid=1000):
        service = self.work / "ollama.service"
        if service_file:
            service.write_text("[Unit]")
        with mock.patch.object(uninstall.platform, "system", return_value=system), \
                mock.patch.object(uninstall.shutil, "which", side_effect=lambda name: which.get(name)), \
                mock.patch.object(uninstall, "LINUX_SERVICE_FILE", service), \
                mock.patch.object(uninstall.os, "geteuid", return_value=euid, create=True):
            return uninstall.ollama_removal_command()

    def test_macos_and_windows_use_their_package_manager(self):
        self.assertEqual(self.command("Darwin", {"brew": "/opt/homebrew/bin/brew"}), ["brew", "uninstall", "ollama"])
        self.assertEqual(self.command("Windows", {"winget": "winget"}), ["winget", "uninstall", "--id", "Ollama.Ollama", "-e", "--silent"])

    def test_without_the_package_manager_there_is_no_command(self):
        self.assertIsNone(self.command("Darwin", {}))
        self.assertIsNone(self.command("Windows", {}))

    def test_linux_is_one_sudo_line_that_ends_by_checking_it_worked(self):
        command = self.command("Linux", {"systemctl": "/bin/systemctl", "ollama": "/usr/local/bin/ollama", "sudo": "/usr/bin/sudo"})
        self.assertEqual(command[:3], ["sudo", "sh", "-c"])
        for step in ("systemctl stop ollama", "systemctl disable ollama", "/usr/local/bin/ollama", "/usr/share/ollama", "userdel ollama"):
            self.assertIn(step, command[3])
        self.assertTrue(command[3].endswith("test ! -e /usr/local/bin/ollama"))

    def test_linux_as_root_needs_no_sudo(self):
        command = self.command("Linux", {"systemctl": "/bin/systemctl", "ollama": "/usr/bin/ollama"}, euid=0)
        self.assertEqual(command[:2], ["sh", "-c"])

    def test_linux_that_does_not_look_like_the_install_script_left_it_gets_no_command(self):
        tools = {"systemctl": "/bin/systemctl", "ollama": "/usr/local/bin/ollama", "sudo": "/usr/bin/sudo"}
        self.assertIsNone(self.command("Linux", tools, service_file=False))
        self.assertIsNone(self.command("Linux", {**tools, "ollama": "/home/me/bin/ollama"}))
        self.assertIsNone(self.command("Linux", {**tools, "systemctl": None}))
        self.assertIsNone(self.command("Linux", {**tools, "sudo": None}))


class DurationTests(unittest.TestCase):
    def test_minutes_hours_and_days_join_once_a_minute_an_hour_or_a_day_is_crossed(self):
        for milliseconds, shown in ((45_000, "45 s"), (59_999, "59 s"), (60_000, "1 min 0 s"), (3_599_000, "59 min 59 s"),
                                    (3_600_000, "1 h 0 min 0 s"), (3_725_000, "1 h 2 min 5 s"),
                                    (86_399_000, "23 h 59 min 59 s"), (86_400_000, "1 d 0 h 0 min 0 s"),
                                    (183_845_000, "2 d 3 h 4 min 5 s")):
            self.assertEqual(uninstall._fmt_duration(milliseconds), shown)


class ConfirmationTests(unittest.TestCase):
    def ask(self, answers, images=12):
        with mock.patch("builtins.input", side_effect=answers), mock.patch.object(uninstall, "make_final_code", return_value=CODE), \
                contextlib.redirect_stdout(io.StringIO()):
            return uninstall.confirm_three_times({"images": images})

    def test_three_right_answers_confirm(self):
        self.assertTrue(self.ask(["yes", "12", CODE]))

    def test_each_wrong_answer_cancels(self):
        self.assertFalse(self.ask(["no"]))
        self.assertFalse(self.ask(["yes", "11"]))
        self.assertFalse(self.ask(["yes", "12", "uninstall imagent"]))

    def test_final_code_is_ten_random_letters_and_digits(self):
        codes = {uninstall.make_final_code() for _ in range(20)}
        self.assertGreater(len(codes), 1)
        for code in codes:
            self.assertEqual(len(code), 10)
            self.assertTrue(code.isalnum() and code.isascii())

    def test_no_database_means_the_count_to_type_is_zero(self):
        self.assertTrue(self.ask(["YES", "0", CODE], images=None))


class DeleteTests(IsolatedHome):
    def setUp(self):
        super().setUp()
        self.data = self.work / "Imagent"
        self.data.mkdir()
        (self.data / "imagent.db").write_bytes(b"db")
        for target, name, value in ((config, "IMAGENT_HOME", self.data), (config, "DB_PATH", self.data / "imagent.db"),
                                    (config, "CONFIG_PATH", self.data / "config.json")):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_delete(self, models=()):
        plan = uninstall.make_plan(None, {"locked_models": []})
        plan["models"] = list(models)
        with mock.patch.object(uninstall, "_call") as call, contextlib.redirect_stdout(io.StringIO()):
            problems = uninstall.delete_everything(plan, "http://x")
        return problems, call

    def test_the_data_folder_and_the_models_go(self):
        problems, call = self.run_delete(models=[("ours:latest", 5)])
        self.assertEqual(problems, [])
        self.assertFalse(self.data.exists())
        call.assert_called_once_with("http://x", "DELETE", "/api/delete", {"model": "ours:latest"}, timeout=120)

    def test_a_failed_model_removal_is_reported_and_the_rest_continues(self):
        plan = uninstall.make_plan(None, {"locked_models": []})
        plan["models"] = [("ours:latest", 5)]
        with mock.patch.object(uninstall, "_call", side_effect=OSError("down")), contextlib.redirect_stdout(io.StringIO()):
            problems = uninstall.delete_everything(plan, "http://x")
        self.assertEqual(len(problems), 1)
        self.assertIn("ollama rm ours:latest", problems[0])
        self.assertFalse(self.data.exists())

    def test_a_folder_not_named_like_imagents_is_never_deleted(self):
        other = self.work / "Documents"
        other.mkdir()
        with mock.patch.object(config, "IMAGENT_HOME", other):
            problems, _ = self.run_delete()
        self.assertTrue(other.exists())
        self.assertEqual(len(problems), 1)


class MainTests(IsolatedHome):
    def run_main(self, answers):
        stdin = mock.Mock()
        stdin.isatty.return_value = True
        with mock.patch.object(sys, "stdin", stdin), mock.patch("builtins.input", side_effect=answers), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_info:
                uninstall.main()
        return exit_info.exception.code, out.getvalue()

    def test_no_terminal_stops_before_anything(self):
        stdin = mock.Mock()
        stdin.isatty.return_value = False
        with mock.patch.object(sys, "stdin", stdin), self.assertRaisesRegex(SystemExit, "interactive terminal"):
            uninstall.main()

    def test_declining_deletes_nothing(self):
        self.home.mkdir(parents=True)
        config.DB_PATH.write_bytes(b"db")
        with mock.patch.object(uninstall, "_ollama", return_value=contextlib.nullcontext(None)):
            code, _ = self.run_main(["no"])
        self.assertNotEqual(code, 0)
        self.assertTrue(config.DB_PATH.exists())

    def test_when_it_is_done_it_says_how_to_remove_the_program_itself(self):
        data = self.work / "Imagent"
        data.mkdir()
        (data / "imagent.db").write_bytes(b"db")
        with mock.patch.object(config, "IMAGENT_HOME", data), mock.patch.object(config, "DB_PATH", data / "imagent.db"), \
                mock.patch.object(config, "CONFIG_PATH", data / "config.json"), \
                mock.patch.object(uninstall, "_ollama", return_value=contextlib.nullcontext(None)), \
                mock.patch.object(uninstall, "make_final_code", return_value=CODE):
            code, out = self.run_main(["yes", "0", CODE])
        self.assertEqual(code, 0)
        self.assertFalse(data.exists())
        self.assertIn("pipx uninstall imagent", out)

    def main_with_ollama_record(self, answers, command, installed=True, returncode=0):
        data = self.work / "Imagent"
        data.mkdir()
        (data / "imagent.db").write_bytes(b"db")
        if installed:
            (data / "install_record.json").write_text('{"ollama_installed": true}')
        with mock.patch.object(config, "IMAGENT_HOME", data), mock.patch.object(config, "DB_PATH", data / "imagent.db"), \
                mock.patch.object(config, "CONFIG_PATH", data / "config.json"), \
                mock.patch.object(uninstall, "_ollama", return_value=contextlib.nullcontext(None)), \
                mock.patch.object(uninstall, "make_final_code", return_value=CODE), \
                mock.patch.object(uninstall, "ollama_removal_command", return_value=command), \
                mock.patch.object(uninstall.subprocess, "run", return_value=SimpleNamespace(returncode=returncode)) as run:
            code, out = self.run_main(answers)
        return code, out, run

    def test_the_plan_lists_ollama_and_the_three_confirmations_cover_its_removal(self):
        code, out, run = self.main_with_ollama_record(["yes", "0", CODE], ["brew", "uninstall", "ollama"])
        self.assertEqual(code, 0)
        self.assertIn("Command: brew uninstall ollama", out)
        run.assert_called_once_with(["brew", "uninstall", "ollama"])
        self.assertIn("Removed Ollama", out)

    def test_declining_the_confirmations_leaves_ollama_alone(self):
        code, out, run = self.main_with_ollama_record(["no"], ["brew", "uninstall", "ollama"])
        self.assertNotEqual(code, 0)
        run.assert_not_called()

    def test_a_failed_removal_is_reported_with_the_command(self):
        code, out, run = self.main_with_ollama_record(["yes", "0", CODE], ["brew", "uninstall", "ollama"], returncode=1)
        self.assertEqual(code, 1)
        self.assertIn("Could not remove Ollama. Remove it yourself with: brew uninstall ollama", out)

    def test_with_no_command_for_this_computer_it_says_to_remove_ollama_by_hand(self):
        code, out, run = self.main_with_ollama_record(["yes", "0", CODE], None)
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertIn("Imagent installed Ollama itself, and it will not be removed", out)

    def test_without_the_record_ollama_is_not_mentioned(self):
        code, out, run = self.main_with_ollama_record(["yes", "0", CODE], ["brew", "uninstall", "ollama"], installed=False)
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertNotIn("Imagent installed Ollama", out)

    def test_with_nothing_to_remove_it_still_says_how_to_remove_the_program(self):
        with mock.patch.object(uninstall, "_ollama", return_value=contextlib.nullcontext(None)):
            stdin = mock.Mock()
            stdin.isatty.return_value = True
            with mock.patch.object(sys, "stdin", stdin), contextlib.redirect_stdout(io.StringIO()) as out:
                uninstall.main()
        self.assertIn("Nothing to uninstall", out.getvalue())
        self.assertIn("pipx uninstall imagent", out.getvalue())


if __name__ == "__main__":
    unittest.main()
