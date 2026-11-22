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

    def test_setup_uses_the_ollama_that_is_there_and_installs_nothing(self):
        for version in ((0, 9, 0), None):  # None: an Ollama that cannot say its version is trusted
            with mock.patch.object(setup, "say"), mock.patch.object(setup, "install_ollama") as install, \
                    mock.patch.object(setup.shutil, "which", return_value="/usr/bin/ollama"), \
                    mock.patch.object(setup, "_ollama_command_version", return_value=version):
                setup.ensure_ollama_installed()
            install.assert_not_called()
        self.assertEqual(config.installed_tools(), [])

    def test_an_ollama_older_than_imagent_needs_is_left_alone_and_imagent_installs_its_own(self):
        with mock.patch.object(setup, "say") as say, mock.patch.object(setup, "install_ollama") as install, \
                mock.patch.object(setup.shutil, "which", return_value="/usr/bin/ollama"), \
                mock.patch.object(setup, "_ollama_command_version", return_value=(0, 4, 0)):
            setup.ensure_ollama_installed()
        install.assert_called_once_with()
        self.assertIn("leaves yours as it is", say.call_args.args[0])

    def test_no_ollama_at_all_installs_imagents_own(self):
        with mock.patch.object(setup, "say"), mock.patch.object(setup, "install_ollama") as install, \
                mock.patch.object(setup.shutil, "which", return_value=None):
            setup.ensure_ollama_installed()
        install.assert_called_once_with()

    def test_imagents_own_copy_is_used_again_without_installing_or_looking_at_the_system(self):
        with mock.patch.object(setup, "say"), mock.patch.object(setup, "install_ollama") as install, \
                mock.patch.object(config, "private_ollama_binary", return_value="/tools/ollama"), \
                mock.patch.object(setup.shutil, "which") as which:
            setup.ensure_ollama_installed()
        install.assert_not_called()
        which.assert_not_called()

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


    def test_the_tools_setup_installed_are_part_of_the_plan(self):
        (self.tools / "ollama").mkdir(parents=True)
        config.save_tool_info(self.tools / "ollama", "Ollama", "Runs the AI models.", "0.40.17")
        (self.tools / "stray").mkdir()  # a folder without the info file is not Imagent's to offer
        with mock.patch.object(uninstall, "_installed_models", return_value={}):
            plan = uninstall.make_plan(None, {"locked_models": []})
        self.assertEqual([(tool["name"], tool["version"], tool["path"]) for tool in plan["tools"]],
                         [("Ollama", "0.40.17", self.tools / "ollama")])


CODE = "aB3dE5gH7j"


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

    def test_the_hidden_default_folder_name_is_deleted(self):
        hidden = self.work / ".imagent"
        hidden.mkdir()
        (hidden / "imagent.db").write_bytes(b"db")
        with mock.patch.object(config, "IMAGENT_HOME", hidden):
            problems, _ = self.run_delete()
        self.assertEqual(problems, [])
        self.assertFalse(hidden.exists())

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
        with mock.patch.object(sys, "stdin", stdin), mock.patch("builtins.input", side_effect=answers) as asked, \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_info:
                uninstall.main()
        self.questions = [call.args[0] for call in asked.call_args_list]  # input() prints its question itself, so it is not in ``out``
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

    def main_with_a_tool(self, answers, installed=True):
        """Run main() with Imagent's data and, when ``installed``, its own Ollama (with a file in it, like the real one has)."""
        data = self.work / "Imagent"
        data.mkdir()
        (data / "imagent.db").write_bytes(b"db")
        if installed:
            (self.tools / "ollama" / "program").mkdir(parents=True)
            (self.tools / "ollama" / "program" / "ollama").write_bytes(b"x" * 2000)
            config.save_tool_info(self.tools / "ollama", "Ollama", "Runs the AI models.", "0.40.17")
        with mock.patch.object(config, "IMAGENT_HOME", data), mock.patch.object(config, "DB_PATH", data / "imagent.db"), \
                mock.patch.object(config, "CONFIG_PATH", data / "config.json"), \
                mock.patch.object(uninstall, "_ollama", return_value=contextlib.nullcontext(None)), \
                mock.patch.object(uninstall, "make_final_code", return_value=CODE):
            return self.run_main(answers)

    def test_each_tool_is_described_and_asked_about_after_the_data_is_gone(self):
        code, out = self.main_with_a_tool(["yes", "0", CODE, "y"])
        self.assertEqual(code, 0)
        self.assertIn("Ollama 0.40.17: Runs the AI models.", out)
        self.assertEqual(len(self.questions), 4)  # three confirmations, then the question about Ollama
        self.assertIn("Remove Ollama too? [y/N]", self.questions[-1])
        self.assertLess(out.index("Deleted the data folder"), out.index("Ollama 0.40.17"))
        self.assertIn("Removed Ollama", out)
        self.assertFalse(self.tools.exists())  # the folder and the then-empty tools folder

    def test_the_default_answer_keeps_the_tool_and_says_where_it_is(self):
        code, out = self.main_with_a_tool(["yes", "0", CODE, ""])
        self.assertEqual(code, 0)
        self.assertTrue((self.tools / "ollama" / "program" / "ollama").exists())
        self.assertIn(f"Ollama was kept. To remove it later, delete the folder {self.tools / 'ollama'}", out)

    def test_declining_the_confirmations_never_gets_as_far_as_the_tools(self):
        code, out = self.main_with_a_tool(["no"])
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.questions), 1)  # the first confirmation, and no more
        self.assertTrue((self.tools / "ollama").exists())

    def test_without_a_tool_nothing_is_asked(self):
        code, out = self.main_with_a_tool(["yes", "0", CODE], installed=False)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.questions), 3)

    def test_a_kept_tool_is_offered_again_by_a_later_uninstall_even_with_no_data_left(self):
        (self.tools / "ollama").mkdir(parents=True)
        config.save_tool_info(self.tools / "ollama", "Ollama", "Runs the AI models.", "0.40.17")
        code, out = self.run_main_with_nothing_else(["y"])
        self.assertEqual(code, 0)
        self.assertIn("already removed", out)
        self.assertIn("Removed Ollama", out)
        self.assertFalse(self.tools.exists())

    def run_main_with_nothing_else(self, answers):
        with mock.patch.object(uninstall, "_ollama", return_value=contextlib.nullcontext(None)):
            return self.run_main(answers)

    def test_a_tool_whose_folder_is_not_in_the_tools_folder_is_never_deleted(self):
        elsewhere = self.work / "Documents"
        elsewhere.mkdir()
        (elsewhere / "keep.txt").write_text("mine")
        tool = {"name": "Ollama", "description": "", "version": "", "path": elsewhere}
        with mock.patch("builtins.input", return_value="y"), contextlib.redirect_stdout(io.StringIO()):
            problems = uninstall.offer_tool_removal([tool])
        self.assertTrue((elsewhere / "keep.txt").exists())
        self.assertEqual(len(problems), 1)

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
