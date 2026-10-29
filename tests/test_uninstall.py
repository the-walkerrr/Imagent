import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import config
from install import install
from db.operations import get_or_add_root
from uninstall import uninstall
from tests.support import IsolatedHome, add_indexed


def _model(name):
    return SimpleNamespace(model=name)


class InstallRecordTests(IsolatedHome):
    def test_a_model_is_recorded_once(self):
        install._remember_pulled("vision")
        install._remember_pulled("vision:latest")
        self.assertEqual(config.load_install_record(), {"ollama_models": ["vision:latest"]})

    def test_only_models_that_were_downloaded_are_recorded(self):
        client = mock.Mock()
        client.list.return_value = SimpleNamespace(models=[_model("had-before:latest")])
        client.pull.return_value = iter([SimpleNamespace(status="done", total=None, completed=None)])
        with contextlib.redirect_stdout(io.StringIO()):
            install._pull_models(client, ["had-before", "new-one:1b"])
        self.assertEqual(config.load_install_record(), {"ollama_models": ["new-one:1b"]})

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


class ConfirmationTests(unittest.TestCase):
    def ask(self, answers, images=12):
        with mock.patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(io.StringIO()):
            return uninstall.confirm_three_times({"images": images})

    def test_three_right_answers_confirm(self):
        self.assertTrue(self.ask(["yes", "12", uninstall.FINAL_PHRASE]))

    def test_each_wrong_answer_cancels(self):
        self.assertFalse(self.ask(["no"]))
        self.assertFalse(self.ask(["yes", "11"]))
        self.assertFalse(self.ask(["yes", "12", "uninstall"]))

    def test_no_database_means_the_count_to_type_is_zero(self):
        self.assertTrue(self.ask(["YES", "0", uninstall.FINAL_PHRASE], images=None))


class DeleteTests(IsolatedHome):
    def setUp(self):
        super().setUp()
        self.project = self.work / "project"
        (self.project / "pkg" / "__pycache__").mkdir(parents=True)
        (self.project / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"x")
        (self.project / "pkg" / "code.py").write_text("print(1)")
        (self.project / ".venv" / "lib").mkdir(parents=True)
        (self.project / ".venv" / "lib" / "pkg.py").write_text("x")
        self.data = self.work / "Imagent"
        self.data.mkdir()
        (self.data / "imagent.db").write_bytes(b"db")
        for target, name, value in ((uninstall, "PROJECT_DIR", self.project), (uninstall, "VENV_DIR", self.project / ".venv"),
                                    (config, "IMAGENT_HOME", self.data), (config, "DB_PATH", self.data / "imagent.db"),
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

    def test_everything_added_goes_and_the_source_stays(self):
        problems, call = self.run_delete(models=[("ours:latest", 5)])
        self.assertEqual(problems, [])
        self.assertFalse(self.data.exists())
        self.assertFalse((self.project / ".venv").exists())
        self.assertFalse((self.project / "pkg" / "__pycache__").exists())
        self.assertTrue((self.project / "pkg" / "code.py").exists())
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

    def test_it_refuses_to_delete_the_environment_it_is_running_in(self):
        venv = Path(sys.prefix)
        with mock.patch.object(uninstall, "VENV_DIR", venv), mock.patch.object(sys, "stdin") as stdin:
            stdin.isatty.return_value = True
            with self.assertRaisesRegex(SystemExit, "deactivate"):
                uninstall.main()

    def test_declining_deletes_nothing(self):
        self.home.mkdir(parents=True)
        config.DB_PATH.write_bytes(b"db")
        with mock.patch.object(uninstall, "_ollama", return_value=contextlib.nullcontext(None)), \
                mock.patch.object(uninstall, "VENV_DIR", self.work / "no-venv"):
            code, _ = self.run_main(["no"])
        self.assertNotEqual(code, 0)
        self.assertTrue(config.DB_PATH.exists())


if __name__ == "__main__":
    unittest.main()
