"""Pictures Pillow cannot open: the copy the operating system's tool makes of them, and what the user is told when none can be made."""

import contextlib
import io
import os
import unittest
from pathlib import Path
from unittest import mock

import config
from db.operations import get_or_add_root, park_backlog, record_backlog_failure, upsert_backlog
from image_processing import convert
from image_processing.convert import ConversionFailed, converted_copy
from image_processing.problems import UNSUPPORTED, ImageProblem, problem
from menu import menu
from scan_and_index import scan_and_index
from tests.support import IsolatedHome
from vlm import analyzer


def finished(returncode=0, stderr=""):
    return mock.Mock(returncode=returncode, stderr=stderr, stdout="")


class ConvertedCopyTests(unittest.TestCase):
    def tool_that_writes(self, content=b"PNG"):
        """A stand-in for subprocess.run that writes the converted file, where the command says to."""
        def run(command, **kwargs):
            target = command[command.index("--out") + 1] if "--out" in command else kwargs["env"]["IMAGENT_OUT"]
            Path(target).write_bytes(content)
            return finished()
        return run

    def test_on_a_mac_sips_makes_the_png_and_the_copy_is_deleted_afterwards(self):
        with mock.patch.object(convert.platform, "system", return_value="Darwin"), mock.patch.object(convert.shutil, "which", return_value="/usr/bin/sips"), \
                mock.patch.object(convert.subprocess, "run", side_effect=self.tool_that_writes()) as run:
            with converted_copy("/pictures/a.heic") as copy:
                self.assertTrue(os.path.isfile(copy))
                self.assertTrue(copy.endswith(".png"))
            self.assertFalse(os.path.exists(copy))
            self.assertFalse(os.path.exists(os.path.dirname(copy)))
        self.assertEqual(run.call_args[0][0][:4], ["sips", "-s", "format", "png"])

    def test_on_windows_powershell_gets_the_paths_in_the_environment_not_in_the_script(self):
        with mock.patch.object(convert.platform, "system", return_value="Windows"), mock.patch.object(convert.shutil, "which", return_value="powershell"), \
                mock.patch.object(convert.subprocess, "run", side_effect=self.tool_that_writes()) as run:
            with converted_copy("C:\\pictures\\a'b.heic") as copy:
                self.assertTrue(os.path.isfile(copy))
        self.assertEqual(run.call_args[1]["env"]["IMAGENT_IN"], os.path.abspath("C:\\pictures\\a'b.heic"))
        self.assertNotIn("a'b", run.call_args[0][0][-1])

    def test_the_copy_is_deleted_when_the_analysis_fails_too(self):
        with mock.patch.object(convert.platform, "system", return_value="Darwin"), mock.patch.object(convert.shutil, "which", return_value="/usr/bin/sips"), \
                mock.patch.object(convert.subprocess, "run", side_effect=self.tool_that_writes()):
            with self.assertRaises(ValueError):
                with converted_copy("/a.heic") as copy:
                    raise ValueError("the analysis failed")
        self.assertFalse(os.path.exists(os.path.dirname(copy)))

    def test_a_computer_without_a_tool_cannot_convert(self):
        with mock.patch.object(convert.platform, "system", return_value="Linux"), self.assertRaisesRegex(ConversionFailed, "no built-in tool"):
            with converted_copy("/a.heic"):
                pass

    def test_a_tool_that_fails_or_writes_nothing_is_a_failed_conversion_and_leaves_nothing_behind(self):
        folders = []
        original = convert.tempfile.mkdtemp

        def mkdtemp(**kwargs):
            folders.append(original(**kwargs))
            return folders[-1]
        for result in (finished(1, "cannot read it"), finished(0)):  # an error, and "success" without a file
            with mock.patch.object(convert.platform, "system", return_value="Darwin"), mock.patch.object(convert.shutil, "which", return_value="/usr/bin/sips"), \
                    mock.patch.object(convert.subprocess, "run", return_value=result), mock.patch.object(convert.tempfile, "mkdtemp", mkdtemp):
                with self.assertRaises(ConversionFailed):
                    with converted_copy("/a.heic"):
                        pass
        self.assertTrue(folders and not any(os.path.exists(f) for f in folders))


class AnalyseUnsupportedTests(unittest.TestCase):
    unsupported = staticmethod(lambda: problem(UNSUPPORTED, "Imagent cannot open .heic pictures on this computer. Convert the picture to JPEG or PNG and keep the copy in the same folder."))

    def analyse(self, converted):
        """analyse_image on a picture Pillow cannot open; ``converted`` is the copy's path, or an exception from making it."""
        seen = []

        def analyse_file(client, model, path, on_stage, on_retry):
            seen.append(path)
            if path == "/a.heic":
                raise self.unsupported()
            return analyzer.Analysis({"a": "b"}, [], {}, [])

        @contextlib.contextmanager
        def copy(path):
            if isinstance(converted, Exception):
                raise converted
            yield converted
        with mock.patch.object(analyzer, "_analyse_file", analyse_file), mock.patch.object(analyzer, "converted_copy", copy):
            try:
                return analyzer.analyse_image(object(), "vlm", "/a.heic"), seen
            except Exception as exc:
                return exc, seen

    def test_a_picture_nothing_can_open_is_analysed_from_a_converted_copy(self):
        result, seen = self.analyse("/tmp/copy.png")
        self.assertEqual(result.reply, {"a": "b"})
        self.assertEqual(seen, ["/a.heic", "/tmp/copy.png"])

    def test_when_it_cannot_be_converted_the_user_is_told_to_convert_it_and_the_reason_is_kept(self):
        result, _ = self.analyse(ConversionFailed("Linux has no built-in tool that converts pictures"))
        self.assertIsInstance(result, ImageProblem)
        self.assertIn("Convert the picture to JPEG or PNG", str(result))
        self.assertEqual(result.attempts[-1]["detail"], "Linux has no built-in tool that converts pictures")

    def test_other_problems_with_the_picture_are_not_converted(self):
        def analyse_file(*args):
            raise problem("unreadable", "damaged")
        with mock.patch.object(analyzer, "_analyse_file", analyse_file), mock.patch.object(analyzer, "converted_copy") as copy:
            with self.assertRaises(ImageProblem):
                analyzer.analyse_image(object(), "vlm", "/a.png")
        copy.assert_not_called()


class WhatTheUserIsToldTests(unittest.TestCase):
    """Every reason comes with a way to put it right, and none of them asks the user to change a model."""

    def test_no_message_suggests_changing_the_model(self):
        import inspect
        from image_processing import memory, problems
        for module in (analyzer, memory, problems, scan_and_index):
            self.assertNotRegex(inspect.getsource(module).lower(), r"another (vision )?model|different (vision )?model|change the model")


class FailedImagesReportTests(IsolatedHome):
    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        with self.conn:
            root_id = get_or_add_root(self.conn, str(self.work))
            for name in ("a.png", "b.png", "c.png"):
                upsert_backlog(self.conn, str(self.work / name), root_id, "t")

    def report(self, limit=10):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            count = scan_and_index.print_failed(self.conn, limit)
        return count, out.getvalue()

    def test_nothing_is_said_when_nothing_failed(self):
        self.assertEqual(self.report(), (0, ""))

    def test_each_image_is_listed_with_its_reason_and_what_to_do_about_it(self):
        with self.conn:
            park_backlog(self.conn, str(self.work / "a.png"), "unreadable", "The picture file is damaged or incomplete. Copy it again.", "file_changed")
            record_backlog_failure(self.conn, str(self.work / "b.png"), "Ollama stopped responding. Close other apps.", False, kind="server_crash")
        count, text = self.report()
        self.assertEqual(count, 2)
        for expected in ("2 image(s) could not be indexed", "a.png", "damaged or incomplete", "b.png", "Close other apps"):
            self.assertIn(expected, text)
        self.assertNotIn("c.png", text)

    def test_an_image_leaves_the_list_when_it_is_indexed_later(self):
        with self.conn:
            record_backlog_failure(self.conn, str(self.work / "b.png"), "Ollama stopped responding.", False, kind="server_crash")
        self.assertEqual(self.report()[0], 1)
        with self.conn:
            self.conn.execute("DELETE FROM backlog WHERE file_path = ?", (str(self.work / "b.png"),))  # what a successful index does
        self.assertEqual(self.report()[0], 0)

    def test_a_long_list_is_cut_short_at_the_end_of_a_run_but_complete_on_request(self):
        with self.conn:
            for name in ("a.png", "b.png", "c.png"):
                record_backlog_failure(self.conn, str(self.work / name), "reason", False, kind="timeout")
        self.assertIn("1 more", self.report(limit=2)[1])
        self.assertNotIn("more.", self.report(limit=None)[1])

    def test_the_menu_offers_the_list_only_when_there_is_something_to_show(self):
        for failed, offered in ((0, False), (2, True)):
            scripts = [choice.value for choice in menu.choices(False, failed)]
            self.assertEqual(menu.SHOW_FAILED_SCRIPT in scripts, offered)
            self.assertEqual(scripts[-1], menu.UNINSTALL)
        titles = [choice.title for choice in menu.choices(False, 2)]
        self.assertIn("Show the 2 image(s) that could not be indexed", titles)

    def test_the_menu_counts_what_the_report_lists(self):
        self.assertEqual(menu.failed_count(), 0)
        with self.conn:
            record_backlog_failure(self.conn, str(self.work / "b.png"), "reason", False, kind="timeout")
        self.assertEqual(menu.failed_count(), 1)


if __name__ == "__main__":
    unittest.main()
