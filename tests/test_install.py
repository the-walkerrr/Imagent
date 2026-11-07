"""Reading one pasted model set from the AI's answer: tolerant of how it is written, strict about what it holds."""

import contextlib
import io
import subprocess
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from db.operations import get_device_settings, set_device_settings
from install import install
from tests.support import DEVICE, IsolatedHome

EXPECTED = ("qwen2.5vl:7b", "bge-m3:567m", "qwen3:8b")  # (vlm, embedding, text), the order ask_models uses
PASTED = "qwen2.5vl:7b,qwen3:8b,bge-m3:567m"  # the order the AI is asked for: vision, text, embedding


class PipInstallTests(unittest.TestCase):
    def run_install(self, result=None, error=None):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(install.subprocess, "run", return_value=result, side_effect=error), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                install._pip_install("python")
                failure = None
            except SystemExit as exc:
                failure = exc
        return out.getvalue(), err.getvalue(), failure

    def test_a_finished_install_shows_the_step_as_done_and_keeps_pip_quiet(self):
        out, err, failure = self.run_install(SimpleNamespace(returncode=0, stdout="noise", stderr=""))
        self.assertIsNone(failure)
        self.assertIn("✓ Installing the required packages", out)
        self.assertNotIn("noise", out + err)

    def test_a_failed_install_shows_the_step_as_failed_then_pips_messages_then_the_way_out(self):
        out, err, failure = self.run_install(SimpleNamespace(returncode=1, stdout="", stderr="No matching distribution for x"))
        self.assertIn("✗ Installing the required packages", out)
        self.assertIn("No matching distribution for x", err)
        self.assertIn("Could not install the required packages", str(failure))

    def test_an_install_that_never_finishes_is_a_failure_too(self):
        out, _, failure = self.run_install(error=subprocess.TimeoutExpired("pip", 1))
        self.assertIn("✗ Installing the required packages", out)
        self.assertIn("Could not install the required packages", str(failure))


class ParseModelSetTests(unittest.TestCase):
    def test_one_line_comma_separated(self):
        self.assertEqual(install.parse_model_set(PASTED), EXPECTED)

    def test_decoration_spaces_and_other_separators_are_ignored(self):
        for raw in (
            f"`{PASTED}`",
            f"```{PASTED}```",
            "  qwen2.5vl:7b , qwen3:8b , bge-m3:567m  ",
            '"qwen2.5vl:7b","qwen3:8b","bge-m3:567m"',
            "qwen2.5vl:7b;qwen3:8b;bge-m3:567m",
            "qwen2.5vl:7b\nqwen3:8b\nbge-m3:567m",
            "[qwen2.5vl:7b, qwen3:8b, bge-m3:567m]",
        ):
            self.assertEqual(install.parse_model_set(raw), EXPECTED, repr(raw))

    def test_a_model_from_another_registry_or_a_user(self):
        raw = "hf.co/org/model-GGUF:Q4_K_M,user/chat:7b,nomic-embed-text"
        self.assertEqual(install.parse_model_set(raw), ("hf.co/org/model-GGUF:Q4_K_M", "nomic-embed-text", "user/chat:7b"))

    def test_the_wrong_number_of_names_is_not_readable(self):
        self.assertIsNone(install.parse_model_set("qwen2.5vl:7b,qwen3:8b"))
        self.assertIsNone(install.parse_model_set(PASTED + ",extra:1b"))
        self.assertIsNone(install.parse_model_set("qwen2.5vl:7b"))
        self.assertIsNone(install.parse_model_set(""))

    def test_none_or_a_bad_name_is_not_readable(self):
        self.assertIsNone(install.parse_model_set("NONE,qwen3:8b,bge-m3:567m"))
        self.assertIsNone(install.parse_model_set("<vision model name>:<tag>,qwen3:8b,bge-m3:567m"))
        self.assertIsNone(install.parse_model_set("qwen2.5vl:7b,,bge-m3:567m"))
        self.assertIsNone(install.parse_model_set("Set 1: qwen2.5vl:7b,qwen3:8b,bge-m3:567m"))  # text around it


class ModelScreenTests(unittest.TestCase):
    """ask_models is one screen of editable rows: names are typed in place, and the paste row exists only after the prompt was copied."""

    DOWN, ENTER, CLEAR_LINE = "\x1b[B", "\r", "\x15"

    def run_screen(self, keys, *args, copied=True, extra_keys_after_failed_copy=""):
        """Run ask_models with these keystrokes; a watchdog presses Ctrl+C so a stuck screen fails instead of hanging."""
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            def offer():
                pipe.send_text(extra_keys_after_failed_copy)
                return False
            watchdog = threading.Timer(10, lambda: pipe.send_text("\x03"))
            watchdog.start()
            try:
                with mock.patch.object(install, "say"), \
                        mock.patch.object(install, "build_ai_prompt", return_value="prompt"), \
                        mock.patch.object(install, "copy_to_clipboard", return_value=copied), \
                        mock.patch.object(install, "offer_prompt_copy", side_effect=offer) as offer_copy:
                    pipe.send_text(keys)
                    self.offer_copy = offer_copy
                    return install.ask_models(*args)
            finally:
                watchdog.cancel()

    def test_names_are_typed_in_the_rows_in_order(self):
        keys = "a:1\rb:2\rc:3\r" + self.DOWN + self.ENTER  # then Down from "Ask AI" to Continue
        self.assertEqual(self.run_screen(keys), ("a:1", "b:2", "c:3"))

    def test_names_that_are_already_set_are_kept_and_continue_is_where_you_start(self):
        self.assertEqual(self.run_screen(self.ENTER, "a:1", "b:2", "c:3"), ("a:1", "b:2", "c:3"))

    def test_a_pasted_line_with_a_label_is_cleaned(self):
        keys = "VLM_1=`a:1`\rb:2\rc:3\r" + self.DOWN + self.ENTER
        self.assertEqual(self.run_screen(keys), ("a:1", "b:2", "c:3"))

    def test_a_bad_name_keeps_you_in_the_row_until_it_is_fixed_or_cleared(self):
        keys = "bad name\r" + self.CLEAR_LINE + "a:1\rb:2\rc:3\r" + self.DOWN + self.ENTER
        self.assertEqual(self.run_screen(keys), ("a:1", "b:2", "c:3"))

    def test_continue_does_nothing_until_all_three_names_are_set(self):
        keys = (
            "\r\r" + "c:3\r"            # skip the VLM and embedding rows, fill the text row (focus ends on Ask AI)
            + self.DOWN + self.ENTER       # Continue, pressed too early: nothing happens
            + self.DOWN + "a:1\rb:2\r"     # wrap to the VLM row, fill it and the embedding row (focus ends on the text row)
            + self.DOWN * 2 + self.ENTER   # Ask AI, Continue
        )
        self.assertEqual(self.run_screen(keys), ("a:1", "b:2", "c:3"))

    def test_asking_the_ai_adds_the_paste_row_and_a_good_paste_fills_all_three(self):
        keys = self.DOWN * 3 + self.ENTER + "a:1,b:2,c:3\r" + self.ENTER  # Ask AI, paste, Continue
        self.assertEqual(self.run_screen(keys), ("a:1", "c:3", "b:2"))  # pasted as vision, text, embedding

    def test_an_unreadable_paste_is_cleared_and_the_row_stays(self):
        keys = self.DOWN * 3 + self.ENTER + "garbage\r" + "a:1,b:2,c:3\r" + self.ENTER
        self.assertEqual(self.run_screen(keys), ("a:1", "c:3", "b:2"))

    def test_without_a_clipboard_the_prompt_is_printed_and_the_screen_comes_back_with_the_paste_row(self):
        result = self.run_screen(self.DOWN * 3 + self.ENTER, copied=False, extra_keys_after_failed_copy="a:1,b:2,c:3\r\r")
        self.assertEqual(result, ("a:1", "c:3", "b:2"))
        self.offer_copy.assert_called_once()

    def test_ctrl_c_cancels_the_setup(self):
        with self.assertRaises(SystemExit):
            self.run_screen("\x03")


if __name__ == "__main__":
    unittest.main()


class AiPromptTests(unittest.TestCase):
    prompt = None

    @classmethod
    def setUpClass(cls):
        with mock.patch.object(install, "system_details", return_value="- RAM: 8 GB"):
            cls.prompt = install.build_ai_prompt()

    def test_the_embedding_model_is_counted_with_the_vision_model_when_indexing(self):
        indexing = next(line for line in self.prompt.splitlines() if line.strip().startswith("- indexing:"))
        self.assertIn("vision model AND the embedding model are loaded at the same time", indexing)
        self.assertIn("compute buffers", indexing)  # the embedding model takes more memory than its weight file

    def test_the_embedding_model_is_counted_with_the_text_model_when_searching(self):
        searching = next(line for line in self.prompt.splitlines() if line.strip().startswith("- searching:"))
        self.assertIn("text model AND the embedding model are loaded at the same time", searching)

    def test_shared_memory_is_not_taken_to_be_all_usable_by_the_gpu(self):
        self.assertIn('if "GPU memory usable" is listed above, it is the figure to use', self.prompt)
        self.assertIn("do not look it up", self.prompt)

    def test_the_prompt_lets_the_model_estimate_memory_instead_of_declining_a_set(self):
        self.assertIn("estimate them and leave a safety margin", self.prompt)
        self.assertIn("verified the tag and capabilities", self.prompt)


class GpuMemoryDetailTests(unittest.TestCase):
    def details(self, metal_gb):
        with mock.patch.object(install, "_metal_gpu_memory_gb", return_value=metal_gb):
            return install.system_details()

    def test_the_gpu_memory_metal_reports_is_listed(self):
        self.assertIn("- GPU memory usable (reported by Metal for this computer): 5.3 GB", self.details(5.33))

    def test_no_gpu_memory_line_when_metal_cannot_be_asked(self):
        self.assertNotIn("GPU memory usable", self.details(None))

    def test_metal_is_not_asked_outside_macos(self):
        with mock.patch.object(install.platform, "system", return_value="Linux"):
            self.assertIsNone(install._metal_gpu_memory_gb())


class MeasureAgainTests(IsolatedHome):
    """Setup run again: when the computer is measured again."""
    settings = {"vlm_model": "vlm", "embedding_model": "emb"}

    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        self.client = SimpleNamespace(list=lambda: SimpleNamespace(models=[SimpleNamespace(model="vlm", digest="d1")]))
        self.measured = {**DEVICE, "num_ctx": 4096, "one_model_at_a_time": True}
        patchers = (mock.patch.object(install, "ollama_client", return_value=contextlib.nullcontext(self.client)),
                    mock.patch.object(install, "measure_computer", return_value=(self.measured, None)))
        self.measure = [patcher.start() for patcher in patchers][1]
        for patcher in patchers:
            self.addCleanup(patcher.stop)

    def store(self, digest="d1", apart=False):
        with self.conn:
            set_device_settings(self.conn, {**DEVICE, "one_model_at_a_time": apart}, digest)

    def run_it(self):
        install.measure_again_if_needed(self.settings, "http://localhost:11434")

    def test_a_computer_measured_with_this_vision_model_and_both_models_together_is_not_measured_again(self):
        for apart in (False, True):
            self.store(apart=apart)
            self.run_it()
        self.measure.assert_not_called()

    def test_a_computer_measured_before_the_two_models_were_tried_together_is_measured_again_with_both(self):
        self.store(apart=None)
        self.run_it()
        self.measure.assert_called_once_with(self.client, "vlm", "emb")
        found = get_device_settings(self.conn)
        self.assertEqual((found["num_ctx"], found["one_model_at_a_time"]), (4096, True))

    def test_a_changed_vision_model_is_measured_again(self):
        self.store(digest="old")
        self.run_it()
        self.measure.assert_called_once()
