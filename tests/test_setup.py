"""Reading one pasted model set from the AI's answer: tolerant of how it is written, strict about what it holds."""

import contextlib
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from imagent import config
from imagent.db.operations import get_device_settings, set_device_settings
from imagent.commands import setup
from tests.support import DEVICE, IsolatedHome

EXPECTED = ("qwen2.5vl:7b", "bge-m3:567m", "qwen3:8b")  # (vlm, embedding, text), the order ask_models uses
PASTED = "qwen2.5vl:7b,qwen3:8b,bge-m3:567m"  # the order the AI is asked for: vision, text, embedding


class ParseModelSetTests(unittest.TestCase):
    def test_one_line_comma_separated(self):
        self.assertEqual(setup.parse_model_set(PASTED), EXPECTED)

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
            self.assertEqual(setup.parse_model_set(raw), EXPECTED, repr(raw))

    def test_a_model_from_another_registry_or_a_user(self):
        raw = "hf.co/org/model-GGUF:Q4_K_M,user/chat:7b,nomic-embed-text"
        self.assertEqual(setup.parse_model_set(raw), ("hf.co/org/model-GGUF:Q4_K_M", "nomic-embed-text", "user/chat:7b"))

    def test_the_wrong_number_of_names_is_not_readable(self):
        self.assertIsNone(setup.parse_model_set("qwen2.5vl:7b,qwen3:8b"))
        self.assertIsNone(setup.parse_model_set(PASTED + ",extra:1b"))
        self.assertIsNone(setup.parse_model_set("qwen2.5vl:7b"))
        self.assertIsNone(setup.parse_model_set(""))

    def test_none_or_a_bad_name_is_not_readable(self):
        self.assertIsNone(setup.parse_model_set("NONE,qwen3:8b,bge-m3:567m"))
        self.assertIsNone(setup.parse_model_set("<vision model name>:<tag>,qwen3:8b,bge-m3:567m"))
        self.assertIsNone(setup.parse_model_set("qwen2.5vl:7b,,bge-m3:567m"))
        self.assertIsNone(setup.parse_model_set("Set 1: qwen2.5vl:7b,qwen3:8b,bge-m3:567m"))  # text around it


class ModelScreenTests(unittest.TestCase):
    """ask_models is one screen of editable rows: names are typed in place, and the paste row exists only after the prompt was copied."""

    DOWN, ENTER, CLEAR_LINE = "\x1b[B", "\r", "\x15"

    def run_screen(self, keys, *args, copied=True, extra_keys_after_failed_copy=""):
        """Run ask_models with these keystrokes; a watchdog presses Ctrl+C so a stuck screen fails instead of hanging."""
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            def offer(*args):
                pipe.send_text(extra_keys_after_failed_copy)
                return False
            watchdog = threading.Timer(10, lambda: pipe.send_text("\x03"))
            watchdog.start()
            try:
                with mock.patch.object(setup, "say"), \
                        mock.patch.object(setup, "build_ai_prompt", return_value="prompt"), \
                        mock.patch.object(setup, "copy_to_clipboard", return_value=copied), \
                        mock.patch.object(setup, "offer_prompt_copy", side_effect=offer) as offer_copy:
                    pipe.send_text(keys)
                    self.offer_copy = offer_copy
                    return setup.ask_models(*args)
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
        keys = self.DOWN * 3 + self.ENTER + "a:1,b:2,c:3\r"  # Ask AI, paste: no separate Continue
        self.assertEqual(self.run_screen(keys), ("a:1", "c:3", "b:2"))  # pasted as vision, text, embedding

    def test_an_unreadable_paste_is_cleared_and_the_row_stays(self):
        keys = self.DOWN * 3 + self.ENTER + "garbage\r" + "a:1,b:2,c:3\r"
        self.assertEqual(self.run_screen(keys), ("a:1", "c:3", "b:2"))

    def test_without_a_clipboard_the_prompt_is_printed_and_the_screen_comes_back_with_the_paste_row(self):
        result = self.run_screen(self.DOWN * 3 + self.ENTER, copied=False, extra_keys_after_failed_copy="a:1,b:2,c:3\r")
        self.assertEqual(result, ("a:1", "c:3", "b:2"))
        self.offer_copy.assert_called_once()

    def test_the_models_that_failed_here_are_passed_on_to_the_prompt(self):
        failed = {"big:70b": "did not work on this computer."}
        self.run_screen(self.DOWN * 3 + self.ENTER, "", "", "", failed, copied=False, extra_keys_after_failed_copy="a:1,b:2,c:3\r")
        self.offer_copy.assert_called_once_with(failed)

    def test_ctrl_c_cancels_the_setup(self):
        with self.assertRaises(SystemExit):
            self.run_screen("\x03")


if __name__ == "__main__":
    unittest.main()


class AiPromptTests(unittest.TestCase):
    prompt = None

    @classmethod
    def setUpClass(cls):
        with mock.patch.object(setup, "system_details", return_value="- RAM: 8 GB"):
            cls.prompt = setup.build_ai_prompt()

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

    def test_memory_is_a_limit_and_not_a_reason_to_choose_big_models(self):
        self.assertIn("not the largest that fit", self.prompt)
        self.assertIn("only an upper limit, not a target", self.prompt)
        self.assertIn("Do not choose a model because it is the largest that fits", self.prompt)

    def test_the_text_model_is_the_vision_model_by_default(self):
        self.assertIn("By default the text model is the same model as the vision model", self.prompt)
        self.assertIn("do not give a larger text model than the vision model", self.prompt)

    def test_a_set_with_the_same_model_twice_is_accepted_by_the_parser(self):
        self.assertIn("write its name twice", self.prompt)
        self.assertEqual(setup.parse_model_set("qwen2.5vl:7b,qwen2.5vl:7b,bge-m3:567m"),
                         ("qwen2.5vl:7b", "bge-m3:567m", "qwen2.5vl:7b"))


class GpuMemoryDetailTests(unittest.TestCase):
    def details(self, metal_gb):
        with mock.patch.object(setup, "_metal_gpu_memory_gb", return_value=metal_gb):
            return setup.system_details()

    def test_the_gpu_memory_metal_reports_is_listed(self):
        self.assertIn("- GPU memory usable (reported by Metal for this computer): 5.3 GB", self.details(5.33))

    def test_no_gpu_memory_line_when_metal_cannot_be_asked(self):
        self.assertNotIn("GPU memory usable", self.details(None))

    def test_metal_is_not_asked_outside_macos(self):
        with mock.patch.object(setup.platform, "system", return_value="Linux"):
            self.assertIsNone(setup._metal_gpu_memory_gb())


class MeasureAgainTests(IsolatedHome):
    """Setup run again: when the computer is measured again."""
    settings = {"vlm_model": "vlm", "embedding_model": "emb"}

    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        self.client = SimpleNamespace(list=lambda: SimpleNamespace(models=[SimpleNamespace(model="vlm", digest="d1")]))
        self.measured = {**DEVICE, "num_ctx": 4096, "one_model_at_a_time": True}
        patchers = (mock.patch.object(setup, "ollama_client", return_value=contextlib.nullcontext(self.client)),
                    mock.patch.object(setup, "measure_computer", return_value=(self.measured, None)))
        self.measure = [patcher.start() for patcher in patchers][1]
        for patcher in patchers:
            self.addCleanup(patcher.stop)

    def store(self, digest="d1", apart=False):
        with self.conn:
            set_device_settings(self.conn, {**DEVICE, "one_model_at_a_time": apart}, digest)

    def run_it(self):
        setup.measure_again_if_needed(self.settings, "http://localhost:11434")

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

    def test_a_changed_vision_model_is_not_measured_again_because_the_models_are_locked(self):
        self.store(digest="old")
        self.run_it()
        self.measure.assert_not_called()


class FailedModelsInThePromptTests(unittest.TestCase):
    def prompt(self, failed=None):
        with mock.patch.object(setup, "system_details", return_value="- RAM: 8 GB"):
            return setup.build_ai_prompt(failed)

    def test_without_a_failure_the_prompt_does_not_mention_one(self):
        self.assertNotIn("already failed", self.prompt())
        self.assertNotIn("already failed", self.prompt({}))

    def test_each_model_that_failed_is_listed_with_why_and_is_not_to_be_suggested_again(self):
        prompt = self.prompt({"qwen2.5vl:72b": "did not work on this computer at any picture size worth using (last problem: Ollama ran\nout of memory).",
                              "x:1b": "is not a vision model."})
        self.assertIn("## Models that already failed on this computer", prompt)
        self.assertIn("- qwen2.5vl:72b did not work on this computer at any picture size worth using (last problem: Ollama ran out of memory).", prompt)
        self.assertIn("- x:1b is not a vision model.", prompt)
        self.assertIn("Do not suggest any of them", prompt)
        self.assertLess(prompt.index("already failed"), prompt.index("## The three models"))  # the AI reads it before it chooses

    def test_the_prompt_copied_to_the_clipboard_has_the_failures(self):
        with mock.patch.object(setup, "copy_to_clipboard", return_value=True) as copy, mock.patch.object(setup, "say"):
            with mock.patch.object(setup, "system_details", return_value="- RAM: 8 GB"):
                setup.offer_prompt_copy({"x:1b": "is not a vision model."})
        self.assertIn("- x:1b is not a vision model.", copy.call_args.args[0])


class FakeOllama:
    """A server that downloads, copies and deletes models in memory, and says what each can do by its first letter: v(ision), e(mbedding), t(ext)."""

    def __init__(self, installed=()):
        self.models = {name: f"sha256:{name.replace(':', '')}{'0' * 20}" for name in installed}
        self.deleted = []

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model=name, digest=digest, size=1) for name, digest in self.models.items()])

    def pull(self, name, stream=True):
        self.models[name] = f"sha256:{name.replace(':', '')}{'0' * 20}"
        return iter([])

    def copy(self, source, target):
        self.models[target] = self.models[source]

    def delete(self, name):
        self.deleted.append(name)
        del self.models[name]

    def show(self, name):
        kind = name.split("/")[-1][0]
        return SimpleNamespace(capabilities={"v": ["completion", "vision"], "e": ["embedding"], "t": ["completion"]}[kind])

    def embed(self, **request):
        return SimpleNamespace(embeddings=[[0.0, 0.0, 0.0]])


class WhenTheComputerCannotRunAModelTests(IsolatedHome):
    """Setup measured the computer and a model did not work: which model is blamed, what is taken off the disk, and what is asked next."""

    def setUp(self):
        super().setUp()
        self.server = FakeOllama()
        patchers = (mock.patch.object(setup, "ollama_client", side_effect=lambda host: contextlib.nullcontext(self.server)),
                    mock.patch.object(setup, "_pull_with_progress", side_effect=lambda client, name: client.pull(name)),  # no progress bar
                    mock.patch.object(setup, "say"))
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_it(self, failing, vlm="v:1b", embedding="e:1b", text="t:1b"):
        """Download the three models, then measure with ``failing`` as the model that does not run ('vlm' or 'embedding')."""
        def measure(client, vlm_model, embedding_model):
            return None, {vlm_model if failing == "vlm" else embedding_model: "did not work on this computer."}
        with mock.patch.object(setup, "measure_computer", side_effect=measure):
            return setup.download_and_verify("http://localhost:11434", vlm, embedding, text)

    def record(self):
        return config.load_install_record()

    def test_the_failing_model_is_named_as_typed_so_that_only_its_row_is_asked_again(self):
        settings, problems = self.run_it("vlm")
        self.assertIsNone(settings)
        self.assertEqual(problems, {"v:1b": "did not work on this computer."})

    def test_a_failing_embedding_model_is_not_blamed_on_the_vision_model(self):
        settings, problems = self.run_it("embedding")
        self.assertIsNone(settings)
        self.assertEqual(problems, {"e:1b": "did not work on this computer."})

    def test_the_failing_model_and_its_fixed_copy_are_removed_and_the_others_stay(self):
        self.run_it("vlm")
        self.assertEqual(sorted(name for name in self.server.models if not name.startswith("imagent-pin/")), ["e:1b", "t:1b"])
        self.assertFalse(any(name.startswith("imagent-pin/v") for name in self.server.models))
        self.assertEqual(len(self.server.deleted), 2)  # the model and its copy
        self.assertTrue(any(name.startswith("imagent-pin/e") for name in self.server.models))

    def test_the_record_the_uninstaller_uses_forgets_what_was_removed(self):
        self.run_it("vlm")
        record = self.record()
        self.assertNotIn("v:1b", record["ollama_models"])
        self.assertIn("e:1b", record["ollama_models"])
        self.assertFalse(any(name.startswith("imagent-pin/v") for name in record["pinned_models"]))
        self.assertTrue(any(name.startswith("imagent-pin/e") for name in record["pinned_models"]))

    def test_a_model_that_was_on_the_computer_before_is_never_removed_only_the_copy_made_for_it(self):
        self.server = FakeOllama(installed=["v:1b"])
        self.run_it("vlm")
        self.assertIn("v:1b", self.server.models)
        self.assertEqual(len(self.server.deleted), 1)
        self.assertTrue(self.server.deleted[0].startswith("imagent-pin/v"))

    def test_the_models_that_work_stay_even_when_one_serves_two_roles(self):
        self.run_it("embedding", vlm="v:1b", embedding="e:1b", text="v:1b")  # the vision model also serves as the text model
        self.assertIn("v:1b", self.server.models)
        self.assertNotIn("e:1b", self.server.models)

    def test_a_model_that_cannot_be_removed_is_said_so_and_does_not_stop_setup(self):
        self.server.delete = mock.Mock(side_effect=RuntimeError("busy"))
        settings, problems = self.run_it("vlm")
        self.assertIsNone(settings)
        self.assertEqual(list(problems), ["v:1b"])
        self.assertIn("v:1b", self.record()["ollama_models"])  # still there, so still on the uninstaller's list

    def test_a_computer_that_manages_every_model_removes_nothing(self):
        with mock.patch.object(setup, "measure_computer", return_value=({**DEVICE}, None)):
            settings, problems = setup.download_and_verify("http://localhost:11434", "v:1b", "e:1b", "t:1b")
        self.assertEqual(problems, {})
        self.assertEqual(settings["device"], DEVICE)
        self.assertEqual(self.server.deleted, [])


class ChoosingAgainAfterAFailureTests(unittest.TestCase):
    def test_the_next_prompt_knows_every_model_that_failed_and_only_the_failed_row_is_asked_again(self):
        asked = []

        def ask(vlm, embedding, text, failed):
            asked.append((vlm, embedding, text, dict(failed)))
            return {1: ("v:70b", "e:1b", "t:1b"), 2: ("v:1b", "e:1b", "t:1b"), 3: ("v:1b", "e:2b", "t:1b")}[len(asked)]

        outcomes = iter([(None, {"v:70b": "did not work."}), (None, {"e:1b": "did not work either."}), ({"vlm_model": "v:1b"}, {})])
        with mock.patch.object(setup, "ask_models", side_effect=ask), \
                mock.patch.object(setup, "review_models", return_value=(True, set())), \
                mock.patch.object(setup, "download_and_verify", side_effect=lambda *args: next(outcomes)), \
                mock.patch.object(setup, "step"), mock.patch.object(setup, "say"):
            settings, host = setup.choose_models("http://localhost:11434")
        self.assertEqual(settings, {"vlm_model": "v:1b"})
        self.assertEqual(asked[0], ("", "", "", {}))
        self.assertEqual(asked[1], ("", "e:1b", "t:1b", {"v:70b": "did not work."}))  # the vision model's row is cleared
        self.assertEqual(asked[2], ("v:1b", "", "t:1b", {"v:70b": "did not work.", "e:1b": "did not work either."}))


class MeasureAgainFailureTests(IsolatedHome):
    def test_the_message_names_the_model_that_failed_and_not_always_the_vision_model(self):
        conn = self.connect()
        client = SimpleNamespace(list=lambda: SimpleNamespace(models=[SimpleNamespace(model="vlm", digest="d1")]))
        with mock.patch.object(setup, "ollama_client", return_value=contextlib.nullcontext(client)), \
                mock.patch.object(setup, "measure_computer", return_value=(None, {"emb": "did not work on this computer."})):
            with self.assertRaises(SystemExit) as raised:
                setup.measure_again_if_needed({"vlm_model": "vlm", "embedding_model": "emb"}, "http://localhost:11434")
        self.assertIn("'emb' did not work on this computer.", str(raised.exception))
        self.assertIsNone(get_device_settings(conn))


class MeasureComputerTests(unittest.TestCase):
    def test_the_model_that_cannot_run_is_the_one_named(self):
        from imagent.vlm.calibration import NotPossible
        with mock.patch("imagent.vlm.calibration.calibrate", side_effect=NotPossible("did not work.", "emb")), mock.patch.object(setup, "say"):
            self.assertEqual(setup.measure_computer(object(), "vlm", "emb"), (None, {"emb": "did not work."}))

    def test_trouble_without_a_named_model_is_put_on_the_vision_model(self):
        with mock.patch("imagent.vlm.calibration.calibrate", side_effect=RuntimeError("boom")), mock.patch.object(setup, "say"):
            device, problems = setup.measure_computer(object(), "vlm", "emb")
        self.assertIsNone(device)
        self.assertEqual(list(problems), ["vlm"])
        self.assertIn("RuntimeError: boom", problems["vlm"])
