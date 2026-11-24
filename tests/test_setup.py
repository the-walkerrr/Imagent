"""Reading one pasted model set from the AI's answer: tolerant of how it is written, strict about what it holds."""

import contextlib
import hashlib
import io
import sys
import tarfile
import threading
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from imagent import config
from imagent.db.operations import get_device_settings, get_keep_deleted, set_device_settings
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

    def test_names_that_are_already_set_are_kept_and_the_first_row_is_where_you_start(self):
        keys = self.DOWN * 4 + self.ENTER  # from the VLM row: embedding, text, Ask AI, Continue
        self.assertEqual(self.run_screen(keys, "a:1", "b:2", "c:3"), ("a:1", "b:2", "c:3"))

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


class ChoosingAgainAfterAFailureTests(IsolatedHome):
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


class ResumingSetupTests(IsolatedHome):
    """Setup that stopped part way shows the confirmed names again; it still asks, and never downloads by itself."""

    def choose(self, ask_returns, review=(True, set()), outcome=({"vlm_model": "v:1b"}, {})):
        self.asked = []

        def ask(vlm, embedding, text, failed):
            self.asked.append((vlm, embedding, text))
            return ask_returns

        patches = [mock.patch.object(setup, "ask_models", side_effect=ask),
                   mock.patch.object(setup, "review_models", return_value=review),
                   mock.patch.object(setup, "download_and_verify", return_value=outcome),
                   mock.patch.object(setup, "step"), mock.patch.object(setup, "say")]
        mocks = []
        for patcher in patches:
            mocks.append(patcher.start())
            self.addCleanup(patcher.stop)
        self.review, self.download = mocks[1], mocks[2]
        return setup.choose_models("http://localhost:11434")

    def test_the_names_are_kept_once_the_user_confirms_them_and_before_the_download(self):
        seen = []
        with mock.patch.object(setup, "download_and_verify", side_effect=lambda *a: seen.append(config.load_config()) or (None, {})):
            with mock.patch.object(setup, "ask_models", side_effect=[("v:1b", "e:1b", "t:1b"), KeyboardInterrupt]), \
                    mock.patch.object(setup, "review_models", return_value=(True, set())), \
                    mock.patch.object(setup, "step"), mock.patch.object(setup, "say"):
                with self.assertRaises(KeyboardInterrupt):
                    setup.choose_models("http://localhost:11434")
        self.assertEqual(seen[0]["pending_models"], {"vlm": "v:1b", "embedding": "e:1b", "text": "t:1b"})

    def test_the_next_run_fills_in_the_names_and_still_asks_for_confirmation(self):
        config.save_config({"ollama_host": "http://localhost:11434",
                            "pending_models": {"vlm": "v:1b", "embedding": "e:1b", "text": "t:1b"}})
        self.choose(("v:1b", "e:1b", "t:1b"))
        self.assertEqual(self.asked, [("v:1b", "e:1b", "t:1b")])
        self.review.assert_called_once()  # the confirmation menu is still shown, and only after it is anything downloaded
        self.download.assert_called_once_with("http://localhost:11434", "v:1b", "e:1b", "t:1b")

    def test_names_that_were_not_found_or_did_not_work_are_not_shown_again(self):
        config.save_config({"pending_models": {"vlm": "v:1b", "embedding": "e:1b", "text": "t:1b"}})
        with mock.patch.object(setup, "ask_models", side_effect=[("v:1b", "e:1b", "t:1b"), ("v:1b", "e2:1b", "t:1b"), KeyboardInterrupt]), \
                mock.patch.object(setup, "review_models", side_effect=[(False, {"e:1b"}), (True, set())]), \
                mock.patch.object(setup, "download_and_verify", return_value=(None, {"t:1b": "did not work."})), \
                mock.patch.object(setup, "step"), mock.patch.object(setup, "say"):
            with self.assertRaises(KeyboardInterrupt):
                setup.choose_models("http://localhost:11434")
        self.assertEqual(config.load_config()["pending_models"], {"vlm": "v:1b", "embedding": "e2:1b", "text": ""})

    def test_a_broken_saved_value_is_ignored(self):
        config.save_config({"pending_models": ["v:1b"]})
        self.assertEqual(setup._load_pending_models(), ("", "", ""))
        config.save_config({"pending_models": {"vlm": 3, "embedding": " e:1b ", "text": None}})
        self.assertEqual(setup._load_pending_models(), ("", "e:1b", ""))

    def test_other_settings_in_config_json_are_kept(self):
        config.save_config({"ollama_host": "http://localhost:11434"})
        setup._store_pending_models("v:1b", "e:1b", "t:1b")
        setup._store_pending_models("", "", "")
        self.assertEqual(config.load_config(), {"ollama_host": "http://localhost:11434"})


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


class KeepDeletedQuestionTests(IsolatedHome):
    """The question at the end of setup: how many deleted pictures Imagent remembers."""

    def answers(self, *given):
        """Answer the questions in turn; ``questions`` collects what was asked (the question objects are built, not shown)."""
        self.questions = []

        def ask(question):
            self.questions.append(question)
            return given[len(self.questions) - 1]

        patcher = mock.patch.object(setup, "ask", side_effect=ask)
        patcher.start()
        self.addCleanup(patcher.stop)
        output = mock.patch.object(setup, "say")
        self.said = output.start()
        self.addCleanup(output.stop)
        step = mock.patch.object(setup, "step")
        step.start()
        self.addCleanup(step.stop)

    def stored(self):
        conn = self.connect()
        try:
            return get_keep_deleted(conn)
        finally:
            conn.close()

    def test_the_choices_are_the_offered_numbers_with_none_among_them_and_nothing_is_typed(self):
        self.answers(500)
        self.assertEqual(setup.ask_keep_deleted(), 500)
        self.assertEqual(setup.KEEP_DELETED_CHOICES, (0, 100, 500, 1000, 2000, 5000))
        self.assertEqual(len(self.questions), 1)  # one menu; no second question for a typed number

    def test_the_explanation_says_why_remembering_helps(self):
        self.answers(100)
        setup.ask_keep_deleted()
        text = " ".join(call.args[0] for call in self.said.call_args_list)
        self.assertIn("does not have to be analysed again", text)
        self.assertIn("oldest are forgotten first", text)

    def test_the_answer_is_stored(self):
        self.answers(2000)
        setup.choose_keep_deleted()
        self.assertEqual(self.stored(), 2000)

    def test_an_answer_given_before_is_kept_and_not_asked_again(self):
        self.answers(2000)
        setup.choose_keep_deleted()
        self.answers()  # nothing is asked the second time
        setup.choose_keep_deleted()
        self.assertEqual((self.stored(), len(self.questions)), (2000, 0))

    def test_zero_is_stored_as_remembering_none(self):
        self.answers(0)
        setup.choose_keep_deleted()
        self.assertEqual(self.stored(), 0)


SCRIPT = b"#!/bin/sh\necho 'ollama version is 0.40.17'\n"  # stands in for the program: it runs and says its version
BROKEN = b"#!/bin/sh\nexit 1\n"


def _tar(path: Path, entries, mode="w:gz") -> None:
    """A tar archive of ``entries``: (name, content or None for a folder, file mode) or (name, "link", target)."""
    with tarfile.open(path, mode) as bundle:
        for name, content, extra in entries:
            info = tarfile.TarInfo(name)
            if content is None:
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                bundle.addfile(info)
            elif content == "link":
                info.type, info.linkname = tarfile.SYMTYPE, extra
                bundle.addfile(info)
            else:
                info.size, info.mode = len(content), extra
                bundle.addfile(info, io.BytesIO(content))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InstallOllamaTests(IsolatedHome):
    """Ollama's own build is downloaded (here from file:// addresses, so no network is used), checked, unpacked and tried."""

    def install(self, archive: Path, sha=None, extras=None, more=(), manifest_sha=None, listed=True, wrong=()):
        """Install ``archive`` (and the ``more`` archives as its extras) from a release folder with a sha256sum.txt of its own."""
        prefix = next(archive.name[:-len(ending)] for ending in config.OLLAMA_ARCHIVE_ENDINGS if archive.name.endswith(ending))
        lines = [f"{'0' * 64 if file.name in wrong else (sha if file == archive and sha else _sha256(file))}  ./{file.name}"
                 for file in (archive, *more) if listed or file != archive]
        manifest = self.work / config.OLLAMA_MANIFEST
        manifest.write_text("\n".join(lines) + "\n")
        patches = (mock.patch.object(config, "OLLAMA_RELEASE_URL", self.work.as_uri() + "/{file}"),
                   mock.patch.object(config, "OLLAMA_MANIFEST_SHA256", manifest_sha or _sha256(manifest)),
                   mock.patch.object(config, "ollama_build_prefix", return_value=prefix),
                   mock.patch.object(setup, "_ollama_extras", return_value=extras or {}),
                   mock.patch.object(setup, "say"), contextlib.redirect_stdout(io.StringIO()))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            setup.install_ollama()

    def failure(self, archive: Path, **kwargs) -> str:
        with self.assertRaises(SystemExit) as raised:
            self.install(archive, **kwargs)
        return str(raised.exception.code)

    def leftovers(self):
        return sorted(path.name for path in config.OLLAMA_DIR.iterdir()) if config.OLLAMA_DIR.exists() else []

    @unittest.skipIf(sys.platform == "win32", "the stand-in program is a shell script")
    def test_a_macos_style_archive_installs_a_program_that_runs_and_is_recorded(self):
        archive = self.work / "ollama-darwin.tgz"
        _tar(archive, [("ollama", SCRIPT, 0o755), ("libggml.0.dylib", b"lib", 0o644), ("libggml.dylib", "link", "libggml.0.dylib")])
        self.install(archive)
        self.assertEqual(config.private_ollama_binary(), str(config.OLLAMA_DIR / "program" / "ollama"))
        self.assertEqual(setup._ollama_command_version(config.private_ollama_binary()), (0, 40, 17))
        self.assertTrue((config.OLLAMA_DIR / "program" / "libggml.dylib").is_symlink())
        self.assertEqual(self.leftovers(), ["imagent-tool.json", "models", "program"])  # the download and the half-unpacked folder are gone
        self.assertEqual([(tool["name"], tool["version"]) for tool in config.installed_tools()], [("Ollama", config.OLLAMA_VERSION)])

    @unittest.skipIf(sys.platform == "win32", "the stand-in program is a shell script")
    def test_a_linux_style_zstd_archive_installs_into_bin(self):
        import zstandard
        plain = self.work / "plain.tar"
        _tar(plain, [("bin/", None, 0), ("bin/ollama", SCRIPT, 0o755), ("lib/ollama/", None, 0), ("lib/ollama/libx.so", b"x", 0o644)], mode="w")
        archive = self.work / "ollama-linux-amd64.tar.zst"
        archive.write_bytes(zstandard.ZstdCompressor().compress(plain.read_bytes()))
        self.install(archive)
        self.assertEqual(config.private_ollama_binary(), str(config.OLLAMA_DIR / "program" / "bin" / "ollama"))
        self.assertTrue((config.OLLAMA_DIR / "program" / "lib" / "ollama" / "libx.so").exists())

    def test_a_windows_style_zip_installs_ollama_exe(self):
        archive = self.work / "ollama-windows-amd64.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("ollama.exe", b"exe")
            bundle.writestr("lib/ollama/libx.dll", b"dll")
        with mock.patch.object(config.platform, "system", return_value="Windows"), \
                mock.patch.object(setup.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            self.install(archive)
            self.assertEqual(config.private_ollama_binary(), str(config.OLLAMA_DIR / "program" / "ollama.exe"))

    def test_a_download_that_is_not_the_published_file_installs_nothing(self):
        archive = self.work / "ollama-darwin.tgz"
        _tar(archive, [("ollama", SCRIPT, 0o755)])
        message = self.failure(archive, sha="0" * 64)
        self.assertIn("checksum", message)
        self.assertIn("https://ollama.com/download", message)  # and how to do it by hand
        self.assertEqual(self.leftovers(), [])
        self.assertIsNone(config.private_ollama_binary())

    def test_an_entry_that_would_leave_the_folder_is_refused(self):
        outside = self.work / "outside.txt"
        for entries in ([("../outside.txt", b"bad", 0o644)], [("ollama", "link", "/etc/passwd")], [("a", "link", "../../outside.txt")]):
            archive = self.work / "ollama-darwin.tgz"
            _tar(archive, entries)
            self.assertIn("unsafe entry", self.failure(archive))
            self.assertFalse(outside.exists())
            self.assertEqual(self.leftovers(), [])

    @unittest.skipIf(sys.platform == "win32", "the stand-in program is a shell script")
    def test_a_program_that_does_not_run_is_removed_again(self):
        archive = self.work / "ollama-darwin.tgz"
        _tar(archive, [("ollama", BROKEN, 0o755)])
        self.assertIn("does not run", self.failure(archive))
        self.assertEqual(self.leftovers(), [])

    def test_an_archive_without_the_program_is_refused(self):
        archive = self.work / "ollama-darwin.tgz"
        _tar(archive, [("readme.txt", b"hi", 0o644)])
        self.assertIn("does not contain", self.failure(archive))
        self.assertEqual(self.leftovers(), [])

    def test_too_little_free_disk_space_stops_before_downloading_it_all(self):
        archive = self.work / "ollama-darwin.tgz"
        _tar(archive, [("ollama", SCRIPT, 0o755)])
        with mock.patch.object(setup.shutil, "disk_usage", return_value=SimpleNamespace(free=10)):
            message = self.failure(archive)
        self.assertIn("not enough free disk space", message)
        self.assertEqual(self.leftovers(), [])

    def test_a_computer_with_no_build_is_told_to_install_it_by_hand(self):
        with mock.patch.object(config, "ollama_build_prefix", return_value=None), self.assertRaises(SystemExit) as raised:
            setup.install_ollama()
        self.assertIn("https://ollama.com/download", str(raised.exception.code))

    def test_a_list_of_files_that_is_not_the_pinned_one_installs_nothing(self):
        archive = self.work / "ollama-darwin.tgz"
        _tar(archive, [("ollama", SCRIPT, 0o755)])
        message = self.failure(archive, manifest_sha="0" * 64)
        self.assertIn("list of Ollama's files", message)
        self.assertEqual(self.leftovers(), [])

    def test_a_release_that_does_not_list_the_build_for_this_computer_installs_nothing(self):
        archive = self.work / "ollama-darwin.tgz"
        _tar(archive, [("ollama", SCRIPT, 0o755)])
        other = self.work / "ollama-windows-amd64.zip"
        other.write_bytes(b"x")
        message = self.failure(archive, more=(other,), listed=False)
        self.assertIn("has no build for this computer", message)
        self.assertEqual(self.leftovers(), [])

    @unittest.skipIf(sys.platform == "win32", "the stand-in program is a shell script")
    def test_the_extra_file_for_the_gpu_is_unpacked_over_the_build(self):
        import zstandard
        build, extra = self.work / "ollama-linux-amd64.tar.zst", self.work / "ollama-linux-amd64-rocm.tar.zst"
        for archive, entries in ((build, [("bin/", None, 0), ("bin/ollama", SCRIPT, 0o755), ("lib/ollama/libx.so", b"x", 0o644)]),
                                 (extra, [("lib/ollama/rocm/librocm.so", b"amd", 0o644)])):
            plain = self.work / "plain.tar"
            _tar(plain, entries, mode="w")
            archive.write_bytes(zstandard.ZstdCompressor().compress(plain.read_bytes()))
        self.install(build, extras={"rocm": "an AMD GPU"}, more=(extra,))
        program = config.OLLAMA_DIR / "program"
        self.assertEqual(setup._ollama_command_version(config.private_ollama_binary()), (0, 40, 17))
        self.assertEqual((program / "lib" / "ollama" / "libx.so").read_bytes(), b"x")  # the build is still there
        self.assertEqual((program / "lib" / "ollama" / "rocm" / "librocm.so").read_bytes(), b"amd")
        self.assertEqual(self.leftovers(), ["imagent-tool.json", "models", "program"])  # neither download is left

    @unittest.skipIf(sys.platform == "win32", "the stand-in program is a shell script")
    def test_an_extra_file_that_is_not_the_published_one_installs_nothing(self):
        build, extra = self.work / "ollama-darwin.tgz", self.work / "ollama-darwin-rocm.tgz"
        _tar(build, [("ollama", SCRIPT, 0o755)])
        _tar(extra, [("rocm", b"amd", 0o644)])
        message = self.failure(build, extras={"rocm": "an AMD GPU"}, more=(extra,), wrong=(extra.name,))
        self.assertIn("checksum", message)
        self.assertEqual(self.leftovers(), [])  # the good build that was already unpacked is not left half-installed
        self.assertIsNone(config.private_ollama_binary())

    @unittest.skipIf(sys.platform == "win32", "the stand-in program is a shell script")
    def test_an_extra_file_the_release_lacks_is_skipped_and_the_build_still_installs(self):
        archive = self.work / "ollama-darwin.tgz"
        _tar(archive, [("ollama", SCRIPT, 0o755)])
        self.install(archive, extras={"rocm": "an AMD GPU"})
        self.assertIsNotNone(config.private_ollama_binary())

    @unittest.skipIf(sys.platform == "win32", "the stand-in program is a shell script")
    def test_installing_again_keeps_the_models_and_clears_what_an_earlier_try_left(self):
        (config.OLLAMA_DIR / "models").mkdir(parents=True)
        (config.OLLAMA_DIR / "models" / "blob").write_bytes(b"weights")
        (config.OLLAMA_DIR / "program.partial").mkdir()
        (config.OLLAMA_DIR / "program.partial" / "junk").write_bytes(b"x")
        archive = self.work / "ollama-darwin.tgz"
        _tar(archive, [("ollama", SCRIPT, 0o755)])
        self.install(archive)
        self.assertEqual((config.OLLAMA_DIR / "models" / "blob").read_bytes(), b"weights")
        self.assertFalse((config.OLLAMA_DIR / "program.partial").exists())


class OllamaExtrasTests(IsolatedHome):
    """Which extra Ollama files a computer's GPU needs: AMD and Jetson yes, NVIDIA and Mac no."""

    def extras(self, system="Linux", nvidia=False, vendors=(), tegra=None, powershell=""):
        cards = self.work / "drm"
        for number, vendor in enumerate(vendors):
            (cards / f"card{number}" / "device").mkdir(parents=True, exist_ok=True)
            (cards / f"card{number}" / "device" / "vendor").write_text(vendor + "\n")
        release = self.work / "nv_tegra_release"
        if tegra is not None:
            release.write_text(f"# {tegra} (release), REVISION: 3.1\n")
        with mock.patch.object(setup.platform, "system", return_value=system), mock.patch.object(setup, "DRM_DIR", cards), \
                mock.patch.object(setup, "TEGRA_RELEASE", release), \
                mock.patch.object(setup.shutil, "which", return_value="/usr/bin/nvidia-smi" if nvidia else None), \
                mock.patch.object(setup, "_run", side_effect=lambda cmd: "GPU 0: NVIDIA" if cmd[0] == "nvidia-smi" else powershell):
            return setup._ollama_extras()

    def test_an_amd_gpu_gets_the_rocm_files_on_linux_and_windows(self):
        self.assertEqual(list(self.extras(vendors=("0x1002",))), ["rocm"])
        self.assertEqual(list(self.extras("Windows", powershell="AMD Radeon RX 7800 XT")), ["rocm"])

    def test_an_nvidia_gpu_or_no_gpu_gets_no_extra_files(self):
        self.assertEqual(self.extras(vendors=("0x10de",)), {})
        self.assertEqual(self.extras(nvidia=True, vendors=("0x10de",)), {})
        self.assertEqual(self.extras(), {})
        self.assertEqual(self.extras("Windows", powershell="NVIDIA GeForce RTX 4090"), {})

    def test_when_both_are_there_the_nvidia_one_is_used_so_nothing_extra_is_installed(self):
        self.assertEqual(self.extras(nvidia=True, vendors=("0x1002", "0x10de")), {})

    def test_a_jetson_board_gets_the_files_of_its_jetpack(self):
        self.assertEqual(list(self.extras(tegra="R36")), ["jetpack6"])
        self.assertEqual(list(self.extras(tegra="R35")), ["jetpack5"])
        self.assertEqual(self.extras(tegra="R34"), {})  # a JetPack Ollama has no files for

    def test_a_mac_needs_none(self):
        self.assertEqual(self.extras("Darwin", vendors=("0x1002",)), {})


class DownloadProgressTests(IsolatedHome):
    def test_what_arrived_is_summed_into_the_checksum(self):
        source = self.work / "file.bin"
        source.write_bytes(b"abc" * 1000)
        target = self.work / "copy.bin"
        with contextlib.redirect_stdout(io.StringIO()):
            digest = setup._download_file(source.as_uri(), target)
        self.assertEqual(digest, _sha256(source))
        self.assertEqual(target.read_bytes(), source.read_bytes())

