"""Measuring the computer at setup: which picture size, context, memory mode and wait it ends up with, and when it gives up."""

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
import ollama
from PIL import Image

import config
from db.operations import get_device_settings, set_device_settings
from tests.support import DEVICE, IsolatedHome
from vlm import calibration

REQUEST = httpx.Request("POST", "http://localhost")
NO_MEMORY = ollama.ResponseError("model requires more system memory than is available", 500)


class FakeOllama:
    """Answers each chat request like a model would: ``outcome(request)`` is a reply's numbers, or an error to raise."""

    def __init__(self, outcome, size=2_000_000_000, context_limit=32768, embedder=lambda: None):
        self.outcome, self.requests, self.size, self.context_limit = outcome, [], size, context_limit
        self.embedder, self.embeds = embedder, []  # embedder() returns an error to raise for this embedding request, or None

    def embed(self, **request):
        self.embeds.append(request)
        error = self.embedder()
        if error is not None:
            raise error
        return SimpleNamespace(embeddings=[[0.0]])

    def chat(self, **request):
        self.requests.append(request)
        result = self.outcome(request)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(**result)

    def show(self, name):
        return SimpleNamespace(modelinfo={"general.architecture": "x", "x.context_length": self.context_limit})

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model="vlm", size=self.size)])


def reply(prompt_tokens=1500, prefill=2.0, load=5.0, rate=10.0):
    """The numbers of a reply of 48 tokens at ``rate`` tokens per second."""
    return {"prompt_eval_count": prompt_tokens, "prompt_eval_duration": int(prefill * 1e9), "load_duration": int(load * 1e9),
            "eval_count": 48, "eval_duration": int(48 / rate * 1e9), "done_reason": "stop",
            "message": SimpleNamespace(content="{}")}


def edge_of(request):
    with Image.open(io.BytesIO(request["messages"][1]["images"][0])) as image:
        return max(image.size)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        for target, replacement in (("available_memory", mock.Mock(return_value=16_000_000_000)),
                                    ("unload_model", mock.Mock())):
            patcher = mock.patch.object(calibration, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.unload = calibration.unload_model
        # No waiting or restarting while the steps for a failure are tried.
        for target in ("restart_server", "other_models_running", "unload_model"):
            patcher = mock.patch("vlm.analyzer." + target, mock.Mock(return_value=False if target == "restart_server" else []))
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch("vlm.analyzer.time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch("vlm.analyzer.make_client")
        self.make_client = patcher.start()
        self.addCleanup(patcher.stop)

    def run_it(self, client):
        return calibration.calibrate(client, "vlm", "emb", say=lambda message: None)

    def test_a_computer_that_manages_the_largest_picture_gets_it(self):
        client = FakeOllama(lambda request: reply())
        found = self.run_it(client)
        self.assertEqual(found["image_long_edge"], config.IMAGE_EDGE_CEILING)
        self.assertEqual({edge_of(r) for r in client.requests}, {config.IMAGE_EDGE_CEILING})
        self.assertFalse(found["use_mmap"])
        self.assertEqual([call.args[1] for call in self.unload.call_args_list], ["vlm", "emb"])  # both models' memory is freed again

    def test_the_picture_is_made_smaller_until_the_computer_manages_it(self):
        def outcome(request):
            return NO_MEMORY if edge_of(request) > 600 else reply()
        found = self.run_it(FakeOllama(outcome))
        self.assertLessEqual(found["image_long_edge"], 600)
        self.assertGreaterEqual(found["image_long_edge"], config.CALIBRATION_MIN_EDGE)

    def test_a_computer_that_manages_no_picture_size_is_told_the_model_does_not_run_on_it(self):
        with self.assertRaisesRegex(calibration.NotPossible, "did not work on this computer"):
            self.run_it(FakeOllama(lambda request: NO_MEMORY))

    def test_the_context_is_the_smallest_power_of_two_that_holds_the_prompt_and_room_to_raise_the_cap(self):
        found = self.run_it(FakeOllama(lambda request: reply(prompt_tokens=1500)))
        needed = 1500 + int(config.OLLAMA_MAX_OUTPUT_TOKENS * config.ANALYSIS_CAP_STEP)
        self.assertEqual(found["num_ctx"], 1 << (needed - 1).bit_length())

    def test_a_context_that_is_too_small_for_the_prompt_is_made_larger_and_tried_again(self):
        client = FakeOllama(lambda request: reply(prompt_tokens=9000))  # more than the first context holds
        found = self.run_it(client)
        self.assertGreaterEqual(found["num_ctx"], 16384)
        contexts = [request["options"]["num_ctx"] for request in client.requests]
        self.assertLess(contexts[0], contexts[-1])

    def test_a_prompt_that_needs_more_than_the_model_supports_goes_on_to_a_smaller_picture(self):
        def outcome(request):  # the picture's size decides how many tokens it takes
            return reply(prompt_tokens=edge_of(request) * 10)
        found = self.run_it(FakeOllama(outcome, context_limit=8192))
        self.assertLess(found["image_long_edge"], config.IMAGE_EDGE_CEILING)

    def test_a_way_of_loading_that_answers_nothing_is_replaced_by_the_other_before_the_picture_is_made_smaller(self):
        def outcome(request):  # some Ollama builds return an empty answer, with no error, when the model is paged in from disk
            return {**reply(), "message": SimpleNamespace(content=""), "eval_count": 0} if request["options"]["use_mmap"] else reply()
        with mock.patch.object(calibration, "available_memory", return_value=2_000_000_000):  # a model this size would be paged in
            found = self.run_it(FakeOllama(outcome, size=4_000_000_000))
        self.assertFalse(found["use_mmap"])
        self.assertEqual(found["image_long_edge"], config.IMAGE_EDGE_CEILING)

    def test_running_out_of_memory_loaded_fully_is_tried_again_paged_in_from_disk(self):
        def outcome(request):
            return NO_MEMORY if not request["options"]["use_mmap"] else reply()
        found = self.run_it(FakeOllama(outcome))
        self.assertTrue(found["use_mmap"])
        self.assertEqual(found["image_long_edge"], config.IMAGE_EDGE_CEILING)

    def test_a_model_that_never_replies_in_time_is_not_tried_the_other_way_as_well(self):
        slow = httpx.ReadTimeout("slow", request=REQUEST)
        client = FakeOllama(lambda request: slow)
        self.make_client.return_value = client  # the longer waits are made with a new client
        with self.assertRaises(calibration.NotPossible):
            self.run_it(client)
        modes = {request["options"]["use_mmap"] for request in client.requests}
        self.assertEqual(len(modes), 1)  # waiting as long again in the other way would not help

    def test_a_model_larger_than_the_free_memory_is_paged_in_from_disk(self):
        with mock.patch.object(calibration, "available_memory", return_value=2_000_000_000):
            found = self.run_it(FakeOllama(lambda request: reply(), size=4_000_000_000))
        self.assertTrue(found["use_mmap"])

    def test_a_model_that_fits_with_room_to_spare_is_loaded_fully(self):
        with mock.patch.object(calibration, "available_memory", return_value=16_000_000_000):
            found = self.run_it(FakeOllama(lambda request: reply(), size=4_000_000_000))
        self.assertFalse(found["use_mmap"])

    def test_the_wait_follows_the_measured_speed(self):
        slow = self.run_it(FakeOllama(lambda request: reply(rate=2.0)))
        fast = self.run_it(FakeOllama(lambda request: reply(rate=40.0)))
        self.assertGreater(slow["timeout_seconds"], fast["timeout_seconds"] * 5)
        expected = (5.0 + 2.0 + config.OLLAMA_MAX_OUTPUT_TOKENS / 2.0) * config.CALIBRATION_TIMEOUT_MARGIN
        self.assertAlmostEqual(slow["timeout_seconds"], expected, delta=1)

    def test_a_model_that_cannot_say_how_fast_it_is_is_not_trusted(self):
        broken = reply()
        broken["eval_duration"] = 0
        with self.assertRaises(calibration.NotPossible):
            self.run_it(FakeOllama(lambda request: broken))

    def test_the_picture_is_made_here_at_the_size_asked_for(self):
        for edge in (336, 1024):
            with Image.open(io.BytesIO(calibration.calibration_image(edge))) as image:
                self.assertEqual((image.format, max(image.size)), ("JPEG", edge))


    def test_an_embedding_model_that_fits_beside_the_vision_model_leaves_both_loaded_together(self):
        client = FakeOllama(lambda request: reply())
        found = self.run_it(client)
        self.assertFalse(found["one_model_at_a_time"])
        self.assertEqual(len(client.embeds), 1)
        self.assertEqual(client.embeds[0]["model"], "emb")
        self.assertIs(client.embeds[0]["options"]["use_mmap"], found["use_mmap"])  # the options indexing will use
        self.assertGreater(len(client.embeds[0]["input"][0]), config.CALIBRATION_EMBED_TOKENS)  # about as long as a stored description

    def test_an_embedding_model_that_fails_beside_the_vision_model_but_runs_alone_means_one_model_at_a_time(self):
        failures = iter([ollama.ResponseError("Compute error.", 500)])  # the first try, with the vision model loaded
        client = FakeOllama(lambda request: reply(), embedder=lambda: next(failures, None))
        found = self.run_it(client)
        self.assertTrue(found["one_model_at_a_time"])
        self.assertEqual(len(client.embeds), 2)
        # both were unloaded between the two tries
        self.assertEqual([call.args[1] for call in self.unload.call_args_list][:2], ["emb", "vlm"])

    def test_an_embedding_model_that_does_not_run_even_alone_means_the_computer_cannot_run_the_models(self):
        client = FakeOllama(lambda request: reply(), embedder=lambda: ollama.ResponseError("Compute error.", 500))
        with self.assertRaisesRegex(calibration.NotPossible, "embedding model 'emb' did not work"):
            self.run_it(client)
        self.assertEqual(len(client.embeds), 2)

    def test_other_trouble_with_the_embedding_model_is_not_taken_for_not_fitting(self):
        client = FakeOllama(lambda request: reply(), embedder=lambda: ollama.ResponseError("model 'emb' not found", 404))
        with self.assertRaises(ollama.ResponseError):
            self.run_it(client)
        self.assertEqual(len(client.embeds), 1)


class DeviceSettingsStoreTests(IsolatedHome):
    def test_what_was_measured_is_stored_with_the_model_it_was_measured_with(self):
        conn = self.connect()
        self.assertIsNone(get_device_settings(conn))
        measured = {**DEVICE, "use_mmap": True, "measurements": {"tokens_per_second": 12.5}}
        with conn:
            set_device_settings(conn, measured, "sha256:abc")
        found = get_device_settings(conn)
        self.assertEqual((found["num_ctx"], found["use_mmap"], found["vlm_digest"]), (8192, True, "sha256:abc"))
        self.assertEqual(found["measurements"], {"tokens_per_second": 12.5})

    def test_whether_the_two_models_fit_together_is_stored_and_not_trying_it_is_not_the_same_as_fitting(self):
        conn = self.connect()
        for stored in (True, False, None):
            with conn:
                set_device_settings(conn, {**DEVICE, "one_model_at_a_time": stored}, "d")
            self.assertIs(get_device_settings(conn)["one_model_at_a_time"], stored)

    def test_a_computer_measured_before_that_was_tried_is_run_with_the_models_sharing_memory(self):
        found = {**DEVICE, "one_model_at_a_time": None}
        config.apply_device_settings(found)
        self.assertIs(config.ONE_MODEL_AT_A_TIME, False)
        config.apply_device_settings({**found, "one_model_at_a_time": True})
        self.assertIs(config.ONE_MODEL_AT_A_TIME, True)

    def test_measuring_again_replaces_the_earlier_measurement(self):
        conn = self.connect()
        with conn:
            set_device_settings(conn, DEVICE, "old")
            set_device_settings(conn, {**DEVICE, "timeout_seconds": 99}, "new")
        found = get_device_settings(conn)
        self.assertEqual((found["timeout_seconds"], found["vlm_digest"]), (99, "new"))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM device_settings").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
