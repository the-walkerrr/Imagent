import contextlib
import io
import json
import sqlite3
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import ollama
from pydantic import ValidationError

import config
from scan_and_index import scan_and_index
from db.operations import get_backlog, get_or_add_root, upsert_backlog
from tests.support import IsolatedHome
from vlm import analyzer

GOOD_REPLY = {
    "short_description": "A cat.", "scene": "A cat sits on a sofa.", "objects_and_layout": "A sofa.",
    "people_details": "No people are visible.", "colors_light_mood": "Warm.", "primary_category": "photo",
    "objects": ["cat"], "people_count": 0, "landmarks": [], "weather": None, "time_of_day": None, "lighting": None,
    "tags": ["cat"],
}


class ClassifyFailureTests(unittest.TestCase):
    def test_what_is_and_is_not_the_files_fault(self):
        request = httpx.Request("POST", "http://localhost")
        cases = [
            (httpx.ConnectError("refused", request=request), (False, True)),
            (ConnectionError("Failed to connect to Ollama"), (False, True)),
            (ollama.ResponseError("model 'x' not found", 404), (False, True)),
            (ollama.ResponseError("model runner has unexpectedly stopped", 500), (True, True)),
            (httpx.ReadTimeout("slow", request=request), (True, True)),
            (httpx.RemoteProtocolError("server disconnected", request=request), (True, True)),  # Ollama crashed mid-request
            (ValueError("bad reply"), (True, False)),
            (OSError("cannot read the file"), (True, False)),
        ]
        for exc, expected in cases:
            self.assertEqual(scan_and_index.classify_failure(exc), expected, repr(exc))


class FakeClient:
    """Stands in for ollama.Client: returns the queued replies and remembers every request."""

    def __init__(self, *replies: str):
        self.replies, self.requests = list(replies), []

    def chat(self, **request):
        self.requests.append(request)
        return SimpleNamespace(message=SimpleNamespace(content=self.replies.pop(0)))


class AnalysisRetryTests(unittest.TestCase):
    def test_a_cut_off_reply_is_asked_for_again_and_differently(self):
        client = FakeClient('{"short_description": "A ca', json.dumps(GOOD_REPLY))
        result = analyzer._request_analysis(client, "vlm", b"jpeg")
        self.assertEqual(result.short_description, "A cat.")
        first, second = client.requests
        self.assertEqual(first["options"]["temperature"], 0)
        self.assertEqual(second["options"]["temperature"], config.ANALYSIS_RETRY_TEMPERATURE)
        self.assertEqual(second["options"]["num_ctx"], first["options"]["num_ctx"])  # a different num_ctx would reload the model
        self.assertIn("not valid", second["messages"][-1]["content"])

    def test_it_gives_up_after_the_configured_retries(self):
        client = FakeClient(*["not json"] * (config.ANALYSIS_RETRIES + 1))
        with self.assertRaises(ValidationError):
            analyzer._request_analysis(client, "vlm", b"jpeg")
        self.assertEqual(len(client.requests), config.ANALYSIS_RETRIES + 1)

    def test_a_short_or_over_long_description_is_accepted(self):
        reply = dict(GOOD_REPLY, scene="x" * 5000, objects=[str(n) for n in range(40)])
        result = analyzer._request_analysis(FakeClient(json.dumps(reply)), "vlm", b"jpeg")
        self.assertLessEqual(len(result.scene), 700)
        self.assertEqual(len(result.objects), 15)
        self.assertEqual(result.people_details, "No people are visible.")


class RunTests(IsolatedHome):
    """The indexing loop with the models replaced: what happens to the queue when something breaks."""

    def setUp(self):
        super().setUp()
        quiet = contextlib.redirect_stdout(io.StringIO())  # the runner prints its summary
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        self.conn = self.connect()
        with self.conn:
            root_id = get_or_add_root(self.conn, str(self.work))
            for name in ("a.png", "b.png", "c.png", "d.png"):
                upsert_backlog(self.conn, str(self.work / name), root_id, "t")
        settings = {"vlm_model": "v", "judge_model": "j", "embedding_model": "e", "embedding_dim": 3}
        for target, replacement in (
            ("check_ready", mock.Mock(return_value=(object(), settings))),
            ("unload_model", mock.Mock()),
            ("Progress", mock.MagicMock()),
        ):
            patcher = mock.patch.object(scan_and_index, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(scan_and_index.ocr, "check_available")
        self.ocr_check = patcher.start()
        self.addCleanup(patcher.stop)

    def _attempts(self):
        return {Path(r["file_path"]).name: (r["attempts"], r["last_error"]) for r in
                self.conn.execute("SELECT file_path, attempts, last_error FROM backlog")}

    def test_a_full_disk_stops_the_run_and_blames_no_file(self):
        full = sqlite3.OperationalError("database or disk is full")
        with mock.patch.object(scan_and_index, "index_image", side_effect=full) as index_image:
            with self.assertRaises(SystemExit) as stop:
                scan_and_index._index_backlog()
        self.assertEqual(stop.exception.code, 1)
        self.assertEqual(index_image.call_count, 1)  # it did not go on to fail the other three
        self.assertEqual(len(get_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS)), 4)
        self.assertTrue(all(attempts == 0 for attempts, _ in self._attempts().values()))

    def test_ollama_down_stops_after_a_few_files_without_using_their_attempts(self):
        refused = httpx.ConnectError("refused", request=httpx.Request("POST", "http://localhost"))
        with mock.patch.object(scan_and_index, "index_image", side_effect=refused):
            with self.assertRaises(SystemExit):
                scan_and_index._index_backlog()
        attempts = self._attempts()
        self.assertEqual(len(attempts), 4)  # all still queued
        self.assertTrue(all(count == 0 for count, _ in attempts.values()))
        self.assertEqual(sum(1 for _, error in attempts.values() if error), config.MAX_CONSECUTIVE_SERVICE_FAILURES)

    def test_a_bad_file_uses_one_attempt_and_the_run_goes_on(self):
        def index_image(conn, client, settings, root_id, path, on_stage):
            if path.name == "b.png":
                raise ValueError("cannot be decoded")
            return "indexed", "ok"
        with mock.patch.object(scan_and_index, "index_image", side_effect=index_image):
            scan_and_index._index_backlog()
        self.assertEqual(self._attempts(), {"b.png": (1, "ValueError: cannot be decoded")})

    def test_missing_ocr_models_stop_the_run_before_any_image(self):
        self.ocr_check.side_effect = OSError("no internet")
        with mock.patch.object(scan_and_index, "index_image") as index_image:
            with self.assertRaisesRegex(SystemExit, "text-reading"):
                scan_and_index._index_backlog()
        index_image.assert_not_called()
        self.assertEqual(len(self._attempts()), 4)

    def test_a_second_run_at_the_same_time_is_refused(self):
        with scan_and_index.single_instance("indexing", "Indexing"):
            with self.assertRaisesRegex(SystemExit, "already running"):
                scan_and_index.run()


if __name__ == "__main__":
    unittest.main()
