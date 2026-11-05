import contextlib
import io
import sqlite3
import unittest
from pathlib import Path
from unittest import mock

import httpx
import ollama

import config
from scan_and_index import scan_and_index
from db.operations import get_backlog, get_or_add_root, upsert_backlog
from tests.support import IsolatedHome
from image_processing.problems import BUSY, UNREADABLE, ImageProblem, problem
from pipeline.failure import RunStopped
from vlm.analyzer import AnalysisFailed


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
            ("collect_versions", mock.Mock(return_value={})),
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
        def index_image(conn, client, settings, root_id, path, on_stage, on_retry):
            if path.name == "b.png":
                raise ValueError("cannot be decoded")
            return "indexed", "ok"
        with mock.patch.object(scan_and_index, "index_image", side_effect=index_image):
            scan_and_index._index_backlog()
        self.assertEqual(self._attempts(), {"b.png": (1, "ValueError: cannot be decoded")})

    def test_a_reply_that_cannot_be_put_right_parks_the_file_and_the_run_goes_on(self):
        attempt = {"step": "first try", "kind": "cut_off", "options": {"num_predict": 5}, "detail": "cut", "raw_reply": "{",
                   "done_reason": "length", "eval_count": 5, "prompt_eval_count": 9}

        def index_image(conn, client, settings, root_id, path, on_stage, on_retry):
            if path.name == "b.png":
                raise AnalysisFailed("cut_off", "cut off at 5 tokens", [attempt])
            return "indexed", "ok"
        with mock.patch.object(scan_and_index, "index_image", side_effect=index_image):
            scan_and_index._index_backlog()
        row = self.conn.execute("SELECT * FROM backlog").fetchall()
        self.assertEqual([r["file_path"].rsplit("/", 1)[-1] for r in row], ["b.png"])
        self.assertEqual((row[0]["status"], row[0]["park_kind"], row[0]["attempts"]), ("parked", "cut_off", 0))
        self.assertEqual(len(get_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS)), 0)  # parked: not queued again by itself
        history = self.conn.execute("SELECT kind, step, raw_reply FROM failure_history").fetchall()
        self.assertEqual([tuple(h) for h in history], [("cut_off", "first try", "{")])

    def test_a_file_that_changes_is_queued_again_after_being_parked(self):
        with mock.patch.object(scan_and_index, "index_image", side_effect=AnalysisFailed("cut_off", "x", [])):
            scan_and_index._index_backlog()
        self.assertEqual(len(get_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS)), 0)
        with self.conn:
            for name in ("a.png", "b.png", "c.png", "d.png"):
                upsert_backlog(self.conn, str(self.work / name), 1, "a later time")
        self.assertEqual(len(get_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS)), 4)

    def test_indexing_stops_when_most_replies_need_asking_for_again(self):
        def index_image(conn, client, settings, root_id, path, on_stage, on_retry):
            on_retry("cut_off")
            return "indexed", "ok"
        with mock.patch.object(config, "RETRY_RATE_WINDOW", 3), mock.patch.object(scan_and_index, "index_image", side_effect=index_image):
            with self.assertRaises(SystemExit) as stop:
                scan_and_index._index_backlog()
        self.assertEqual(stop.exception.code, 1)
        self.assertEqual(len(self._attempts()), 1)  # three were indexed before it stopped; the fourth is still queued

    def run_with(self, outcomes):
        """Run the indexing loop with ``index_image`` giving each file in turn the next outcome (an exception is raised)."""
        outcomes = list(outcomes)

        def index_image(conn, client, settings, root_id, path, on_stage, on_retry):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        with mock.patch.object(scan_and_index, "index_image", side_effect=index_image):
            try:
                scan_and_index._index_backlog()
                return None
            except SystemExit as stop:
                return stop

    def _rows(self):
        return {Path(r["file_path"]).name: dict(r) for r in self.conn.execute("SELECT * FROM backlog")}

    def test_an_image_with_a_problem_of_its_own_is_set_aside_with_the_reason_and_when_to_retry(self):
        damaged = problem(UNREADABLE, "The image cannot be read; it is damaged.", retry_when="file_changed")
        self.run_with([("indexed", "ok"), damaged, ("indexed", "ok"), ("indexed", "ok")])
        row = self._rows()["b.png"]
        self.assertEqual((row["status"], row["park_kind"], row["retry_when"], row["attempts"]), ("parked", "unreadable", "file_changed", 0))
        self.assertEqual(self.conn.execute("SELECT kind, step FROM failure_history").fetchone()[:], ("unreadable", "opened the image"))

    def test_a_file_still_being_written_stays_queued_and_loses_no_attempt(self):
        busy = ImageProblem(BUSY, "still being written", park=False)
        self.run_with([("indexed", "ok"), busy, ("indexed", "ok"), ("indexed", "ok")])
        row = self._rows()["b.png"]
        self.assertEqual((row["status"], row["attempts"], row["last_error"]), ("queued", 0, "still being written"))

    def test_trouble_with_the_machine_several_times_in_a_row_stops_the_run_and_parks_nothing_that_is_not_the_files_fault(self):
        short = AnalysisFailed("out_of_memory", "not enough memory", [])
        stop = self.run_with([short] * 4)
        self.assertEqual(stop.code, 1)
        rows = self._rows()
        self.assertEqual(sum(1 for r in rows.values() if r["last_error"]), config.MAX_CONSECUTIVE_SERVICE_FAILURES)
        self.assertTrue(all(r["status"] == "queued" and r["attempts"] == 0 for r in rows.values()))

    def test_timeouts_in_a_row_stop_the_run_too(self):
        stop = self.run_with([AnalysisFailed("timeout", "no reply", [])] * 4)
        self.assertEqual(stop.code, 1)
        self.assertEqual(sum(1 for r in self._rows().values() if r["status"] == "parked"), config.MAX_CONSECUTIVE_SERVICE_FAILURES)

    def test_a_problem_of_the_files_does_not_count_towards_stopping(self):
        damaged = problem(UNREADABLE, "damaged", retry_when="file_changed")
        self.assertIsNone(self.run_with([damaged] * 4))
        self.assertTrue(all(r["status"] == "parked" for r in self._rows().values()))

    def test_files_that_cannot_be_read_one_after_another_stop_the_run_and_keep_their_attempts(self):
        stop = self.run_with([PermissionError(13, "denied")] * 4)
        self.assertEqual(stop.code, 1)
        rows = self._rows()
        self.assertTrue(all(r["attempts"] == 0 for r in rows.values()))  # given back: a drive that went away is not their fault
        self.assertEqual(sum(1 for r in rows.values() if r["last_error"]), config.MAX_CONSECUTIVE_SERVICE_FAILURES)

    def test_one_file_that_cannot_be_read_costs_an_attempt_and_the_run_goes_on(self):
        self.assertIsNone(self.run_with([("indexed", "ok"), PermissionError(13, "denied"), ("indexed", "ok"), ("indexed", "ok")]))
        self.assertEqual(self._attempts(), {"b.png": (1, "PermissionError: [Errno 13] denied")})

    def test_a_success_ends_a_run_of_unreadable_files(self):
        denied = PermissionError(13, "denied")
        self.assertIsNone(self.run_with([denied, denied, ("indexed", "ok"), denied]))

    def test_a_problem_with_the_whole_setup_stops_the_run_without_touching_the_file(self):
        stop = self.run_with([RunStopped("The embedding model has changed.")])
        self.assertEqual(stop.code, 1)
        rows = self._rows()
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["attempts"] == 0 and r["status"] == "queued" and not r["last_error"] for r in rows.values()))

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
