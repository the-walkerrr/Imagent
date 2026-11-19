import contextlib
import io
import json
import os
import shutil
import sqlite3
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import httpx
import ollama

from imagent import config
from imagent.commands import scan
from imagent.db.operations import get_backlog, get_exhausted_backlog, get_or_add_root, upsert_backlog
from tests.support import IsolatedHome
from imagent.image_processing.hashing import compute_pixel_hash
from imagent.image_processing.problems import BUSY, TOO_LARGE, UNREADABLE, ImageProblem, problem
from imagent.pipeline.error_messages import describe
from imagent.pipeline.failure import RunStopped
from imagent.vlm.analyzer import AnalysisFailed


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
            self.assertEqual(scan.classify_failure(exc), expected, repr(exc))


class ModelMemoryTests(IsolatedHome):
    """``index_image`` with the models replaced: what is in memory when the embedding model runs."""

    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        with self.conn:
            self.root_id = get_or_add_root(self.conn, str(self.work))
        self.settings = {"vlm_model": "vlm", "embedding_model": "emb", "embedding_dim": 2, "versions": {}}
        self.steps = []
        analysis = SimpleNamespace(reply={config.SUMMARY_FIELD: "A cat."}, ocr_text=[], features={}, attempts=[])

        def step(name, result=None):
            def run(*args, **kwargs):
                self.steps.append(name if name != "unload" else f"unload {args[1]}")
                if isinstance(result, Exception):
                    raise result
                return result
            return run

        self.patch("analyse_image", step("analyse", analysis))
        self.patch("embed_search_text", step("embed", [0.5, 0.5]))
        self.patch("unload_model", step("unload"))

    def patch(self, name, replacement):
        patcher = mock.patch.object(scan, name, side_effect=replacement)
        patcher.start()
        self.addCleanup(patcher.stop)

    def index(self):
        image = self.make_image(self.work / "a.png")
        return scan.index_image(self.conn, object(), self.settings, self.root_id, image)

    def test_models_that_fit_together_are_left_to_share_the_memory(self):
        with mock.patch.object(config, "ONE_MODEL_AT_A_TIME", False):
            self.assertEqual(self.index()[0], "indexed")
        self.assertEqual(self.steps, ["analyse", "embed"])

    def test_models_that_do_not_fit_together_are_loaded_one_at_a_time(self):
        with mock.patch.object(config, "ONE_MODEL_AT_A_TIME", True):
            self.assertEqual(self.index()[0], "indexed")
        self.assertEqual(self.steps, ["analyse", "unload vlm", "embed", "unload emb"])

    def test_the_embedding_model_is_unloaded_even_when_it_fails(self):
        self.patch("embed_search_text", self.patch_failing_embed())
        with mock.patch.object(config, "ONE_MODEL_AT_A_TIME", True), self.assertRaises(ollama.ResponseError):
            self.index()
        self.assertEqual(self.steps[-2:], ["embed", "unload emb"])

    def patch_failing_embed(self):
        def fail(*args, **kwargs):
            self.steps.append("embed")
            raise ollama.ResponseError("Compute error.", 500)
        return fail


class IndexedRecordTests(IsolatedHome):
    """What ``index_image`` stores about the picture, so that a later scan can tell an edited picture from edited metadata."""

    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        with self.conn:
            self.root_id = get_or_add_root(self.conn, str(self.work))
        self.settings = {"vlm_model": "vlm", "embedding_model": "emb", "embedding_dim": 2, "versions": {}}
        self.analysis = SimpleNamespace(reply={config.SUMMARY_FIELD: "A cat."}, ocr_text=[], features={}, attempts=[])
        patchers = {name: mock.patch.object(scan, name, **options) for name, options in (
            ("analyse_image", {"return_value": self.analysis}), ("embed_search_text", {"return_value": [0.5, 0.5]}), ("unload_model", {}))}
        for patcher in patchers.values():
            self.addCleanup(patcher.stop)
        self.analyse = patchers["analyse_image"].start()
        patchers["embed_search_text"].start()
        patchers["unload_model"].start()

    def stored(self, path):
        row = self.conn.execute("SELECT * FROM image_data WHERE file_path = ?", (str(path),)).fetchone()
        return dict(row) if row else None

    def test_the_pixel_hash_of_the_picture_is_stored_with_the_hash_of_the_file(self):
        image = self.make_image(self.work / "a.png", shade=3)
        scan.index_image(self.conn, object(), self.settings, self.root_id, image)
        self.assertEqual(self.stored(image)["pixel_hash"], compute_pixel_hash(str(image)))

    def test_a_picture_pillow_cannot_read_is_stored_without_one(self):
        image = self.work / "a.xyz"
        image.write_bytes(b"not a picture Pillow knows")  # analysed through a converted copy, which is mocked here
        scan.index_image(self.conn, object(), self.settings, self.root_id, image)
        self.assertIsNone(self.stored(image)["pixel_hash"])

    def test_a_file_that_changed_while_it_was_analysed_is_not_stored_and_costs_no_attempt(self):
        image = self.make_image(self.work / "a.png")

        def edited_meanwhile(*args, **kwargs):
            later = os.stat(image).st_mtime + 100
            os.utime(image, (later, later))
            return self.analysis

        self.analyse.side_effect = edited_meanwhile
        with self.assertRaises(ImageProblem) as raised:
            scan.index_image(self.conn, object(), self.settings, self.root_id, image)
        self.assertEqual((raised.exception.kind, raised.exception.park), (BUSY, False))
        self.assertIsNone(self.stored(image))


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
            patcher = mock.patch.object(scan, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(scan.ocr, "check_available")
        self.ocr_check = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(scan, "_offer_lower_settings")  # it asks the user a question
        self.offer = patcher.start()
        self.addCleanup(patcher.stop)

    def _attempts(self):
        return {Path(r["file_path"]).name: (r["attempts"], describe(r["error_id"], json.loads(r["error_params"] or "{}")) or None) for r in
                self.conn.execute("SELECT file_path, attempts, error_id, error_params FROM backlog")}

    def test_only_the_files_of_the_chosen_folders_are_indexed(self):
        with self.conn:
            other_id = get_or_add_root(self.conn, str(self.work / "other"))
            upsert_backlog(self.conn, str(self.work / "other" / "e.png"), other_id, "t")
        with mock.patch.object(scan, "index_image", return_value=("indexed", "")) as index_image:
            scan._index_backlog({other_id})
        self.assertEqual([call.args[4].name for call in index_image.call_args_list], ["e.png"])
        self.assertEqual(len(get_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS)), 4)  # the other folder's files stay queued

    def test_a_full_disk_stops_the_run_and_blames_no_file(self):
        full = sqlite3.OperationalError("database or disk is full")
        with mock.patch.object(scan, "index_image", side_effect=full) as index_image:
            with self.assertRaises(SystemExit) as stop:
                scan._index_backlog()
        self.assertEqual(stop.exception.code, 1)
        self.assertEqual(index_image.call_count, 1)  # it did not go on to fail the other three
        self.assertEqual(len(get_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS)), 4)
        self.assertTrue(all(attempts == 0 for attempts, _ in self._attempts().values()))

    def test_ollama_down_stops_after_a_few_files_without_using_their_attempts(self):
        refused = httpx.ConnectError("refused", request=httpx.Request("POST", "http://localhost"))
        with mock.patch.object(scan, "index_image", side_effect=refused):
            with self.assertRaises(SystemExit):
                scan._index_backlog()
        attempts = self._attempts()
        self.assertEqual(len(attempts), 4)  # all still queued
        self.assertTrue(all(count == 0 for count, _ in attempts.values()))
        self.assertEqual(sum(1 for _, error in attempts.values() if error), config.MAX_CONSECUTIVE_SERVICE_FAILURES)

    def test_a_bad_file_uses_one_attempt_and_the_run_goes_on(self):
        def index_image(conn, client, settings, root_id, path, on_stage, on_retry):
            if path.name == "b.png":
                raise ValueError("cannot be decoded")
            return "indexed", "ok"
        with mock.patch.object(scan, "index_image", side_effect=index_image):
            scan._index_backlog()
        self.assertEqual(self._attempts(), {"b.png": (1, "ValueError: cannot be decoded")})

    def test_a_file_deleted_since_the_scan_is_dropped_from_the_queue_and_the_run_goes_on(self):
        def index_image(conn, client, settings, root_id, path, on_stage, on_retry):
            if path.name == "b.png":
                raise FileNotFoundError(str(path))  # the folder is still there, the file is not
            return "indexed", "ok"
        with mock.patch.object(scan, "index_image", side_effect=index_image):
            scan._index_backlog()
        self.assertEqual(self._attempts(), {})  # b.png is no longer queued, and nothing was held against it

    def test_files_missing_together_with_their_folder_stop_the_run_and_keep_their_attempts(self):
        shutil.rmtree(self.work)  # the drive went away
        with mock.patch.object(scan, "index_image", side_effect=FileNotFoundError("gone")):
            with self.assertRaises(SystemExit) as stop:
                scan._index_backlog()
        self.assertEqual(stop.exception.code, 1)
        attempts = self._attempts()
        self.assertEqual(len(attempts), 4)  # none was dropped from the queue
        self.assertTrue(all(count == 0 for count, _ in attempts.values()))  # the attempts were given back

    def test_a_reply_that_cannot_be_put_right_parks_the_file_and_the_run_goes_on(self):
        attempt = {"step": "first try", "kind": "cut_off", "options": {"num_predict": 5}, "detail": "cut", "raw_reply": "{",
                   "done_reason": "length", "eval_count": 5, "prompt_eval_count": 9}

        def index_image(conn, client, settings, root_id, path, on_stage, on_retry):
            if path.name == "b.png":
                raise AnalysisFailed("cut_off", "reply_cut_off", [attempt])
            return "indexed", "ok"
        with mock.patch.object(scan, "index_image", side_effect=index_image):
            scan._index_backlog()
        row = self.conn.execute("SELECT * FROM backlog").fetchall()
        self.assertEqual([r["file_path"].rsplit("/", 1)[-1] for r in row], ["b.png"])
        self.assertEqual((row[0]["status"], row[0]["park_kind"], row[0]["attempts"]), ("parked", "cut_off", 0))
        self.assertEqual(len(get_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS)), 0)  # parked: not queued again by itself
        history = self.conn.execute("SELECT kind, step, raw_reply FROM failure_history").fetchall()
        self.assertEqual([tuple(h) for h in history], [("cut_off", "first try", "{")])

    def test_a_file_that_changes_is_queued_again_after_being_parked(self):
        with mock.patch.object(scan, "index_image", side_effect=AnalysisFailed("cut_off", "reply_cut_off", [])):
            scan._index_backlog()
        self.assertEqual(len(get_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS)), 0)
        with self.conn:
            for name in ("a.png", "b.png", "c.png", "d.png"):
                upsert_backlog(self.conn, str(self.work / name), 1, "a later time")
        self.assertEqual(len(get_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS)), 4)

    def test_indexing_stops_when_most_replies_need_asking_for_again(self):
        def index_image(conn, client, settings, root_id, path, on_stage, on_retry):
            on_retry("cut_off")
            return "indexed", "ok"
        with mock.patch.object(config, "RETRY_RATE_WINDOW", 3), mock.patch.object(scan, "index_image", side_effect=index_image):
            with self.assertRaises(SystemExit) as stop:
                scan._index_backlog()
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
        with mock.patch.object(scan, "index_image", side_effect=index_image):
            try:
                scan._index_backlog()
                return None
            except SystemExit as stop:
                return stop

    def _rows(self):
        return {Path(r["file_path"]).name: dict(r) for r in self.conn.execute("SELECT * FROM backlog")}

    def test_an_image_with_a_problem_of_its_own_is_set_aside_with_the_reason_and_when_to_retry(self):
        damaged = problem(UNREADABLE, "damaged_picture", retry_when="file_changed")
        self.run_with([("indexed", "ok"), damaged, ("indexed", "ok"), ("indexed", "ok")])
        row = self._rows()["b.png"]
        self.assertEqual((row["status"], row["park_kind"], row["retry_when"], row["attempts"]), ("parked", "unreadable", "file_changed", 0))
        self.assertEqual(self.conn.execute("SELECT kind, step FROM failure_history").fetchone()[:], ("unreadable", "opened the image"))

    def test_a_damaged_looking_file_that_changed_since_the_scan_is_still_being_written_and_stays_queued(self):
        self.make_image(self.work / "b.png")  # its modification time is not the "t" the scan recorded for it
        damaged = problem(UNREADABLE, "damaged_picture", retry_when="file_changed")
        with mock.patch("time.sleep", side_effect=AssertionError("it waited")):
            self.run_with([("indexed", "ok"), damaged, ("indexed", "ok"), ("indexed", "ok")])
        row = self._rows()["b.png"]
        self.assertEqual((row["status"], row["attempts"], row["park_kind"]), ("queued", 0, None))
        self.assertEqual(row["error_id"], "changed_since_scan")
        self.assertEqual(get_exhausted_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS), [])  # not reported: tried again by the next run
        self.assertEqual(self.conn.execute("SELECT kind FROM failure_history").fetchone()[0], "unreadable")  # what was found is kept

    def test_a_damaged_looking_file_that_is_as_the_scan_saw_it_is_damaged_and_set_aside(self):
        from imagent.image_processing.metadata import iso_utc
        import os
        path = self.make_image(self.work / "b.png")
        with self.conn:
            upsert_backlog(self.conn, str(path), 1, iso_utc(os.stat(path).st_mtime))
        damaged = problem(UNREADABLE, "damaged_picture", retry_when="file_changed")
        self.run_with([("indexed", "ok"), damaged, ("indexed", "ok"), ("indexed", "ok")])
        row = self._rows()["b.png"]
        self.assertEqual((row["status"], row["park_kind"]), ("parked", "unreadable"))

    def test_a_file_still_being_written_stays_queued_and_loses_no_attempt(self):
        busy = ImageProblem(BUSY, "changed_since_scan", park=False, report=False)
        self.run_with([("indexed", "ok"), busy, ("indexed", "ok"), ("indexed", "ok")])
        row = self._rows()["b.png"]
        self.assertEqual((row["status"], row["attempts"], row["error_id"], row["park_kind"]), ("queued", 0, "changed_since_scan", None))
        self.assertEqual(get_exhausted_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS), [])  # nothing to tell the user: it is tried again later

    def test_trouble_with_the_machine_several_times_in_a_row_stops_the_run_and_parks_nothing_that_is_not_the_files_fault(self):
        short = AnalysisFailed("out_of_memory", "model_out_of_memory", [])
        stop = self.run_with([short] * (4 * config.SERVICE_TRIES))
        self.assertEqual(stop.code, 1)
        rows = self._rows()
        self.assertEqual(sum(1 for r in rows.values() if r["error_id"]), config.MAX_CONSECUTIVE_SERVICE_FAILURES)
        self.assertTrue(all(r["status"] == "queued" and r["attempts"] == 0 for r in rows.values()))

    def test_images_that_failed_for_the_computers_trouble_are_offered_a_try_with_lower_settings_after_the_run(self):
        self.run_with([AnalysisFailed("out_of_memory", "model_out_of_memory", [])] * (4 * config.SERVICE_TRIES))
        self.offer.assert_called_once()
        failures = self.offer.call_args.args[3]
        self.assertEqual([(Path(row["file_path"]).name, kind) for row, kind, _ in failures],
                         [("a.png", "out_of_memory"), ("b.png", "out_of_memory"), ("c.png", "out_of_memory")])
        self.assertIn("Close other apps", failures[0][2])  # the message the user was shown for it

    def test_each_image_is_offered_with_the_trouble_it_had(self):
        self.run_with([AnalysisFailed("timeout", "model_timeout", [])] * config.SERVICE_TRIES
                      + [AnalysisFailed("out_of_memory", "model_out_of_memory", [])] * config.SERVICE_TRIES + [("indexed", "ok")] * 2)
        self.assertEqual([kind for _, kind, _ in self.offer.call_args.args[3]], ["timeout", "out_of_memory"])

    def test_no_offer_when_every_image_was_indexed_or_the_trouble_was_the_images(self):
        self.run_with([("indexed", "ok")] * 4)
        self.run_with([problem(UNREADABLE, "damaged_picture", retry_when="file_changed")] * 4)
        self.offer.assert_not_called()

    def test_no_offer_when_the_run_stopped_for_a_reason_lower_settings_would_not_help(self):
        crash = AnalysisFailed("server_crash", "model_stopped", [])
        self.run_with([crash] * config.SERVICE_TRIES + [RunStopped("The embedding model has changed.")])
        self.offer.assert_not_called()

    def test_no_offer_when_the_database_could_not_be_written(self):
        crash = AnalysisFailed("server_crash", "model_stopped", [])
        self.run_with([crash] * config.SERVICE_TRIES + [sqlite3.OperationalError("database or disk is full")])
        self.offer.assert_not_called()

    def test_timeouts_in_a_row_stop_the_run_too_and_the_files_stay_queued_for_the_next_run(self):
        stop = self.run_with([AnalysisFailed("timeout", "model_timeout", [])] * (4 * config.SERVICE_TRIES))
        self.assertEqual(stop.code, 1)
        failed = [r for r in self._rows().values() if r["error_id"]]
        self.assertEqual(len(failed), config.MAX_CONSECUTIVE_SERVICE_FAILURES)
        self.assertTrue(all(r["status"] == "queued" and r["attempts"] == 0 for r in failed))

    def test_trouble_with_the_machine_is_tried_again_right_away_and_a_try_that_works_ends_it(self):
        crash = AnalysisFailed("server_crash", "model_stopped", [])
        self.assertIsNone(self.run_with([crash, crash, ("indexed", "ok")] + [("indexed", "ok")] * 3))
        self.assertEqual(self._rows(), {})  # every image was indexed, the first one on its third try
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM failure_history").fetchone()[0], 0)

    def test_after_every_try_the_user_is_told_what_happened_and_what_to_do_and_the_file_stays_queued(self):
        crash = AnalysisFailed("server_crash", "model_stopped", [])
        self.assertIsNone(self.run_with([crash] * config.SERVICE_TRIES + [("indexed", "ok")] * 3))
        row = self._rows()["a.png"]
        self.assertEqual((row["status"], row["attempts"], row["park_kind"]), ("queued", 0, "server_crash"))  # tried by the next run too
        self.assertEqual((row["error_id"], json.loads(row["error_params"])), ("model_stopped", {"tries": config.SERVICE_TRIES}))
        self.assertNotIn("Ollama stopped responding", str(dict(row)))  # only the id is stored, not the wording
        message = describe(row["error_id"], json.loads(row["error_params"]))
        self.assertIn(f"tried {config.SERVICE_TRIES} times", message)
        self.assertIn("Close other apps", message)
        self.assertEqual([f["file_path"] for f in get_exhausted_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS)], [str(self.work / "a.png")])

    def test_what_a_failed_try_was_is_shown_while_the_next_one_runs(self):
        crash = AnalysisFailed("timeout", "model_timeout", [])
        stages = []
        outcomes = [crash, ("indexed", "ok")]

        def index_image(conn, client, settings, root_id, path, on_stage, on_retry):
            on_stage("analysing image")
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        with mock.patch.object(scan, "index_image", side_effect=index_image):
            scan._index_with_tries(self.conn, object(), {}, 1, self.work / "a.png", stages.append, lambda kind: None)
        self.assertEqual(stages[0], "analysing image")
        self.assertTrue(stages[1].startswith("timeout: trying again, 2 of "), stages)
        self.assertTrue(stages[2].startswith("timeout: trying again, 2 of ") and stages[2].endswith("analysing image"), stages)

    def test_a_picture_too_large_for_the_free_memory_stays_queued_and_is_reported_until_a_run_indexes_it(self):
        large = problem(TOO_LARGE, "picture_too_large", params={"megapixels": "900"}, park=False)
        self.run_with([("indexed", "ok"), large, ("indexed", "ok"), ("indexed", "ok")])
        row = self._rows()["b.png"]
        self.assertEqual((row["status"], row["attempts"], row["park_kind"]), ("queued", 0, "too_large"))
        self.assertEqual(len(get_exhausted_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS)), 1)
        self.run_with([("indexed", "ok")])  # the next run: the memory is free now
        self.assertEqual(get_exhausted_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS), [])

    def test_a_problem_of_the_files_does_not_count_towards_stopping(self):
        damaged = problem(UNREADABLE, "damaged_picture", retry_when="file_changed")
        self.assertIsNone(self.run_with([damaged] * 4))
        self.assertTrue(all(r["status"] == "parked" for r in self._rows().values()))

    def test_files_that_cannot_be_read_one_after_another_stop_the_run_and_keep_their_attempts(self):
        stop = self.run_with([PermissionError(13, "denied")] * 4)
        self.assertEqual(stop.code, 1)
        rows = self._rows()
        self.assertTrue(all(r["attempts"] == 0 for r in rows.values()))  # given back: a drive that went away is not their fault
        self.assertEqual(sum(1 for r in rows.values() if r["error_id"]), config.MAX_CONSECUTIVE_SERVICE_FAILURES)

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
        self.assertTrue(all(r["attempts"] == 0 and r["status"] == "queued" and not r["error_id"] for r in rows.values()))

    def test_missing_ocr_models_stop_the_run_before_any_image(self):
        self.ocr_check.side_effect = OSError("no internet")
        with mock.patch.object(scan, "index_image") as index_image:
            with self.assertRaisesRegex(SystemExit, "text-reading"):
                scan._index_backlog()
        index_image.assert_not_called()
        self.assertEqual(len(self._attempts()), 4)

    def test_a_second_run_at_the_same_time_is_refused(self):
        with scan.single_instance("indexing", "Indexing"):
            with self.assertRaisesRegex(SystemExit, "already running"):
                scan.run()



class LowerSettingsStepsTests(IsolatedHome):
    """What is changed, in what order, for an image by what went wrong with it."""

    def steps(self, kind, long_side=None, **device):
        for name, value in {"OLLAMA_TIMEOUT": 300, "OLLAMA_USE_MMAP": False, "IMAGE_MAX_LONG_EDGE": 1024, **device}.items():
            setattr(config, name, value)
        return scan.lower_settings_steps(kind, long_side)

    def test_no_reply_in_time_first_waits_longer_and_then_asks_for_smaller_pictures(self):
        steps = self.steps("timeout")
        self.assertEqual(steps[0], {"timeout_seconds": 300 * config.OLLAMA_TIMEOUT_STEP ** config.OLLAMA_TIMEOUT_RAISES})
        self.assertEqual([step.get("image_long_edge") for step in steps], [None, 768, 576, 432])
        self.assertTrue(all(step["timeout_seconds"] == steps[0]["timeout_seconds"] for step in steps))  # the longer wait is kept

    def test_running_out_of_memory_first_pages_the_model_in_from_disk_and_then_asks_for_smaller_pictures(self):
        for kind in ("out_of_memory", "server_crash"):
            steps = self.steps(kind)
            self.assertEqual(steps[0], {"use_mmap": True})
            self.assertEqual([step.get("image_long_edge") for step in steps], [None, 768, 576, 432])
            self.assertFalse(any("timeout_seconds" in step for step in steps))

    def test_a_model_that_is_already_paged_in_goes_straight_to_smaller_pictures(self):
        steps = self.steps("out_of_memory", OLLAMA_USE_MMAP=True)
        self.assertEqual(steps[0], {"image_long_edge": 768})
        self.assertFalse(any("use_mmap" in step for step in steps))

    def test_sizes_never_go_below_what_calibration_goes_down_to(self):
        edges = [step["image_long_edge"] for step in self.steps("out_of_memory") if "image_long_edge" in step]
        self.assertTrue(all(edge >= config.CALIBRATION_MIN_EDGE for edge in edges))

    def test_sizes_start_below_the_pictures_own_size_because_a_larger_one_would_change_nothing(self):
        steps = self.steps("out_of_memory", long_side=800, OLLAMA_USE_MMAP=True)
        self.assertEqual([step["image_long_edge"] for step in steps], [600, 450, 337])
        self.assertEqual(self.steps("out_of_memory", long_side=400, OLLAMA_USE_MMAP=True), [])  # nothing lower to ask for

    def test_the_usual_settings_are_not_touched_by_working_out_the_steps(self):
        before = config.device_settings()
        self.steps("timeout")
        self.assertEqual(config.device_settings(), {**before})


class IndexWithLowerSettingsTests(IsolatedHome):
    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        self.row = {"file_path": str(self.work / "a.png"), "root_id": 1}  # no file: its size is not known, so the whole ladder is tried
        self.settings = {"vlm_model": "v", "embedding_model": "e"}
        self.seen = []  # the settings in place at each try
        config.OLLAMA_TIMEOUT, config.OLLAMA_USE_MMAP, config.IMAGE_MAX_LONG_EDGE = 300, False, 1024
        self.unload = self.patch("unload_model")
        self.make_client = self.patch("make_client")

    def patch(self, name, **kwargs):
        patcher = mock.patch.object(scan, name, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def index_image(self, works_when):
        """An ``index_image`` that works when ``works_when(settings now in place)``, and otherwise fails for lack of memory."""
        def index(conn, client, settings, root_id, path, on_stage, on_retry):
            now = config.device_settings()
            self.seen.append((now, client))
            on_stage("analysing image")
            if not works_when(now):
                raise AnalysisFailed("out_of_memory", "model_out_of_memory", [])
            return "indexed", "a cat"
        return self.patch("index_image", side_effect=index)

    def run_it(self, kind="out_of_memory", stages=None):
        return scan._index_with_lower_settings(self.conn, "client", self.settings, self.row, kind,
                                               (stages if stages is not None else []).append, lambda kind: None)

    def test_the_first_step_that_works_ends_it_and_says_what_was_changed(self):
        self.index_image(lambda now: now["image_long_edge"] <= 576)
        result, description, changes = self.run_it()
        self.assertEqual((result, description), ("indexed", "a cat"))
        self.assertEqual(changes, {"use_mmap": True, "image_long_edge": 576})
        self.assertEqual([(now["use_mmap"], now["image_long_edge"]) for now, _ in self.seen], [(True, 1024), (True, 768), (True, 576)])

    def test_the_usual_settings_come_back_whether_it_worked_or_not(self):
        usual = config.device_settings()
        index = self.index_image(lambda now: False)
        self.assertIsNone(self.run_it())
        self.assertEqual(config.device_settings(), usual)
        self.assertEqual(index.call_count, 4)  # paged in, 768, 576, 432
        self.assertEqual(config.LLM_OPTIONS["use_mmap"], usual["use_mmap"])  # the options sent to Ollama too
        self.index_image(lambda now: True)
        self.run_it()
        self.assertEqual(config.device_settings(), usual)

    def test_a_failure_that_is_not_the_computers_is_raised_and_the_usual_settings_come_back(self):
        usual = config.device_settings()
        reply = AnalysisFailed("invalid_reply", "reply_invalid", [])
        self.patch("index_image", side_effect=reply)
        with self.assertRaises(AnalysisFailed):
            self.run_it()
        self.assertEqual(config.device_settings(), usual)

    def test_the_models_memory_is_freed_before_each_try_for_lack_of_memory(self):
        self.index_image(lambda now: now["image_long_edge"] <= 768)
        self.run_it()
        self.assertEqual([call.args[1] for call in self.unload.call_args_list], ["v", "e", "v", "e"])  # two tries

    def test_no_reply_in_time_is_tried_with_a_client_that_waits_longer_and_the_models_are_not_unloaded(self):
        self.index_image(lambda now: now["timeout_seconds"] >= 1200)
        self.run_it("timeout")
        self.make_client.assert_called_once_with(timeout=1200)
        self.assertIs(self.seen[0][1], self.make_client.return_value)
        self.unload.assert_not_called()

    def test_what_is_being_tried_is_shown(self):
        self.index_image(lambda now: now["image_long_edge"] <= 768)
        stages = []
        self.run_it(stages=stages)
        self.assertTrue(any(stage.startswith("lower settings (model paged in from disk) · ") for stage in stages), stages)
        self.assertTrue(any("pictures up to 768 px" in stage for stage in stages), stages)

    def test_an_image_with_nothing_lower_to_try_gives_none(self):
        self.row = {"file_path": str(self.make_image(self.work / "tiny.png")), "root_id": 1}  # 8 pixels: nothing smaller to ask for
        config.OLLAMA_USE_MMAP = True
        index = self.index_image(lambda now: True)
        self.assertIsNone(self.run_it())
        index.assert_not_called()


class OfferLowerSettingsTests(IsolatedHome):
    """The question to the user, and what is done with the answer."""

    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        with self.conn:
            root_id = get_or_add_root(self.conn, str(self.work))
            for name in ("a.png", "b.png"):
                upsert_backlog(self.conn, str(self.work / name), root_id, "t")
        rows = {Path(r["file_path"]).name: dict(r) for r in self.conn.execute("SELECT * FROM backlog")}
        self.failures = [(rows["a.png"], "out_of_memory", "Ollama ran out of memory. Close other apps."),
                         (rows["b.png"], "timeout", "The model did not reply in time.")]
        config.OLLAMA_TIMEOUT, config.OLLAMA_USE_MMAP, config.IMAGE_MAX_LONG_EDGE = 300, False, 1024
        self.answer = True
        self.tty = True
        self.seen = []
        patchers = (mock.patch.object(scan, "unload_model"), mock.patch.object(scan, "make_client"),
                    mock.patch.object(scan, "Progress"),
                    mock.patch.object(scan.questionary, "confirm", side_effect=lambda *a, **k: SimpleNamespace(ask=lambda: self.answer)),
                    mock.patch.object(sys, "stdin", SimpleNamespace(isatty=lambda: self.tty)))
        mocks = [patcher.start() for patcher in patchers]
        for patcher in patchers:
            self.addCleanup(patcher.stop)
        self.progress, self.confirm = mocks[2].return_value, mocks[3]

    def index_image(self, outcome=lambda path, now: ("indexed", "a cat")):
        def index(conn, client, settings, root_id, path, on_stage, on_retry):
            now = config.device_settings()
            self.seen.append((path.name, now))
            result = outcome(path, now)
            if isinstance(result, Exception):
                raise result
            return result
        patcher = mock.patch.object(scan, "index_image", side_effect=index)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def offer(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            scan._offer_lower_settings(self.conn, "client", {"vlm_model": "v", "embedding_model": "e"}, self.failures, None)
        return out.getvalue()

    def queued(self):
        return sorted(Path(r["file_path"]).name for r in get_backlog(self.conn, config.BACKLOG_MAX_ATTEMPTS))

    def test_the_user_is_told_which_images_and_why_before_being_asked(self):
        self.answer = False
        shown = self.offer()
        self.assertIn("2 image(s) were not indexed", shown)
        self.assertIn("a.png", shown)
        self.assertIn("Ollama ran out of memory", shown)
        self.assertIn("The model did not reply in time", shown)
        self.confirm.assert_called_once()
        self.assertFalse(self.confirm.call_args.kwargs["default"])  # Enter alone never lowers anything

    def test_a_no_tries_nothing_and_leaves_both_images_queued(self):
        self.answer = False
        index = self.index_image()
        shown = self.offer()
        index.assert_not_called()
        self.assertEqual(self.queued(), ["a.png", "b.png"])
        self.assertIn("usual settings", shown)

    def test_a_yes_tries_each_image_with_the_settings_for_what_went_wrong_with_it(self):
        self.index_image()
        shown = self.offer()
        (first_name, first), (second_name, second) = self.seen
        self.assertEqual((first_name, first["use_mmap"], first["timeout_seconds"]), ("a.png", True, 300))  # memory: paged in from disk
        self.assertEqual((second_name, second["use_mmap"], second["timeout_seconds"]), ("b.png", False, 1200))  # no reply: a longer wait
        self.assertEqual(self.queued(), [])  # both were indexed, so no longer queued
        self.assertIn("2 of 2 image(s) indexed with lower settings", shown)
        self.assertEqual(config.device_settings()["timeout_seconds"], 300)  # nothing stays lowered

    def test_one_image_getting_lower_settings_does_not_lower_them_for_the_next(self):
        self.index_image(lambda path, now: ("indexed", "ok"))
        self.offer()
        self.assertFalse(self.seen[1][1]["use_mmap"])  # a.png's paged-in loading was not carried over to b.png

    def test_an_image_that_still_fails_stays_queued_and_the_others_go_on(self):
        self.index_image(lambda path, now: AnalysisFailed("out_of_memory", "model_out_of_memory", []) if path.name == "a.png" else ("indexed", "ok"))
        shown = self.offer()
        self.assertEqual(self.queued(), ["a.png"])
        self.assertIn("1 of 2 image(s) indexed with lower settings", shown)
        results = [call.args[0] for call in self.progress.done.call_args_list]
        self.assertEqual(results, ["FAILED", "indexed"])

    def test_an_image_whose_lower_try_fails_for_the_models_reply_is_set_aside_like_in_the_run(self):
        self.index_image(lambda path, now: AnalysisFailed("cut_off", "reply_cut_off", []) if path.name == "a.png" else ("indexed", "ok"))
        self.offer()
        row = {Path(r["file_path"]).name: dict(r) for r in self.conn.execute("SELECT * FROM backlog")}["a.png"]
        self.assertEqual((row["status"], row["park_kind"]), ("parked", "cut_off"))

    def test_without_a_terminal_nothing_is_asked_or_tried(self):
        self.tty = False
        index = self.index_image()
        shown = self.offer()
        self.confirm.assert_not_called()
        index.assert_not_called()
        self.assertIn("scan again in a terminal", shown)
        self.assertEqual(self.queued(), ["a.png", "b.png"])

    def test_a_long_list_is_cut_short_with_how_many_more(self):
        self.answer = False
        self.failures = [(self.failures[0][0], "out_of_memory", "memory")] * 13
        self.assertIn("… and 3 more.", self.offer())


if __name__ == "__main__":
    unittest.main()
