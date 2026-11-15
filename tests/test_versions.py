"""What made each record (versions), and when a file set aside for its model's replies is tried again."""

import contextlib
import io
import unittest
from unittest import mock

from imagent import config
from imagent.db.operations import (
    get_backlog,
    get_or_add_root,
    list_parked,
    park_backlog,
    requeue_changed_parked,
    requeue_parked,
    upsert_backlog,
    upsert_image_data,
    upsert_text_embedding,
)
from imagent.pipeline import versions as versions_module
from tests.support import SETTINGS, IsolatedHome, ReadyClient, models


class VersionsTests(unittest.TestCase):
    def test_everything_that_produces_a_record_is_listed(self):
        client = ReadyClient(models(**{"v:latest": "vd", "e:latest": "ed"}))
        with mock.patch.object(versions_module, "server_version", return_value=(0, 9, 1)), mock.patch.object(config, "OLLAMA_HOST", "http://localhost"):
            found = versions_module.collect_versions(client, SETTINGS)
        self.assertEqual((found["vlm_digest"], found["embedding_digest"], found["ollama_version"]), ("vd", "ed", "0.9.1"))
        self.assertEqual(found["app_version"], config.APP_VERSION)
        self.assertEqual(len(found["prompt_hash"]), 64)
        self.assertIn("search_text_version", found)

    def test_changing_the_prompt_or_the_schema_changes_the_fingerprint(self):
        client = ReadyClient(models(**{"v:latest": "vd", "e:latest": "ed"}))
        before = versions_module.collect_versions(client, SETTINGS)
        with mock.patch.object(config, "ANALYSIS_PROMPT", config.ANALYSIS_PROMPT + " More."):
            self.assertNotEqual(versions_module.collect_versions(client, SETTINGS)["prompt_hash"], before["prompt_hash"])
        with mock.patch.object(config, "ANALYSIS_SCHEMA", {**config.ANALYSIS_SCHEMA, "title": "x"}):
            self.assertNotEqual(versions_module.collect_versions(client, SETTINGS)["schema_hash"], before["schema_hash"])

    def test_the_versions_that_decide_a_retry_must_all_be_known(self):
        full = {"vlm_digest": "a", "prompt_hash": "b", "schema_hash": "c", "app_version": "1"}
        self.assertEqual(versions_module.model_versions(full), {"vlm_digest": "a", "prompt_hash": "b", "schema_hash": "c"})
        self.assertIsNone(versions_module.model_versions({**full, "vlm_digest": None}))
        self.assertIsNone(versions_module.model_versions({}))

    def test_a_client_that_cannot_list_models_gives_unknown_digests(self):
        found = versions_module.collect_versions(mock.Mock(list=mock.Mock(side_effect=RuntimeError)), SETTINGS)
        self.assertIsNone(found["vlm_digest"])


class DatabaseTests(IsolatedHome):
    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        with self.conn:
            self.root = get_or_add_root(self.conn, str(self.work))
            for name in ("a", "b", "c"):
                upsert_backlog(self.conn, f"/{name}.png", self.root, "t")

    def test_a_file_parked_for_the_model_is_queued_again_when_the_versions_change(self):
        v1, v2 = {"vlm_digest": "1", "prompt_hash": "p", "schema_hash": "s"}, {"vlm_digest": "2", "prompt_hash": "p", "schema_hash": "s"}
        with self.conn:
            park_backlog(self.conn, "/a.png", "cut_off", "cut", "model_changed", v1)
            park_backlog(self.conn, "/b.png", "unreadable", "damaged", "file_changed", v1)
            self.assertEqual(requeue_changed_parked(self.conn, v1), 0)  # nothing changed
            self.assertEqual(requeue_changed_parked(self.conn, v2), 1)  # only the one parked for the model
        self.assertEqual([r["file_path"] for r in get_backlog(self.conn, 3)], ["/a.png", "/c.png"])
        self.assertEqual([p["file_path"] for p in list_parked(self.conn)], ["/b.png"])

    def test_a_file_parked_without_known_versions_is_tried_again_once_they_are_known(self):
        with self.conn:
            park_backlog(self.conn, "/a.png", "cut_off", "cut", "model_changed", None)
            self.assertEqual(requeue_changed_parked(self.conn, {"vlm_digest": "1", "prompt_hash": "p", "schema_hash": "s"}), 1)

    def test_parked_files_are_queued_again_on_request_with_a_clean_slate(self):
        with self.conn:
            park_backlog(self.conn, "/a.png", "cut_off", "cut", "model_changed")
            park_backlog(self.conn, "/b.png", "unreadable", "damaged", "file_changed")
            park_backlog(self.conn, "/c.png", "unreadable", "damaged", "file_changed")
            self.assertEqual(requeue_parked(self.conn, kind="unreadable"), 2)
            self.assertEqual(requeue_parked(self.conn, paths=[]), 0)
        row = self.conn.execute("SELECT status, attempts, park_kind, last_error FROM backlog WHERE file_path = '/b.png'").fetchone()
        self.assertEqual(tuple(row), ("queued", 0, None, None))
        with self.conn:
            self.assertEqual(requeue_parked(self.conn), 1)  # the rest

    def test_any_number_of_parked_files_can_be_queued_again_by_path(self):
        paths = [f"/many/{n}.png" for n in range(1200)]

        class Counting:
            """Wraps the connection and notes the most ? that any one statement was given."""
            most = 0

            def __init__(self, conn):
                self.conn = conn

            def execute(self, sql, params=()):
                Counting.most = max(Counting.most, len(params))
                return self.conn.execute(sql, params)

        with self.conn:
            for path in paths:
                upsert_backlog(self.conn, path, self.root, "t")
                park_backlog(self.conn, path, "cut_off", "cut", "model_changed")
            self.assertEqual(requeue_parked(Counting(self.conn), paths=paths), 1200)
        self.assertLessEqual(Counting.most, 999)  # the limit of the oldest SQLite Imagent supports
        self.assertEqual(list_parked(self.conn), [])

    def test_a_record_and_its_vector_say_what_made_them(self):
        record = {"root_id": self.root, "file_path": "/a.png", "hash": "h", "short_description": "s", "description": "d",
                  "vlm_metadata_json": "{}", "width": 1, "height": 1, "file_size": 1, "mime_type": "image/png", "created_at": None,
                  "modified_at": "t", "indexed_at": "t", "processing_time_ms": 1, "metadata_version": 1, "vlm_model": "v",
                  "taken_at": None, "camera_make": None, "camera_model": None, "gps_lat": None, "gps_lon": None,
                  "vlm_digest": "vd", "prompt_hash": "ph", "app_version": "9"}
        with self.conn:
            file_id = upsert_image_data(self.conn, record)  # columns not given are left empty
            upsert_text_embedding(self.conn, file_id, b"1234", "e:latest", "ed", 3)
        row = self.conn.execute("SELECT vlm_digest, prompt_hash, schema_hash, app_version FROM image_data").fetchone()
        self.assertEqual(tuple(row), ("vd", "ph", None, "9"))
        row = self.conn.execute("SELECT embedding_model, embedding_digest, embedding_dim FROM text_embeddings").fetchone()
        self.assertEqual(tuple(row), ("e:latest", "ed", 3))


class IndexLoopVersionsTests(IsolatedHome):
    """A file parked for its model's replies is parked with the versions of the time, and queued again when they change."""

    def test_a_changed_prompt_queues_the_parked_file_again(self):
        from imagent.commands import scan
        from imagent.vlm.analyzer import AnalysisFailed
        conn = self.connect()
        with conn:
            root = get_or_add_root(conn, str(self.work))
            upsert_backlog(conn, "/a.png", root, "t")
        versions = {"vlm_digest": "1", "prompt_hash": "p1", "schema_hash": "s", "embedding_digest": "e"}
        settings = {"vlm_model": "v", "judge_model": "j", "embedding_model": "e", "embedding_dim": 3}
        with contextlib.redirect_stdout(io.StringIO()) as out, mock.patch.object(scan, "check_ready", return_value=(object(), settings)), \
                mock.patch.object(scan, "unload_model"), mock.patch.object(scan, "Progress"), \
                mock.patch.object(scan.ocr, "check_available"), \
                mock.patch.object(scan, "collect_versions", side_effect=[versions, versions, {**versions, "prompt_hash": "p2"}]), \
                mock.patch.object(scan, "index_image", side_effect=[AnalysisFailed("cut_off", "cut", []), ("indexed", "ok")]):
            scan._index_backlog()  # parks the file
            self.assertEqual([p["park_kind"] for p in list_parked(conn)], ["cut_off"])
            scan._index_backlog()  # same versions: nothing is queued
            self.assertEqual(len(list_parked(conn)), 1)
            scan._index_backlog()  # the prompt changed: queued again and indexed
        self.assertEqual(list_parked(conn), [])
        self.assertIn("queued again", out.getvalue())


if __name__ == "__main__":
    unittest.main()
