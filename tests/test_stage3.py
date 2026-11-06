"""Versions, the embedding model that must not change, models that are local only, and the commands for set-aside and re-embedding."""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import config
from db.operations import (
    add_failure_history, get_backlog, get_model_settings, get_or_add_root, list_parked, park_backlog, requeue_changed_parked,
    requeue_parked, set_device_settings, set_model_settings, update_embedding_settings, upsert_backlog, upsert_image_data, upsert_text_embedding,
)
from install import install
from pipeline import versions as versions_module
from pipeline.serialisation import serialize_embedding
from reembed import reembed
from retry_failed import retry_failed
from tests.support import DEVICE, IsolatedHome
from vlm import client as vlm_client
from vlm.client import NotReady, check_ready, pin_model, pinned_name

SETTINGS = {"vlm_model": "v:latest", "embedding_model": "e:latest", "judge_model": "j:latest",
            "vlm_digest": "vd", "embedding_digest": "ed", "judge_digest": "jd", "embedding_dim": 3}


def models(**digests):
    return SimpleNamespace(models=[SimpleNamespace(model=name, digest=digest) for name, digest in digests.items()])


class ReadyClient:
    """A stand-in client that lists the given models and says what the vision model can do."""

    def __init__(self, listing, capabilities=("completion", "vision")):
        self.listing, self.capabilities, self.copied = listing, list(capabilities), []

    def list(self):
        return self.listing

    def show(self, name):
        return SimpleNamespace(capabilities=self.capabilities)

    def copy(self, source, destination):
        self.copied.append((source, destination))
        self.listing.models.append(SimpleNamespace(model=destination, digest=next(m.digest for m in self.listing.models if m.model == source)))


class ReadinessTests(IsolatedHome):
    def ready(self, client, roles, host="http://localhost:11434", settings=SETTINGS):
        conn = self.connect()
        with conn:
            set_model_settings(conn, settings)
            set_device_settings(conn, DEVICE, "vd")
        said = []
        with mock.patch.object(config, "OLLAMA_HOST", host), mock.patch.object(vlm_client, "make_client", return_value=client), \
                mock.patch.object(vlm_client, "server_version", return_value=(0, 9, 0)):
            return check_ready(conn, roles, say=said.append), said

    def test_a_changed_embedding_model_blocks_with_the_way_out(self):
        client = ReadyClient(models(**{"v:latest": "vd", "e:latest": "NEW"}))
        with self.assertRaises(NotReady) as raised:
            self.ready(client, ("embedding",))
        message = str(raised.exception)
        self.assertIn("not the same version", message)
        self.assertIn("Re-embed images", message)

    def test_a_changed_vision_model_only_warns(self):
        client = ReadyClient(models(**{"v:latest": "NEW", "e:latest": "ed"}))
        _, said = self.ready(client, ("vlm",))
        self.assertTrue(any("not the same version" in line for line in said))

    def test_an_unchanged_model_says_nothing(self):
        client = ReadyClient(models(**{"v:latest": "vd", "e:latest": "ed"}))
        _, said = self.ready(client, ("vlm", "embedding"))
        self.assertEqual(said, [])

    def test_a_model_that_cannot_see_images_is_refused(self):
        client = ReadyClient(models(**{"v:latest": "vd"}), capabilities=["completion"])
        with self.assertRaisesRegex(NotReady, "cannot look at images"):
            self.ready(client, ("vlm",))

    def test_an_ollama_that_cannot_say_what_a_model_can_do_is_trusted(self):
        client = ReadyClient(models(**{"v:latest": "vd"}))
        client.show = mock.Mock(side_effect=RuntimeError("too old"))
        self.ready(client, ("vlm",))  # setup already checked it; no error

    def test_an_ollama_on_another_computer_is_refused(self):
        client = ReadyClient(models(**{"v:latest": "vd"}))
        with self.assertRaisesRegex(NotReady, "on this computer"):
            self.ready(client, ("vlm",), host="http://192.168.1.20:11434")


class PinTests(unittest.TestCase):
    def test_the_pinned_name_follows_the_weights(self):
        self.assertEqual(pinned_name("bge-m3:latest", "sha256:3a1f9c2d7e11aabbcc"), "imagent-pin/bge-m3:3a1f9c2d7e11")
        self.assertEqual(pinned_name("hf.co/org/model:Q4", "abcdef0123456789"), "imagent-pin/model:abcdef012345")
        self.assertTrue(vlm_client.is_pinned("imagent-pin/x:1"))
        self.assertFalse(vlm_client.is_pinned("x:1"))

    def test_a_model_is_copied_once(self):
        client = ReadyClient(models(**{"e:latest": "sha256:abc123abc123xx"}))
        name, digest = pin_model(client, "e:latest")
        self.assertEqual((name, digest), ("imagent-pin/e:abc123abc123", "sha256:abc123abc123xx"))
        self.assertEqual(client.copied, [("e:latest", name)])
        pin_model(client, "e:latest")
        self.assertEqual(len(client.copied), 1)  # the copy exists already


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

    def test_the_locked_embedding_model_can_be_replaced_only_by_the_dedicated_call(self):
        with self.conn:
            set_model_settings(self.conn, SETTINGS)
            update_embedding_settings(self.conn, "imagent-pin/e:123", "newdigest", 5)
        found = get_model_settings(self.conn)
        self.assertEqual((found["embedding_model"], found["embedding_digest"], found["embedding_dim"]), ("imagent-pin/e:123", "newdigest", 5))


class IndexLoopVersionsTests(IsolatedHome):
    """A file parked for its model's replies is parked with the versions of the time, and queued again when they change."""

    def test_a_changed_prompt_queues_the_parked_file_again(self):
        from scan_and_index import scan_and_index
        from vlm.analyzer import AnalysisFailed
        conn = self.connect()
        with conn:
            root = get_or_add_root(conn, str(self.work))
            upsert_backlog(conn, "/a.png", root, "t")
        versions = {"vlm_digest": "1", "prompt_hash": "p1", "schema_hash": "s", "embedding_digest": "e"}
        settings = {"vlm_model": "v", "judge_model": "j", "embedding_model": "e", "embedding_dim": 3}
        with contextlib.redirect_stdout(io.StringIO()) as out, mock.patch.object(scan_and_index, "check_ready", return_value=(object(), settings)), \
                mock.patch.object(scan_and_index, "unload_model"), mock.patch.object(scan_and_index, "Progress"), \
                mock.patch.object(scan_and_index.ocr, "check_available"), \
                mock.patch.object(scan_and_index, "collect_versions", side_effect=[versions, versions, {**versions, "prompt_hash": "p2"}]), \
                mock.patch.object(scan_and_index, "index_image", side_effect=[AnalysisFailed("cut_off", "cut", []), ("indexed", "ok")]):
            scan_and_index._index_backlog()  # parks the file
            self.assertEqual([p["park_kind"] for p in list_parked(conn)], ["cut_off"])
            scan_and_index._index_backlog()  # same versions: nothing is queued
            self.assertEqual(len(list_parked(conn)), 1)
            scan_and_index._index_backlog()  # the prompt changed: queued again and indexed
        self.assertEqual(list_parked(conn), [])
        self.assertIn("queued again", out.getvalue())


class ReembedTests(IsolatedHome):
    def make_database(self, with_old_format=False):
        conn = self.connect()
        with conn:
            set_model_settings(conn, {**SETTINGS, "embedding_model": "imagent-pin/e:old", "embedding_digest": "olddigest"})
            root = get_or_add_root(conn, str(self.work))
            for number, reply in enumerate(({"short_description": "A cat.", "scene": "A cat on a sofa."}, {"short_description": "A dog.", "scene": "A dog."})):
                upsert_image_data(conn, {
                    "root_id": root, "file_path": f"/{number}.png", "hash": "h", "short_description": "s", "description": "d",
                    "vlm_metadata_json": json.dumps({"vlm": reply, "ocr_text": ["EXIT"], "features": {}}), "width": 1, "height": 1,
                    "file_size": 1, "mime_type": None, "created_at": None, "modified_at": "t", "indexed_at": "t",
                    "processing_time_ms": 1, "metadata_version": 1, "vlm_model": "v", "taken_at": None, "camera_make": None,
                    "camera_model": None, "gps_lat": None, "gps_lon": None})
            if with_old_format:
                upsert_image_data(conn, {
                    "root_id": root, "file_path": "/old.png", "hash": "h", "short_description": "s", "description": "d",
                    "vlm_metadata_json": "{}", "width": 1, "height": 1, "file_size": 1, "mime_type": None, "created_at": None,
                    "modified_at": "t", "indexed_at": "t", "processing_time_ms": 1, "metadata_version": 1, "vlm_model": "v",
                    "taken_at": None, "camera_make": None, "camera_model": None, "gps_lat": None, "gps_lon": None})
        return conn

    class Embedder(ReadyClient):
        def __init__(self, dimension=4, fail_on=None):
            super().__init__(models(**{"e:latest": "sha256:newdigest00000000"}))
            self.dimension, self.fail_on, self.calls = dimension, fail_on, 0

        def embed(self, **request):
            self.calls += 1
            if self.fail_on and self.calls == self.fail_on:
                raise RuntimeError("Ollama stopped")
            return SimpleNamespace(embeddings=[[float(len(request["input"][0]))] * self.dimension])

    def run_reembed(self, client, answer="y", record=None, **kwargs):
        settings = get_model_settings(self.connect())
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(reembed, "check_ready", return_value=(client, settings)), \
                mock.patch("builtins.input", return_value=answer), \
                mock.patch.object(config, "load_install_record", return_value={"pinned_models": {"imagent-pin/e:old": "e:latest"}}), \
                mock.patch.object(config, "record_pinned_model") as recorded:
            try:
                reembed.reembed(**kwargs)
                error = None
            except SystemExit as exc:
                error = exc
        return out.getvalue(), recorded, error

    def test_every_vector_is_made_again_with_the_installed_model_and_the_new_model_is_locked(self):
        conn = self.make_database()
        client = self.Embedder(dimension=4)
        out, recorded, error = self.run_reembed(client)
        self.assertIsNone(error)
        self.assertIn("2 image(s) to embed again", out)
        self.assertRegex(out, r"takes about \d+ s")
        rows = conn.execute("SELECT embedding_model, embedding_digest, embedding_dim, length(embedding) FROM text_embeddings").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(tuple(r) == ("imagent-pin/e:newdigest000", "sha256:newdigest00000000", 4, 16) for r in rows))
        settings = get_model_settings(conn)
        self.assertEqual((settings["embedding_model"], settings["embedding_dim"], settings["embedding_digest"]),
                         ("imagent-pin/e:newdigest000", 4, "sha256:newdigest00000000"))
        recorded.assert_called_once_with("imagent-pin/e:newdigest000", "e:latest")  # the new copy is known to the uninstaller
        self.assertEqual(conn.execute("SELECT DISTINCT search_text_version FROM image_data").fetchall()[0][0], 1)

    def test_it_says_how_long_it_will_take_and_changes_nothing_if_declined(self):
        conn = self.make_database()
        out, _, _ = self.run_reembed(self.Embedder(), answer="n")
        self.assertIn("Nothing was changed", out)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM text_embeddings").fetchone()[0], 0)
        self.assertEqual(get_model_settings(conn)["embedding_digest"], "olddigest")

    def test_an_interrupted_run_changes_nothing(self):
        conn = self.make_database()
        _, _, error = self.run_reembed(self.Embedder(fail_on=4), assume_yes=True)  # 2 calls for the estimate, then the second image fails
        self.assertIn("Nothing was changed", str(error))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM text_embeddings").fetchone()[0], 0)
        self.assertEqual(get_model_settings(conn)["embedding_digest"], "olddigest")

    def test_images_without_a_stored_reply_are_reported_and_left_alone(self):
        conn = self.make_database(with_old_format=True)
        out, _, _ = self.run_reembed(self.Embedder(), assume_yes=True)
        self.assertIn("1 image(s) have no stored reply", out)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM text_embeddings").fetchone()[0], 2)

    def test_the_search_text_is_the_one_indexing_would_build(self):
        from text_processing.formatters import build_search_text
        records = [{"file_id": 1, "vlm_metadata_json": json.dumps({"vlm": {"short_description": "A cat.", "scene": "x"}, "ocr_text": ["EXIT"]})},
                   {"file_id": 2, "vlm_metadata_json": "not json"}]
        texts, without = reembed.search_texts(records)
        self.assertEqual(without, 1)
        self.assertEqual(texts, [(1, build_search_text({"short_description": "A cat.", "scene": "x"}, ["EXIT"], config.SUMMARY_FIELD))])

    def test_a_missing_embedding_model_is_pointed_at_the_installer(self):
        self.make_database()
        client = self.Embedder()
        client.listing = models()  # nothing installed
        with mock.patch.object(config, "load_install_record", return_value={}), self.assertRaises(SystemExit) as raised:
            reembed._model_to_use(client, get_model_settings(self.connect()))
        self.assertIn(config.INSTALL_COMMAND, str(raised.exception))


class RetryFailedTests(IsolatedHome):
    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        with self.conn:
            root = get_or_add_root(self.conn, str(self.work))
            for name, kind in (("a", "cut_off"), ("b", "cut_off"), ("c", "unreadable")):
                upsert_backlog(self.conn, str(self.work / f"{name}.png"), root, "t")
                park_backlog(self.conn, str(self.work / f"{name}.png"), kind, f"{kind} message", "on_request")

    def run_command(self, argv, answers=(), run_now=False):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(sys, "argv", ["retry_failed.py", *argv]), \
                mock.patch("builtins.input", side_effect=list(answers)), \
                mock.patch.object(retry_failed, "_ask_yes_no", return_value=run_now), mock.patch.object(retry_failed, "run") as run:
            try:
                retry_failed.main()
                error = None
            except SystemExit as exc:
                error = exc
        return out.getvalue(), run, error

    def test_one_kind_can_be_given_on_the_command_line(self):
        out, run, _ = self.run_command(["unreadable"])
        self.assertIn("1 image(s) queued again", out)
        self.assertEqual(len(list_parked(self.conn)), 2)

    def test_all_can_be_asked_for_and_indexing_offered(self):
        out, run, _ = self.run_command(["all"], run_now=True)
        self.assertIn("3 image(s) queued again", out)
        run.assert_called_once()

    def test_without_an_argument_the_kind_is_chosen_from_a_list(self):
        out, _, _ = self.run_command([], answers=["1"])  # kinds are sorted: 1 = cut_off
        self.assertIn("2 image(s) queued again", out)
        self.assertEqual([p["park_kind"] for p in list_parked(self.conn)], ["unreadable"])

    def test_the_reason_of_each_kind_is_shown(self):
        out, _, _ = self.run_command(["all"])
        self.assertIn("cut_off message", out)
        self.assertIn("unreadable message", out)

    def test_an_unknown_kind_is_refused_and_nothing_changes(self):
        _, _, error = self.run_command(["nonsense"])
        self.assertIn("cut_off", str(error))
        self.assertEqual(len(list_parked(self.conn)), 3)

    def test_nothing_set_aside_is_said_plainly(self):
        with self.conn:
            requeue_parked(self.conn)
        out, run, _ = self.run_command([])
        self.assertIn("nothing to try again", out)
        run.assert_not_called()


class SetupPinTests(IsolatedHome):
    def test_setup_locks_the_embedding_model_as_a_copy_of_its_own_and_records_it(self):
        client = ReadyClient(models(**{"v:latest": "vd", "e:latest": "sha256:abc123abc123xx", "j:latest": "jd"}))
        captured = {}

        def verify(c, vlm, embedding, text_model):
            captured["embedding"] = embedding
            return {"vlm_model": "v:latest", "embedding_model": embedding}, {}

        @contextlib.contextmanager
        def fake_ollama(host):
            yield client
        with mock.patch.object(install, "ollama_client", fake_ollama), mock.patch.object(install, "_pull_models"), \
                mock.patch.object(install, "verify_models", verify), mock.patch.object(install, "say"), \
                mock.patch.object(install, "measure_computer", return_value=(DEVICE, None)):
            settings, rejected = install.download_and_verify("http://localhost:11434", "v", "e", "j")
        self.assertEqual(captured["embedding"], "imagent-pin/e:abc123abc123")
        self.assertEqual(settings["embedding_model"], "imagent-pin/e:abc123abc123")
        record = config.load_install_record()
        self.assertIn("imagent-pin/e:abc123abc123", record["ollama_models"])  # the uninstaller removes it
        self.assertEqual(record["pinned_models"], {"imagent-pin/e:abc123abc123": "e:latest"})

    def test_a_pinned_copy_that_was_removed_is_made_again_from_the_same_version(self):
        config.save_install_record({"pinned_models": {"imagent-pin/e:abc123abc123": "e:latest"}})
        client = ReadyClient(models(**{"e:latest": "sha256:abc123abc123xx"}))
        with mock.patch.object(install, "say") as say:
            install._reinstall(client, "imagent-pin/e:abc123abc123")
        self.assertEqual(client.copied, [("e:latest", "imagent-pin/e:abc123abc123")])

    def test_a_pinned_copy_whose_original_changed_points_to_re_embedding(self):
        config.save_install_record({"pinned_models": {"imagent-pin/e:abc123abc123": "e:latest"}})
        client = ReadyClient(models(**{"e:latest": "sha256:ffffffffffff"}))
        with mock.patch.object(install, "say") as say:
            install._reinstall(client, "imagent-pin/e:abc123abc123")
        self.assertEqual(client.copied, [])
        self.assertIn("Re-embed images", say.call_args[0][0])


class LaunchersTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent

    def test_every_menu_entry_runs_a_script_that_exists(self):
        import re
        shell = (self.ROOT / "run.sh").read_text()
        batch = (self.ROOT / "run.bat").read_bytes().decode()
        scripts = set(re.findall(r'script="([\w/]+\.py)"', shell)) | {s.replace("\\", "/") for s in re.findall(r'SCRIPT=([\w\\]+\.py)', batch)}
        self.assertEqual(len(scripts), 4)
        for script in scripts:
            self.assertTrue((self.ROOT / script).is_file(), script)

    def test_batch_files_keep_their_windows_line_endings(self):
        for name in ("run.bat", "install.bat", "uninstall/uninstall.bat"):
            data = (self.ROOT / name).read_bytes()
            self.assertEqual(data.count(b"\n"), data.count(b"\r\n"), name)  # a batch file with bare LF can misbehave


if __name__ == "__main__":
    unittest.main()
