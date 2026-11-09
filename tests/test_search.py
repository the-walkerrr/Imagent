import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from search import search
from db.operations import get_or_add_root, upsert_text_embedding
from pipeline.serialisation import serialize_embedding
from tests.support import IsolatedHome, add_indexed

SETTINGS = {"judge_model": "j", "embedding_model": "e", "embedding_dim": 3}


class SearchTests(IsolatedHome):
    def setUp(self):
        super().setUp()
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(self.work))
        self.paths = {}
        for number, (name, vector) in enumerate((("closest", [1, 0, 0]), ("next", [0.9, 0.1, 0]), ("far", [0, 1, 0]))):
            self.paths[name] = self.make_image(self.work / f"{name}.png", number)
            add_indexed(conn, root_id, self.paths[name])
            with conn:
                file_id = conn.execute("SELECT file_id FROM image_data WHERE file_path = ?", (str(self.paths[name]),)).fetchone()[0]
                upsert_text_embedding(conn, file_id, serialize_embedding(vector))
        for target, replacement in (
            ("check_ready", mock.Mock(return_value=(object(), SETTINGS))),
            ("embed_search_text", mock.Mock(return_value=[1.0, 0.0, 0.0])),
            ("understand_query", mock.Mock(return_value=SimpleNamespace(hypothetical_description="a hypothetical image"))),
            ("SearchProgress", mock.MagicMock()),
        ):
            patcher = mock.patch.object(search, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _names(self, results):
        return [path.stem for result in results for path in [self.paths[next(n for n, p in self.paths.items() if str(p) == result["file_path"])]]]

    def test_results_are_ranked_closest_first(self):
        self.assertEqual(self._names(search.search("anything")), ["closest", "next", "far"])

    def test_files_that_are_gone_are_skipped_not_shown(self):
        self.paths["closest"].unlink()
        results = search.search("anything")
        self.assertEqual(self._names(results), ["next", "far"])

    def test_a_hyde_embedding_of_the_wrong_size_falls_back_to_the_query_alone(self):
        sizes = iter([[1.0, 0.0, 0.0], [1.0, 0.0]])  # the query embeds fine, the HyDE description comes back too short
        with mock.patch.object(search, "embed_search_text", side_effect=lambda *a: next(sizes)):
            results = search.search("anything")
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result["hyde_score"] is None for result in results))

    def test_an_empty_query_is_turned_down_with_an_example_and_asked_again(self):
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["search.py", "   "]), \
                mock.patch("builtins.input", side_effect=["  ", "exit"]) as ask, contextlib.redirect_stdout(out):
            search.main()
        self.assertIn("Type what you are looking for", out.getvalue())
        self.assertEqual(ask.call_count, 2)

    def test_the_session_stays_open_for_more_searches_until_exit(self):
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["search.py", "first"]), \
                mock.patch("builtins.input", side_effect=["second", "exit"]) as ask, \
                mock.patch.object(search, "search", wraps=search.search) as run, contextlib.redirect_stdout(out):
            search.main()
        self.assertEqual([call.args[0] for call in run.call_args_list], ["first", "second"])
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(search.check_ready.call_count, 1)  # Ollama is started once, not per search

    def test_no_terminal_ends_the_session_quietly(self):
        with mock.patch.object(sys, "argv", ["search.py"]), mock.patch("builtins.input", side_effect=EOFError):
            search.main()

    def test_a_search_that_cannot_start_ends_the_session(self):
        with mock.patch.object(search, "check_ready", side_effect=search.NotReady("Run the installer")), \
                mock.patch.object(sys, "argv", ["search.py", "anything"]), mock.patch("builtins.input") as ask:
            search.main()
        ask.assert_not_called()

    def _busy(self, others):
        return mock.patch.object(search, "other_models_running", return_value=others)

    def test_another_running_model_closes_the_search_with_a_warning(self):
        with self._busy(["vision"]), mock.patch.object(search, "load_model") as load, \
                mock.patch.object(search, "is_running", return_value=False):
            self.assertEqual(search.search("anything"), [])
        load.assert_not_called()
        said = search.SearchProgress.return_value.say.call_args.args[0]
        self.assertIn("another model (vision)", said)
        self.assertNotIn("Indexing", said)
        search.embed_search_text.assert_not_called()

    def test_the_warning_mentions_indexing_when_it_is_running_in_the_background(self):
        with self._busy(["vision"]), mock.patch.object(search, "is_running", return_value=True):
            self.assertEqual(search.search("anything"), [])
        self.assertIn("Indexing is running in the background", search.SearchProgress.return_value.say.call_args.args[0])

    def test_the_session_ends_instead_of_asking_for_a_query(self):
        with self._busy(["vision"]), mock.patch.object(search, "is_running", return_value=False), \
                mock.patch.object(sys, "argv", ["search.py"]), mock.patch("builtins.input") as ask:
            search.main()
        ask.assert_not_called()

    def test_no_other_model_lets_the_search_load_its_own(self):
        with self._busy([]), mock.patch.object(search, "load_model") as load:
            self.assertIsNotNone(search.start_session())
        self.assertEqual(load.call_count, 2)


if __name__ == "__main__":
    unittest.main()


class BatchedRankingTests(IsolatedHome):
    """The ranking reads the embeddings a batch at a time: the same answer as comparing them all at once, with memory that does not grow."""

    def fill(self, count, dim, seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        vectors = rng.standard_normal((count, dim)).astype("<f4")
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(self.work))
            conn.executemany("INSERT INTO image_data(root_id, file_path, hash, vlm_metadata_json, modified_at) VALUES (?, ?, 'h', '{}', 't')",
                             [(root_id, f"/{n}.png") for n in range(count)])
            ids = [row[0] for row in conn.execute("SELECT file_id FROM image_data ORDER BY file_id")]
            conn.executemany("INSERT INTO text_embeddings(file_id, embedding) VALUES (?, ?)", [(i, v.tobytes()) for i, v in zip(ids, vectors)])
        return conn, ids, vectors

    def brute_force(self, ids, vectors, queries, keep):
        import numpy as np
        unit = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        q = np.stack([v / (np.linalg.norm(v) + 1e-10) for v in np.array(queries, dtype=np.float32)])
        mean = (unit @ q.T).mean(axis=1)
        return [ids[i] for i in np.argsort(-mean, kind="stable")[:keep]]

    def test_the_closest_images_are_the_same_whatever_the_batch_size(self):
        import numpy as np
        conn, ids, vectors = self.fill(137, 16)
        queries = [np.random.default_rng(5).standard_normal(16).tolist(), np.random.default_rng(6).standard_normal(16).tolist()]
        wanted = self.brute_force(ids, vectors, queries, 10)
        for batch in (1, 7, 50, 137, 1000):
            with mock.patch.object(search.config, "DB_BATCH_ROWS", batch):
                found, similarities, skipped = search._best_matches(conn, 16, queries, 10)
            self.assertEqual(found, wanted, batch)
            self.assertEqual((similarities.shape, skipped), ((10, 2), 0))

    def test_a_single_query_gives_one_column(self):
        conn, ids, vectors = self.fill(20, 8)
        found, similarities, _ = search._best_matches(conn, 8, [vectors[3].tolist()], 3)
        self.assertEqual((found[0], similarities.shape), (ids[3], (3, 1)))
        self.assertAlmostEqual(float(similarities[0, 0]), 1.0, places=5)

    def test_fewer_images_than_asked_for_are_all_returned_and_none_is_no_error(self):
        conn, ids, _ = self.fill(4, 8)
        self.assertEqual(len(search._best_matches(conn, 8, [[1.0] * 8], 103)[0]), 4)
        empty = self.connect()
        empty.execute("DELETE FROM text_embeddings")
        self.assertEqual(search._best_matches(empty, 8, [[1.0] * 8], 5)[0], [])

    def test_an_embedding_of_the_wrong_size_is_skipped_and_counted(self):
        conn, ids, _ = self.fill(5, 8)
        with conn:
            conn.execute("UPDATE text_embeddings SET embedding = x'0000' WHERE file_id = ?", (ids[2],))
        with mock.patch.object(search.config, "DB_BATCH_ROWS", 2):
            found, _, skipped = search._best_matches(conn, 8, [[1.0] * 8], 10)
        self.assertEqual((len(found), skipped), (4, 1))
        self.assertNotIn(ids[2], found)

    def test_memory_does_not_grow_with_the_number_of_images(self):
        import tracemalloc
        conn, ids, _ = self.fill(6000, 256)
        full_matrix = 6000 * 256 * 4
        tracemalloc.start()
        with mock.patch.object(search.config, "DB_BATCH_ROWS", 200):
            search._best_matches(conn, 256, [[1.0] * 256, [0.5] * 256], 103)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        self.assertLess(peak, full_matrix / 3, f"peak {peak} bytes for a matrix of {full_matrix}")

    def test_the_batches_are_read_in_file_id_order_and_none_twice(self):
        from db.operations import iter_embedding_batches
        conn, ids, _ = self.fill(23, 4)
        batches = list(iter_embedding_batches(conn, 5))
        self.assertEqual([len(batch) for batch in batches], [5, 5, 5, 5, 3])
        self.assertEqual([row["file_id"] for batch in batches for row in batch], ids)
