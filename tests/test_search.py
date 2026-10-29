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
