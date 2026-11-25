import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from imagent.commands import search
from imagent.db.operations import get_or_add_root, upsert_text_embedding
from imagent.pipeline.serialisation import serialize_embedding
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

    def _add_copy(self, name: str, shade: int, vector: list[float]):
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(self.work))
        self.paths[name] = self.make_image(self.work / f"{name}.png", shade)
        add_indexed(conn, root_id, self.paths[name])
        with conn:
            file_id = conn.execute("SELECT file_id FROM image_data WHERE file_path = ?", (str(self.paths[name]),)).fetchone()[0]
            upsert_text_embedding(conn, file_id, serialize_embedding(vector))

    def test_copies_of_a_picture_are_one_result_that_says_where_the_others_are(self):
        self._add_copy("copy", 0, [1, 0, 0])  # the same pixels as "closest", in another place
        results = search.search("anything")
        self.assertEqual(len(results), 3)  # three different pictures: closest (with its copy), next, far
        first = results[0]
        self.assertEqual(sorted([first["file_path"], *first["other_paths"]]), sorted([str(self.paths["closest"]), str(self.paths["copy"])]))
        self.assertTrue(all(result["other_paths"] == [] for result in results[1:]))

    def test_a_copy_that_ranks_below_the_last_result_is_still_listed_with_its_picture(self):
        self._add_copy("copy_of_far", 2, [-0.1, 1, 0])  # "far" has shade 2; the copy ranks after it
        results = search.search("anything")
        self.assertEqual(len(results), 3)
        self.assertEqual(results[-1]["other_paths"], [str(self.paths["copy_of_far"])])

    def test_the_result_shows_the_places_of_the_copies(self):
        self._add_copy("copy", 0, [1, 0, 0])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            search._show("q", search.search("anything"))
        self.assertIn(f"Same picture also at: {self.paths['copy']}", out.getvalue().replace("\n   ", " "))

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
        with mock.patch("builtins.input", side_effect=["  ", "exit"]) as ask, contextlib.redirect_stdout(out):
            search.main("   ")
        self.assertIn("Type what you are looking for", out.getvalue())
        self.assertEqual(ask.call_count, 2)

    def test_the_session_stays_open_for_more_searches_until_exit(self):
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=["first", "second", "exit"]) as ask, \
                mock.patch.object(search, "search", wraps=search.search) as run, contextlib.redirect_stdout(out):
            search.main()
        self.assertEqual([call.args[0] for call in run.call_args_list], ["first", "second"])
        self.assertEqual(ask.call_count, 3)
        self.assertEqual(search.check_ready.call_count, 1)  # Ollama is started once, not per search

    def test_a_query_given_on_the_command_line_is_answered_once_and_the_command_ends(self):
        out = io.StringIO()
        with mock.patch("builtins.input") as ask, mock.patch.object(search, "search", wraps=search.search) as run, contextlib.redirect_stdout(out):
            search.main("  red car  ")
        self.assertEqual([call.args[0] for call in run.call_args_list], ["red car"])
        ask.assert_not_called()

    def test_a_search_on_a_terminal_starts_with_a_grey_hint_about_the_detail_flags_and_settings(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        grey, plain, piped = Terminal(), Terminal(), io.StringIO()
        for stream, env in ((grey, {"TERM": "xterm"}), (plain, {"TERM": "dumb"}), (piped, {"TERM": "xterm"})):
            with mock.patch.dict(os.environ, env), mock.patch("builtins.input"), contextlib.redirect_stdout(stream):
                search.main("red car")
        self.assertTrue(grey.getvalue().lstrip("\n").startswith("\033[38;5;244m" + search.DETAIL_HINT))
        self.assertTrue(plain.getvalue().lstrip("\n").startswith(search.DETAIL_HINT))
        self.assertNotIn(search.DETAIL_HINT, piped.getvalue())  # not part of piped results
        for flag in ("--full", "--short"):
            self.assertIn(flag, search.DETAIL_HINT)
        self.assertIn("Settings", search.DETAIL_HINT)

    def test_no_terminal_ends_the_session_quietly(self):
        with mock.patch("builtins.input", side_effect=EOFError):
            search.main()

    def test_a_search_that_cannot_start_ends_the_session(self):
        with mock.patch.object(search, "check_ready", side_effect=search.NotReady("Run the installer")), \
                mock.patch("builtins.input") as ask:
            with self.assertRaises(SystemExit) as stop:  # a failure is not a success: the exit code says so
                search.main("anything")
        self.assertEqual(stop.exception.code, 1)
        ask.assert_not_called()

    def _describe(self, name, **columns):
        """Give the indexed image ``name`` the descriptions and facts that a finished index run stores."""
        values = {"short_description": "a short line", "description": "the long description\n\nsecond paragraph",
                  "vlm_metadata_json": '{"vlm": {"tags": ["dog", "beach"]}}', "width": 4032, "height": 3024, "file_size": 3_250_000,
                  "mime_type": "image/png", "taken_at": "2026-09-06T07:24:12", "camera_make": "iQOO", "camera_model": "Neo9 Pro",
                  "gps_lat": 12.97, "gps_lon": 77.59, **columns}
        conn = self.connect()
        with conn:
            conn.execute(f"UPDATE image_data SET {', '.join(f'{key} = ?' for key in values)} WHERE file_path = ?",
                         (*values.values(), str(self.paths[name])))

    def _shown(self, full=False):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            search._show("anything", search.search("anything"), full)
        return out.getvalue()

    def test_a_result_shows_its_summary_and_facts_but_not_the_long_description_tags_or_scores(self):
        self._describe("closest")
        shown = self._shown()
        self.assertIn("a short line", shown)
        self.assertIn("07:24 AM 6 Sep, 2026 \u00b7 iQOO Neo9 Pro \u00b7 4032\u00d73024 \u00b7 3.1 MB", shown)
        self.assertNotIn("the long description", shown)
        self.assertNotIn("Location", shown)
        self.assertNotIn("tags", shown)
        self.assertNotIn("score", shown)

    def test_a_result_is_titled_with_its_file_name_and_the_link_opens_the_original_path(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        path = self.paths["closest"]
        plain, terminal = io.StringIO(), Terminal()
        for stream, env in ((plain, {}), (terminal, {"TERM": "xterm"})):
            with contextlib.redirect_stdout(stream), mock.patch.dict(os.environ, env):
                search._show("anything", search.search("anything"))
        self.assertIn(f"1. {path.name}\n", plain.getvalue().replace("\033[1m", "").replace("\033[0m", ""))
        self.assertIn(f"\033]8;;{path.absolute().as_uri()}\033\\{path.name}\033]8;;\033\\", terminal.getvalue())

    def test_the_full_path_is_shown_in_the_normal_mode_too_on_its_own_line_after_the_facts(self):
        self._describe("closest")
        lines = self._shown().splitlines()
        first = lines.index(next(line for line in lines if line.startswith("1. ")))
        self.assertEqual(lines[first + 6], f"   {self.paths['closest']}")
        self.assertEqual(lines[first + 7:first + 9], ["", ""])

    def test_the_month_is_its_three_letter_name_and_the_day_is_not_padded(self):
        for taken, shown in (("2024-05-09T08:00:00", "08:00 AM 9 May, 2024"), ("2024-01-31T20:30:00", "08:30 PM 31 Jan, 2024"),
                             ("2024-12-01T13:00:00", "01:00 PM 1 Dec, 2024")):
            self.assertEqual(search._when({"taken_at": taken, "modified_at": None}), shown)

    def test_full_no_longer_shows_the_format(self):
        self._describe("closest")
        self.assertNotIn("Format", self._shown(full=True))
        self.assertNotIn("image/png", self._shown(full=True))

    def test_a_line_that_is_too_long_continues_under_the_same_indent_on_a_terminal(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        words = " ".join(f"word{number}" for number in range(30))
        self._describe("closest", description=words)
        out = Terminal()
        with contextlib.redirect_stdout(out), mock.patch.dict(os.environ, {"TERM": "xterm"}), \
                mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((40, 24))):
            search._show("anything", search.search("anything"), full=True)
        wrapped = [line for line in out.getvalue().splitlines() if "word" in line]
        self.assertGreater(len(wrapped), 2)
        self.assertTrue(all(line.startswith("   word") and len(line) <= 39 for line in wrapped))
        self.assertEqual(" ".join(line.strip() for line in wrapped), words)

    def test_the_query_in_the_heading_is_bold_on_a_terminal_only(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        plain, terminal = io.StringIO(), Terminal()
        for stream, env in ((plain, {}), (terminal, {"TERM": "xterm"})):
            with contextlib.redirect_stdout(stream), mock.patch.dict(os.environ, env):
                search._show("bike", search.search("anything"))
        self.assertIn("Top 3 result(s) for: 'bike'\n", plain.getvalue())
        self.assertIn("Top 3 result(s) for: \033[1m'bike'\033[0m\n", terminal.getvalue())

    def test_the_heading_shows_how_long_the_search_took_when_it_is_known(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            search._show("bike", search.search("anything"), seconds=3.24)
            search._show("bike", [], seconds=0.5)
        self.assertIn("Top 3 result(s) for: 'bike' in 3.2s\n", out.getvalue())
        self.assertIn("No results for: 'bike' in 0.5s\n", out.getvalue())

    def test_the_make_is_not_repeated_when_the_model_already_starts_with_it(self):
        base = {"taken_at": None, "modified_at": None, "width": None, "height": None, "file_size": None}
        for make, model, shown in (("iQOO", "iQOO Neo9 Pro", "iQOO Neo9 Pro"), ("Canon", "CANON EOS R5", "CANON EOS R5"),
                                   ("iQOO", "Neo9 Pro", "iQOO Neo9 Pro"), ("Apple", None, "Apple"), (None, "iPhone 15", "iPhone 15")):
            self.assertEqual(search._facts({**base, "camera_make": make, "camera_model": model}), shown)

    def test_the_time_is_shown_with_am_and_pm(self):
        for taken, shown in (("2026-09-07T14:14:00", "02:14 PM 7 Sep, 2026"), ("2026-09-07T00:05:00", "12:05 AM 7 Sep, 2026"),
                             ("2026-09-07T12:00:00", "12:00 PM 7 Sep, 2026")):
            self.assertEqual(search._when({"taken_at": taken, "modified_at": None}), shown)
        self.assertEqual(search._when({"taken_at": None, "modified_at": "2026-09-07T14:14:00+00:00"}), "modified 7 Sep, 2026")

    def test_the_lines_of_a_result_are_separated_by_a_blank_line_and_results_by_two(self):
        self._describe("closest")
        lines = self._shown().splitlines()
        first = lines.index(next(line for line in lines if line.startswith("1. ")))
        self.assertEqual(lines[first + 1], "")
        self.assertEqual(lines[first + 2], "   a short line")
        self.assertEqual(lines[first + 3], "")
        self.assertIn("\u00b7", lines[first + 4])
        self.assertEqual(lines[first + 5], "")
        self.assertEqual(lines[first + 6], f"   {self.paths['closest']}")
        self.assertEqual(lines[first + 7:first + 9], ["", ""])
        self.assertTrue(lines[first + 9].startswith("2. "))

    def test_full_shows_the_long_description_in_place_of_the_summary_with_the_full_path(self):
        self._describe("closest")
        shown = self._shown(full=True)
        self.assertEqual(shown.count("the long description"), 1)  # only "closest" was described above
        self.assertIn("   second paragraph", shown)
        self.assertNotIn("a short line", shown)
        self.assertIn(f"   {self.paths['closest']}\n", shown)  # the full path, on its own line with no label
        self.assertNotIn("Path", shown)
        self.assertNotIn("Folder", shown)
        self.assertNotIn("Location", shown)
        self.assertNotIn("12.97", shown)
        self.assertNotIn("tags", shown)
        self.assertNotIn("score", shown)

    def test_what_an_image_does_not_have_is_left_out(self):
        self._describe("closest", taken_at=None, camera_make=None, camera_model=None, width=None, height=None, file_size=None,
                       gps_lat=None, gps_lon=None)
        shown = self._shown(full=True)
        self.assertIn("   modified ", shown)
        self.assertNotIn("Location", shown)
        self.assertNotIn("None", shown)

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
                mock.patch("builtins.input") as ask:
            with self.assertRaises(SystemExit) as stop:
                search.main()
        self.assertEqual(stop.exception.code, 1)
        ask.assert_not_called()

    def test_no_other_model_lets_the_search_load_its_own(self):
        with self._busy([]), mock.patch.object(search, "load_model") as load:
            self.assertIsNotNone(search.start_session())
        self.assertEqual(load.call_count, 2)


class EmptyIndexTests(IsolatedHome):
    def test_with_nothing_indexed_ollama_is_not_started_and_no_model_is_loaded(self):
        with mock.patch.object(search, "check_ready") as ready, mock.patch.object(search, "load_model") as load, \
                mock.patch.object(search, "SearchProgress"), mock.patch.object(search, "_offer_scan") as offer:
            self.assertIsNone(search.start_session(exit_on_failure=True))  # nothing to search is not a failure: no exit code
        ready.assert_not_called()
        load.assert_not_called()
        offer.assert_called_once_with()


class LongQueryTests(unittest.TestCase):
    def test_a_very_long_query_is_cut_in_the_heading_of_the_results_only(self):
        query = "word " * 200
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            search._show(query, [])
            search._show(query, [{"file_path": "/p/a.png", "description": "d", "short_description": "s", "width": None, "height": None,
                                  "file_size": None, "taken_at": None, "modified_at": None, "camera_make": None, "camera_model": None, "other_paths": []}])
        self.assertLess(len(out.getvalue()), 800)
        self.assertEqual(out.getvalue().count("…"), 2)


class NoIndexedImagesTests(unittest.TestCase):
    """With nothing indexed, the user is offered a scan or Exit instead of being told a command to run."""

    def offer(self, answer, tty=True, folders=True):
        with mock.patch.object(search.sys.stdin, "isatty", return_value=tty), \
                mock.patch.object(search.questionary, "select") as select, \
                mock.patch("imagent.commands.menu.has_folders", return_value=folders), \
                mock.patch("imagent.commands.scan.add") as add, \
                mock.patch("imagent.commands.scan.main") as scan:
            select.return_value.ask.return_value = answer
            search._offer_scan()
        self.add = add
        return select, scan

    def test_scan_runs_the_scan(self):
        select, scan = self.offer("scan")
        self.assertEqual([choice.title for choice in select.call_args.kwargs["choices"][1:]], ["Scan and index images", "Exit"])
        scan.assert_called_once_with()

    def test_with_no_folder_added_it_offers_to_add_one_instead_of_a_scan(self):
        select, scan = self.offer("add", folders=False)
        self.assertEqual([choice.title for choice in select.call_args.kwargs["choices"][1:]], ["Add a folder", "Exit"])
        self.add.assert_called_once_with()
        scan.assert_not_called()

    def test_exit_runs_nothing(self):
        self.assertFalse(self.offer("exit")[1].called)

    def test_ctrl_c_ends_the_program(self):
        with self.assertRaises(SystemExit):
            self.offer(None)

    def test_no_terminal_means_no_menu(self):
        select, scan = self.offer("scan", tty=False)
        select.assert_not_called()
        scan.assert_not_called()


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
        from imagent.db.operations import iter_embedding_batches
        conn, ids, _ = self.fill(23, 4)
        batches = list(iter_embedding_batches(conn, 5))
        self.assertEqual([len(batch) for batch in batches], [5, 5, 5, 5, 3])
        self.assertEqual([row["file_id"] for batch in batches for row in batch], ids)
