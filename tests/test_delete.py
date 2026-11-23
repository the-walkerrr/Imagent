"""Removing added folders and what was indexed from them (imagent delete)."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

from imagent.commands import delete
from imagent.db.operations import (
    count_deleted_analyses, count_keepable_analyses, get_or_add_root, list_roots, root_stats, set_keep_deleted, upsert_backlog,
)
from imagent.pipeline.locking import AlreadyRunning
from tests.support import IsolatedHome, add_indexed


class DeleteBase(IsolatedHome):
    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        with self.conn:
            set_keep_deleted(self.conn, 1000)  # what setup leaves: the choice of how many deleted pictures to remember
        stdin = mock.Mock()
        stdin.isatty.return_value = True
        patcher = mock.patch.object(sys, "stdin", stdin)
        patcher.start()
        self.addCleanup(patcher.stop)

    def add_folder(self, name: str, pictures: int, embedded: bool = True, shade_from: int = 0) -> int:
        """Add the folder ``name`` with ``pictures`` indexed pictures (with an analysis time and, by default, an embedding)."""
        root = (self.work / name).resolve()
        with self.conn:
            root_id = get_or_add_root(self.conn, str(root))
        for number in range(pictures):
            path = self.make_image(root / f"{number}.png", shade=shade_from + number)
            add_indexed(self.conn, root_id, path)
        with self.conn:
            self.conn.execute("UPDATE image_data SET processing_time_ms = 2000 WHERE root_id = ?", (root_id,))
            if embedded:
                self.conn.execute("INSERT INTO text_embeddings (file_id, embedding) SELECT file_id, x'00' FROM image_data WHERE root_id = ?", (root_id,))
        return root_id

    def folders(self) -> list[str]:
        return sorted(Path(root["path"]).name for root in list_roots(self.conn))

    def images(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM image_data").fetchone()[0]


class MatchFoldersTests(DeleteBase):
    def test_a_path_names_the_added_folder_however_it_is_written(self):
        self.add_folder("pictures", 1)
        roots = list_roots(self.conn)
        folder = self.work / "pictures"
        for written in (str(folder), str(folder) + "/", str(folder / ".." / "pictures")):
            chosen, unknown = delete.match_folders([written], roots)
            self.assertEqual([Path(root["path"]).name for root in chosen], ["pictures"])
            self.assertEqual(unknown, [])

    def test_a_folder_is_chosen_once_and_one_that_was_not_added_is_unknown(self):
        self.add_folder("pictures", 1)
        folder = str(self.work / "pictures")
        chosen, unknown = delete.match_folders([folder, folder, str(self.work / "other")], list_roots(self.conn))
        self.assertEqual(len(chosen), 1)
        self.assertEqual(unknown, [str(self.work / "other")])


class StatsTests(DeleteBase):
    def test_a_folders_numbers(self):
        root_id = self.add_folder("pictures", 3)
        with self.conn:
            upsert_backlog(self.conn, str(self.work / "pictures" / "new.png"), root_id, "2026-01-01T00:00:00")
        stats = root_stats(self.conn, root_id)
        self.assertEqual((stats["images"], stats["analysis_ms"], stats["queued"]), (3, 6000, 1))
        self.assertGreater(stats["bytes"], 0)

    def test_only_analysed_pictures_can_be_kept_and_a_picture_seen_twice_counts_once(self):
        first = self.add_folder("first", 2)
        second = self.add_folder("second", 2, embedded=False, shade_from=10)
        self.assertEqual(count_keepable_analyses(self.conn, [first]), 2)
        self.assertEqual(count_keepable_analyses(self.conn, [second]), 0)
        self.assertEqual(count_keepable_analyses(self.conn, [first, second]), 2)


class MainTests(DeleteBase):
    def run_main(self, paths=(), answers=(), checked=None):
        """Run ``delete.main``; ``answers`` are the choices for the questions, ``checked`` the folders ticked in the list. Returns (code, output, questions)."""
        asked = []
        self.titles = []
        answers = list(answers)

        def choose_folders(conn, roots):  # what the list of folders returns: the ones ticked, or Ctrl+C when ``checked`` is None
            if checked is None:
                sys.exit(f"\n{delete.CANCELLED}")
            return [root for root in roots if root["root_id"] in checked]

        def select(message, **kwargs):
            asked.append((message, [choice.value for choice in kwargs["choices"]]))
            self.titles.append([choice.title for choice in kwargs["choices"]])
            return mock.Mock(ask=lambda: answers.pop(0))

        with mock.patch.object(delete.questionary, "select", side_effect=select), mock.patch.object(delete, "choose_folders", side_effect=choose_folders), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            try:
                delete.main(list(paths))
                code = None
            except SystemExit as exit_info:
                code = exit_info.code
        return code, out.getvalue(), asked

    def test_with_no_terminal_it_asks_for_one(self):
        sys.stdin.isatty.return_value = False
        code, _, _ = self.run_main()
        self.assertIn("interactive terminal", code)

    def test_with_no_folders_added_it_says_so(self):
        code, out, asked = self.run_main()
        self.assertIsNone(code)
        self.assertIn("nothing to delete", out)
        self.assertEqual(asked, [])

    def test_a_path_that_was_not_added_deletes_nothing_and_says_to_quote_paths_with_spaces(self):
        self.add_folder("my pictures", 1)
        code, _, asked = self.run_main([str(self.work / "my"), "pictures"], answers=[])
        self.assertIn("not folders you added", code)
        self.assertIn('put it in quotes: imagent delete "folder path"', code)
        self.assertIn("Nothing was deleted", code)
        self.assertEqual((asked, self.folders()), ([], ["my pictures"]))

    def test_a_path_with_spaces_in_quotes_is_found(self):
        self.add_folder("my pictures", 2)
        code, out, _ = self.run_main([str(self.work / "my pictures")], answers=[delete.KEEP, delete.DELETE])
        self.assertIsNone(code)
        self.assertEqual(self.folders(), [])

    def test_several_paths_remove_several_folders(self):
        for name in ("a", "b", "c"):
            self.add_folder(name, 1, shade_from=ord(name))
        code, out, _ = self.run_main([str(self.work / "a"), str(self.work / "c")], answers=[delete.DISCARD, delete.DELETE])
        self.assertIsNone(code)
        self.assertEqual(self.folders(), ["b"])
        self.assertEqual(self.images(), 1)
        self.assertIn("Removed 2 folders", out)

    def test_without_paths_the_ticked_folders_are_removed(self):
        first = self.add_folder("first", 1)
        self.add_folder("second", 1, shade_from=5)
        code, _, _ = self.run_main(answers=[delete.DISCARD, delete.DELETE], checked=[first])
        self.assertIsNone(code)
        self.assertEqual(self.folders(), ["second"])


    def test_it_shows_what_will_be_lost_before_asking(self):
        self.add_folder("pictures", 3)
        _, out, _ = self.run_main([str(self.work / "pictures")], answers=[delete.DISCARD, delete.EXIT])
        for text in ("Pictures indexed    3", "Analysis time       6s", "Pictures' size", "are not touched"):
            self.assertIn(text, out)

    def test_several_folders_also_show_their_total(self):
        self.add_folder("a", 1)
        self.add_folder("b", 2, shade_from=9)
        _, out, _ = self.run_main([str(self.work / "a"), str(self.work / "b")], answers=[delete.DISCARD, delete.EXIT])
        self.assertIn("All 2 folders\n      Pictures indexed    3", out)

    def test_the_two_questions_are_asked_in_order_and_each_can_exit(self):
        self.add_folder("pictures", 1)
        folder = str(self.work / "pictures")
        code, _, asked = self.run_main([folder], answers=[delete.KEEP, delete.DELETE])
        self.assertEqual([values for _, values in asked], [[delete.KEEP, delete.DISCARD, delete.EXIT], [delete.DELETE, delete.EXIT]])
        self.assertIn("Keep the analysis", asked[0][0])
        self.assertIn("Are you sure", asked[1][0])

    def test_exiting_at_either_question_deletes_nothing(self):
        self.add_folder("pictures", 2)
        folder = str(self.work / "pictures")
        for answers in ([delete.EXIT], [delete.KEEP, delete.EXIT], [delete.DISCARD, delete.EXIT]):
            code, _, _ = self.run_main([folder], answers=answers)
            self.assertEqual(code, f"\n{delete.CANCELLED}")
            self.assertEqual((self.folders(), self.images()), (["pictures"], 2))

    def test_ctrl_c_at_a_question_deletes_nothing(self):
        self.add_folder("pictures", 1)
        code, _, _ = self.run_main([str(self.work / "pictures")], answers=[None])
        self.assertEqual(code, f"\n{delete.CANCELLED}")
        self.assertEqual(self.folders(), ["pictures"])

    def test_nothing_ticked_deletes_nothing(self):
        self.add_folder("pictures", 1)
        code, _, _ = self.run_main(checked=None)
        self.assertEqual(code, f"\n{delete.CANCELLED}")
        self.assertEqual(self.folders(), ["pictures"])

    def test_the_folders_pictures_embeddings_and_queue_go_with_it_and_other_folders_stay(self):
        gone = self.add_folder("gone", 2)
        self.add_folder("stays", 1, shade_from=7)
        with self.conn:
            upsert_backlog(self.conn, str(self.work / "gone" / "new.png"), gone, "2026-01-01T00:00:00")
        self.run_main([str(self.work / "gone")], answers=[delete.DISCARD, delete.DELETE])
        self.assertEqual(self.folders(), ["stays"])
        self.assertEqual(self.images(), 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM text_embeddings").fetchone()[0], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM backlog").fetchone()[0], 0)
        self.assertTrue((self.work / "gone" / "0.png").exists())  # the picture files are never touched


class ChooseFoldersTests(DeleteBase):
    """The list of folders, driven by real key presses on a fake keyboard and screen."""
    ENTER, DOWN, CTRL_C = "\r", "\x1b[B", "\x03"

    def choose(self, *keys):
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        with create_pipe_input() as keyboard, contextlib.redirect_stdout(io.StringIO()):
            keyboard.send_text("".join(keys))
            return delete.choose_folders(self.conn, list_roots(self.conn), input=keyboard, output=DummyOutput())

    def setUp(self):
        super().setUp()
        self.first, self.second = self.add_folder("first", 1), self.add_folder("second", 1, shade_from=5)

    def test_enter_ticks_a_folder_and_confirm_ends_the_choice(self):
        first = list_roots(self.conn)[0]["root_id"]
        self.assertEqual([root["root_id"] for root in self.choose(self.ENTER, self.DOWN, self.DOWN, self.ENTER)], [first])

    def test_enter_ticks_every_folder_it_is_pressed_on(self):
        self.assertEqual(len(self.choose(self.ENTER, self.DOWN, self.ENTER, self.DOWN, self.ENTER)), 2)

    def test_enter_on_a_ticked_folder_unticks_it(self):
        # tick the first, tick the second, go back up and untick the first, then confirm
        chosen = self.choose(self.ENTER, self.DOWN, self.ENTER, "\x1b[A", self.ENTER, self.DOWN, self.DOWN, self.ENTER)
        self.assertEqual([root["root_id"] for root in chosen], [list_roots(self.conn)[1]["root_id"]])

    def test_confirm_cannot_be_chosen_until_a_folder_is_ticked(self):
        # with nothing ticked the pointer skips Confirm: from the first folder, three Downs go to the second folder, to Exit (past Confirm)
        # and back to the first folder, where Enter ticks it; then two Downs reach Confirm, which is now allowed
        chosen = self.choose(self.DOWN, self.DOWN, self.DOWN, self.ENTER, self.DOWN, self.DOWN, self.ENTER)
        self.assertEqual([root["root_id"] for root in chosen], [list_roots(self.conn)[0]["root_id"]])

    def test_exit_below_confirm_deletes_nothing(self):
        for ticked in ((), (self.ENTER,)):  # Exit is offered whether or not a folder is ticked
            downs = (self.DOWN,) * (2 if not ticked else 3)  # with nothing ticked the pointer skips Confirm
            with self.assertRaises(SystemExit) as stop:
                self.choose(*ticked, *downs, self.ENTER)
            self.assertEqual(stop.exception.code, f"\n{delete.CANCELLED}")

    def test_ctrl_c_cancels(self):
        with self.assertRaises(SystemExit) as stop:
            self.choose(self.ENTER, self.CTRL_C)
        self.assertEqual(stop.exception.code, f"\n{delete.CANCELLED}")


class KeepTests(DeleteBase):
    def test_keeping_remembers_the_pictures_up_to_the_number_chosen_at_setup(self):
        self.add_folder("pictures", 3)
        with self.conn:
            set_keep_deleted(self.conn, 2)
        code, out, asked = MainTests.run_main(self, [str(self.work / "pictures")], answers=[delete.KEEP, delete.DELETE])
        self.assertIn("2 of the 3 pictures (the most Imagent was set up to remember)", self.titles[0][0])
        self.assertEqual(count_deleted_analyses(self.conn), 2)
        self.assertIn("Imagent remembers 2 deleted picture(s)", out)

    def test_when_everything_can_be_kept_the_question_says_so(self):
        self.add_folder("pictures", 3)
        MainTests.run_main(self, [str(self.work / "pictures")], answers=[delete.DISCARD, delete.EXIT])
        self.assertIn("all 3 pictures are not analysed again", self.titles[0][0])

    def test_not_keeping_remembers_nothing(self):
        self.add_folder("pictures", 3)
        MainTests.run_main(self, [str(self.work / "pictures")], answers=[delete.DISCARD, delete.DELETE])
        self.assertEqual(count_deleted_analyses(self.conn), 0)

    def test_kept_analyses_are_stored_by_the_pictures_pixels(self):
        self.add_folder("pictures", 2)
        MainTests.run_main(self, [str(self.work / "pictures")], answers=[delete.KEEP, delete.DELETE])
        self.assertEqual(count_deleted_analyses(self.conn), 2)
        pixel_hashes = {row[0] for row in self.conn.execute("SELECT pixel_hash FROM deleted_analyses")}
        self.assertEqual(len(pixel_hashes), 2)

    def test_with_nothing_to_remember_the_keep_question_is_not_asked(self):
        for limit in (None, 0):
            with self.conn:
                self.conn.execute("DELETE FROM index_preferences")
                if limit is not None:
                    set_keep_deleted(self.conn, limit)
            self.add_folder(f"pictures{limit}", 2, shade_from=20 if limit is None else 40)
            code, out, asked = MainTests.run_main(self, [str(self.work / f"pictures{limit}")], answers=[delete.DELETE])
            self.assertEqual(len(asked), 1)
            self.assertIn("Are you sure", asked[0][0])
            self.assertIn("none of this can be kept", out)

    def test_pictures_never_analysed_leave_nothing_to_ask_about(self):
        self.add_folder("pictures", 2, embedded=False)
        code, out, asked = MainTests.run_main(self, [str(self.work / "pictures")], answers=[delete.DELETE])
        self.assertEqual(len(asked), 1)
        self.assertNotIn("none of this can be kept", out)


class LockTests(DeleteBase):
    def test_while_indexing_runs_nothing_is_deleted(self):
        self.add_folder("pictures", 1)
        with mock.patch.object(delete, "single_instance", side_effect=AlreadyRunning("Indexing is already running.")):
            code, _, _ = MainTests.run_main(self, [str(self.work / "pictures")], answers=[delete.DISCARD, delete.DELETE])
        self.assertIn("Indexing is already running. Nothing was deleted.", code)
        self.assertEqual(self.folders(), ["pictures"])


if __name__ == "__main__":
    unittest.main()
