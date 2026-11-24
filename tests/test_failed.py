"""The list of pictures that could not be indexed (imagent failed) and the menu entry that offers it."""

import contextlib
import io
import unittest

from imagent.commands import menu, scan
from imagent.db.operations import get_or_add_root, park_backlog, record_backlog_failure, upsert_backlog
from tests.support import IsolatedHome


class FailedImagesReportTests(IsolatedHome):
    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        with self.conn:
            root_id = get_or_add_root(self.conn, str(self.work))
            for name in ("a.png", "b.png", "c.png"):
                upsert_backlog(self.conn, str(self.work / name), root_id, "t")

    def report(self, limit=10):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            count = scan.print_failed(self.conn, limit)
        return count, out.getvalue()

    def test_nothing_is_said_when_nothing_failed(self):
        self.assertEqual(self.report(), (0, ""))

    def test_each_image_is_listed_with_its_reason_and_what_to_do_about_it(self):
        with self.conn:
            park_backlog(self.conn, str(self.work / "a.png"), "unreadable", "damaged_picture", "file_changed")
            record_backlog_failure(self.conn, str(self.work / "b.png"), "model_stopped", False, kind="server_crash", params={"tries": 3})
        count, text = self.report()
        self.assertEqual(count, 2)
        for expected in ("2 image(s) could not be indexed", "a.png", "damaged or incomplete", "b.png", "Close other apps"):
            self.assertIn(expected, text)
        self.assertNotIn("c.png", text)

    def test_an_image_leaves_the_list_when_it_is_indexed_later(self):
        with self.conn:
            record_backlog_failure(self.conn, str(self.work / "b.png"), "model_stopped", False, kind="server_crash")
        self.assertEqual(self.report()[0], 1)
        with self.conn:
            self.conn.execute("DELETE FROM backlog WHERE file_path = ?", (str(self.work / "b.png"),))  # what a successful index does
        self.assertEqual(self.report()[0], 0)

    def test_a_long_list_is_cut_short_at_the_end_of_a_run_but_complete_on_request(self):
        with self.conn:
            for name in ("a.png", "b.png", "c.png"):
                record_backlog_failure(self.conn, str(self.work / name), "model_timeout", False, kind="timeout")
        self.assertIn("1 more", self.report(limit=2)[1])
        self.assertNotIn("more.", self.report(limit=None)[1])

    def test_the_menu_offers_the_list_only_when_there_is_something_to_show(self):
        for failed, offered in ((0, False), (2, True)):
            scripts = [choice.value for choice in menu.choices(failed)]
            self.assertEqual(menu.FAILED_COMMAND in scripts, offered)
            self.assertEqual(scripts[-1], menu.EXIT)
        titles = [choice.title for choice in menu.choices(2)]
        self.assertIn("Show the 2 image(s) that could not be indexed", titles)

    def test_the_menu_counts_what_the_report_lists(self):
        self.assertEqual(menu.failed_count(), 0)
        with self.conn:
            record_backlog_failure(self.conn, str(self.work / "b.png"), "model_timeout", False, kind="timeout")
        self.assertEqual(menu.failed_count(), 1)


if __name__ == "__main__":
    unittest.main()
