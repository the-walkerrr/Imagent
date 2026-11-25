import unittest
from unittest import mock

from imagent import config
from imagent.commands import search, settings
from imagent.db.connection import DatabaseProblem
from imagent.db.operations import get_keep_deleted, get_search_detail, set_keep_deleted, set_search_detail
from tests.support import IsolatedHome


class PreferenceStorageTests(IsolatedHome):
    def test_nothing_is_chosen_in_a_new_database(self):
        conn = self.connect()
        self.assertEqual((get_keep_deleted(conn), get_search_detail(conn)), (None, None))

    def test_each_setting_is_stored_without_touching_the_others(self):
        conn = self.connect()
        with conn:
            set_search_detail(conn, "full")
        self.assertEqual((get_keep_deleted(conn), get_search_detail(conn)), (None, "full"))  # keep_deleted still "not answered"
        with conn:
            set_keep_deleted(conn, 1000)
        self.assertEqual((get_keep_deleted(conn), get_search_detail(conn)), (1000, "full"))
        with conn:
            set_keep_deleted(conn, 0)
        self.assertEqual((get_keep_deleted(conn), get_search_detail(conn)), (0, "full"))

    def test_values_the_settings_never_offer_are_refused_by_the_database(self):
        import sqlite3
        conn = self.connect()
        for setter, bad in ((set_search_detail, "medium"), (set_keep_deleted, -1)):
            with self.assertRaises(sqlite3.IntegrityError), conn:
                setter(conn, bad)

    def test_lowering_how_many_are_remembered_forgets_the_oldest(self):
        conn = self.connect()
        with conn:
            conn.executemany(
                "INSERT INTO deleted_analyses (pixel_hash, vlm_metadata_json, metadata_version, embedding, deleted_at) VALUES (?, '{}', 1, x'00', ?)",
                [(f"h{n}", f"2026-01-0{n}") for n in range(1, 6)])
            set_keep_deleted(conn, 2)
        self.assertEqual([row[0] for row in conn.execute("SELECT pixel_hash FROM deleted_analyses ORDER BY deleted_at")], ["h4", "h5"])


class ChoicesTests(unittest.TestCase):
    VALUES = {settings.KEEP_DELETED: 500, settings.SEARCH_DETAIL: "short", "remembered": 3}

    def test_the_menu_lists_the_settings_then_exit_then_uninstall(self):
        values = [choice.value for choice in settings.choices(self.VALUES)]
        self.assertEqual(values, [settings.KEEP_DELETED, settings.SEARCH_DETAIL, settings.EXIT, settings.UNINSTALL])

    def test_each_entry_shows_the_value_now_in_place(self):
        titles = [choice.title for choice in settings.choices({**self.VALUES, settings.SEARCH_DETAIL: "full"})]
        self.assertIn("Deleted pictures Imagent remembers: 500", titles)
        self.assertIn("Search result detail: Full (the whole description)", titles)

    def test_a_question_not_answered_yet_is_said_so(self):
        titles = [choice.title for choice in settings.choices({**self.VALUES, settings.KEEP_DELETED: None})]
        self.assertIn("Deleted pictures Imagent remembers: not chosen yet (none)", titles)

    def test_with_an_unreadable_database_uninstall_is_still_offered(self):
        self.assertEqual([choice.value for choice in settings.choices(None)], [settings.EXIT, settings.UNINSTALL])


class ReadSettingsTests(IsolatedHome):
    def test_the_defaults_stand_in_for_what_was_never_chosen(self):
        self.connect().close()
        self.assertEqual(settings.read_settings(), {settings.KEEP_DELETED: None, settings.SEARCH_DETAIL: config.SEARCH_DETAIL_DEFAULT,
                                                     "remembered": 0})

    def test_an_unreadable_database_gives_none(self):
        with mock.patch.object(settings, "init_db", side_effect=DatabaseProblem("damaged")):
            self.assertIsNone(settings.read_settings())


class NoTerminalTests(unittest.TestCase):
    def test_settings_say_they_need_a_terminal_instead_of_failing(self):
        with mock.patch.object(settings, "make_output_safe"), mock.patch.object(settings.sys, "stdin", mock.Mock(isatty=lambda: False)):
            with self.assertRaises(SystemExit) as stop:
                settings.main()
        self.assertIn("interactive terminal", str(stop.exception.code))


class SettingsMenuTests(IsolatedHome):
    """The menu driven with the answers given in turn: the setting to change, the value, and so on, ending with Exit."""

    def run_menu(self, *answers):
        self.questions = []
        self.notes = []
        given = iter(answers)

        def select(question, choices, default=None, **kwargs):
            self.notes.append([c.line for c in choices if isinstance(c, settings.questionary.Separator) and c.line.strip()])
            self.questions.append((question, [c.value for c in choices if isinstance(c, settings.questionary.Choice)], default))
            return next(given)

        def prompt(*a, **k):
            self.controls.append(control := mock.Mock(selected_options=["the default"]))
            window = mock.Mock(content=control)
            return mock.Mock(ask=lambda: select(*a, **k), application=mock.Mock(layout=mock.Mock(find_all_windows=lambda: [window])))

        self.controls = []
        with mock.patch.object(settings.questionary, "select", side_effect=prompt), \
                mock.patch.object(settings.questionary, "print"), mock.patch.object(settings, "make_output_safe"), \
                mock.patch.object(settings.sys, "stdin", mock.Mock(isatty=lambda: True)):
            settings.main()

    def stored(self):
        conn = self.connect()
        try:
            return get_keep_deleted(conn), get_search_detail(conn)
        finally:
            conn.close()

    def test_every_setting_can_be_changed_and_is_stored(self):
        self.run_menu(settings.KEEP_DELETED, 1000, settings.SEARCH_DETAIL, "full", settings.EXIT)
        self.assertEqual(self.stored(), (1000, "full"))

    def test_the_value_in_place_is_the_one_the_list_starts_on(self):
        conn = self.connect()
        with conn:
            set_keep_deleted(conn, 2000)
            set_search_detail(conn, "full")
        conn.close()
        self.run_menu(settings.KEEP_DELETED, 2000, settings.SEARCH_DETAIL, "full", settings.EXIT)
        defaults = [default for question, _, default in self.questions if question != "Settings"]
        self.assertEqual(defaults, [2000, "full"])

    def test_the_default_is_only_pointed_at_never_marked_selected(self):
        self.run_menu(settings.KEEP_DELETED, 500, settings.EXIT)
        self.assertEqual([control.selected_options for control in self.controls], [[]] * 3)

    def test_exit_in_the_deleted_pictures_list_changes_nothing(self):
        self.run_menu(settings.KEEP_DELETED, settings.EXIT, settings.EXIT)
        self.assertEqual(self.questions[1][1][-1], settings.EXIT)
        self.assertEqual(self.stored()[0], None)

    def test_exit_in_the_search_detail_list_changes_nothing(self):
        self.run_menu(settings.SEARCH_DETAIL, settings.EXIT, settings.EXIT)
        self.assertEqual(self.questions[1][1][-1], settings.EXIT)
        self.assertEqual(self.stored()[1], None)

    def test_the_deleted_pictures_question_has_a_small_note_under_it(self):
        self.run_menu(settings.KEEP_DELETED, 500, settings.EXIT)
        self.assertEqual(self.questions[1][0], "How many deleted pictures should Imagent remember?")
        self.assertEqual(self.notes[1], ["After you delete a picture, Imagent can remember what it learned about it.",
                                         "If the picture comes back, it does not need to be analysed again.",
                                         "Once the limit is reached, the oldest are forgotten first."])

    def test_a_number_chosen_before_that_is_not_offered_any_more_stays_in_the_list(self):
        conn = self.connect()
        with conn:
            set_keep_deleted(conn, 250)
        conn.close()
        self.run_menu(settings.KEEP_DELETED, 250, settings.EXIT)
        self.assertIn(250, self.questions[1][1])

    def test_fewer_than_are_remembered_asks_before_forgetting(self):
        conn = self.connect()
        with conn:
            conn.executemany("INSERT INTO deleted_analyses (pixel_hash, vlm_metadata_json, metadata_version, embedding, deleted_at) "
                             "VALUES (?, '{}', 1, x'00', ?)", [(f"h{n}", f"2026-01-0{n}") for n in range(1, 6)])
            set_keep_deleted(conn, 1000)
        conn.close()
        self.run_menu(settings.KEEP_DELETED, 2, False, settings.EXIT)  # refused: nothing changes
        self.assertEqual(self.stored()[0], 1000)
        conn = self.connect()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM deleted_analyses").fetchone()[0], 5)
        conn.close()
        self.run_menu(settings.KEEP_DELETED, 0, True, settings.EXIT)
        self.assertEqual(self.stored()[0], 0)
        conn = self.connect()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM deleted_analyses").fetchone()[0], 0)
        conn.close()

    def test_raising_the_number_asks_nothing_more(self):
        self.run_menu(settings.KEEP_DELETED, 5000, settings.EXIT)
        self.assertEqual(len(self.questions), 3)  # the menu, the number, the menu again

    def test_exit_changes_nothing(self):
        self.run_menu(settings.EXIT)
        self.assertEqual(self.stored(), (None, None))

    def test_uninstall_runs_the_uninstaller_and_ends(self):
        from imagent.commands import uninstall
        with mock.patch.object(uninstall, "main") as run_uninstall:
            self.run_menu(settings.UNINSTALL)
        run_uninstall.assert_called_once_with()
        self.assertEqual(len(self.questions), 1)


class SettingsAreUsedTests(IsolatedHome):
    def test_a_search_shows_the_chosen_detail_unless_the_command_says_otherwise(self):
        self.assertFalse(search._saved_detail_is_full())  # the default is the summary
        conn = self.connect()
        with conn:
            set_search_detail(conn, "full")
        conn.close()
        self.assertTrue(search._saved_detail_is_full())

    def test_the_search_command_uses_the_saved_detail_only_when_not_told(self):
        for full, saved, expected in ((None, True, True), (None, False, False), (True, False, True), (False, True, False)):
            with mock.patch.object(search, "_saved_detail_is_full", return_value=saved), mock.patch.object(search, "start_session", return_value=("s",)), \
                    mock.patch.object(search, "_answer") as answer:
                search.main("cat", full=full)
            answer.assert_called_once_with("cat", ("s",), expected)


class MigrationTests(IsolatedHome):
    def test_a_database_of_version_5_keeps_its_answer_and_gains_the_new_settings(self):
        import sqlite3
        from imagent.db import connection
        self.home.mkdir(parents=True)
        raw = sqlite3.connect(config.DB_PATH)
        raw.executescript(connection._SCHEMA_PATH.read_text(encoding="utf-8"))
        raw.executescript("""
            DROP TABLE index_preferences;
            CREATE TABLE index_preferences (id INTEGER PRIMARY KEY CHECK (id = 1), keep_deleted INTEGER NOT NULL CHECK (keep_deleted >= 0));
            INSERT INTO index_preferences VALUES (1, 750);
            PRAGMA user_version = 5;
        """)
        raw.close()
        conn = self.connect()
        self.assertEqual((get_keep_deleted(conn), get_search_detail(conn)), (750, None))
        with conn:
            set_search_detail(conn, "full")
        self.assertEqual(get_search_detail(conn), "full")


if __name__ == "__main__":
    unittest.main()
