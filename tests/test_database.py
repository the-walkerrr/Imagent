import sqlite3
import unittest
from unittest import mock

import config
from db import connection
from db.connection import SCHEMA_VERSION, DatabaseProblem, describe_db_error, init_db
from tests.support import IsolatedHome


def _tables(conn):
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')")}


class MigrationTests(IsolatedHome):
    def test_new_database_gets_the_latest_schema(self):
        conn = self.connect()
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
        self.assertTrue({"image_data", "backlog", "text_embeddings", "scan_roots", "model_settings"} <= _tables(conn))
        self.assertFalse({"image_search_fts", "image_text_fts"} & _tables(conn))

    def test_opening_twice_changes_nothing(self):
        self.connect().close()
        conn = self.connect()
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)

    def _old_database(self):
        """A database as made before schema versions were recorded: user_version 0, with the full-text tables."""
        self.home.mkdir(parents=True)
        conn = sqlite3.connect(config.DB_PATH)
        conn.executescript(connection._SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.executescript("""
            CREATE VIRTUAL TABLE image_search_fts USING fts5(file_id UNINDEXED, search_text);
            CREATE VIRTUAL TABLE image_text_fts USING fts5(file_id UNINDEXED, partial_text, tokenize = 'trigram');
            CREATE TRIGGER trg_image_data_delete_fts AFTER DELETE ON image_data
            BEGIN DELETE FROM image_search_fts WHERE file_id = OLD.file_id; END;
            INSERT INTO scan_roots(path, added_at) VALUES ('/pictures', 'now');
            INSERT INTO image_data(root_id, file_path, hash, vlm_metadata_json) VALUES (1, '/pictures/a.jpg', 'h', '{}');
            PRAGMA user_version = 0;
        """)
        conn.close()

    def test_old_database_is_upgraded_and_keeps_its_data(self):
        self._old_database()
        conn = self.connect()
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
        self.assertFalse({"image_search_fts", "image_text_fts", "trg_image_data_delete_fts"} & _tables(conn))
        self.assertEqual(conn.execute("SELECT file_path FROM image_data").fetchone()[0], "/pictures/a.jpg")

    def test_a_failing_migration_changes_nothing(self):
        self._old_database()
        broken = {2: ["DROP TABLE image_search_fts", "THIS IS NOT SQL"]}
        with mock.patch.dict(connection._MIGRATIONS, broken):
            with self.assertRaises(DatabaseProblem):
                init_db(str(config.DB_PATH))
        raw = sqlite3.connect(config.DB_PATH)
        self.addCleanup(raw.close)
        self.assertEqual(raw.execute("PRAGMA user_version").fetchone()[0], 0)
        self.assertIn("image_search_fts", _tables(raw))  # the DROP before the failing step was rolled back

    def test_a_newer_database_is_refused(self):
        self.home.mkdir(parents=True)
        raw = sqlite3.connect(config.DB_PATH)
        raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
        raw.close()
        with self.assertRaisesRegex(DatabaseProblem, "newer version"):
            init_db(str(config.DB_PATH))

    def test_a_damaged_file_gives_a_plain_message(self):
        self.home.mkdir(parents=True)
        config.DB_PATH.write_bytes(b"this is not a database" * 100)
        with self.assertRaisesRegex(DatabaseProblem, "damaged"):
            init_db(str(config.DB_PATH))


class DescribeErrorTests(unittest.TestCase):
    def test_common_failures_are_explained(self):
        cases = {
            "database is locked": "busy",
            "database or disk is full": "disk is full",
            "attempt to write a readonly database": "read-only",
            "unable to open database file": "cannot open",
            "disk I/O error": "input/output",
            "database disk image is malformed": "damaged",
        }
        for raw, expected in cases.items():
            self.assertIn(expected, describe_db_error(sqlite3.OperationalError(raw)), raw)

    def test_unknown_errors_still_name_the_problem(self):
        self.assertIn("something odd", describe_db_error(sqlite3.OperationalError("something odd")))


if __name__ == "__main__":
    unittest.main()
