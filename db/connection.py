"""
db/connection.py — Database initialisation and connection factory.

Opens (or creates) the SQLite database, applies the schema from schema.sql,
and configures per-connection PRAGMAs required by the pipeline.
"""

import sqlite3
from pathlib import Path

# Locate schema.sql relative to this file (two directories up = project root)
_SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"


def init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database and apply the schema.

    Args:
        db_path: Absolute or relative path to the .db file.

    Returns:
        An open ``sqlite3.Connection`` with ``row_factory`` set to
        ``sqlite3.Row`` so columns are accessible by name.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Apply the full schema (CREATE TABLE IF NOT EXISTS — idempotent).
    # executescript issues an implicit COMMIT first, which is fine here.
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)

    # Re-apply PRAGMAs after executescript, which opens a fresh context.
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL mode allows concurrent reads during writes — better for long pipeline runs.
    conn.execute("PRAGMA journal_mode = WAL")

    return conn
