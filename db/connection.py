"""
db/connection.py — Database initialisation and connection factory.

Opens (or creates) the SQLite database, brings its schema up to date, and configures
per-connection PRAGMAs required by the pipeline.

The schema version is kept in ``PRAGMA user_version``. A new database gets ``schema.sql``
(always the latest schema); an older one is upgraded by running the steps of ``_MIGRATIONS``
in a single transaction, so a crash or a power cut leaves it exactly as it was.
"""

import os
import sqlite3
from pathlib import Path

import config

# schema.sql sits next to this file
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# The upserts (INSERT ... ON CONFLICT DO UPDATE) used throughout arrived in SQLite 3.24 (June 2018).
_MIN_SQLITE_VERSION = (3, 24, 0)

# The version of the schema in schema.sql. Raise it, and add the steps to _MIGRATIONS, with every schema change.
SCHEMA_VERSION = 2

# Version 1 is the schema of databases made before versions were recorded (user_version 0 with tables).
# _MIGRATIONS[n] holds the statements that turn version n - 1 into version n.
_MIGRATIONS: dict[int, list[str]] = {
    # 2: the full-text-search tables were never queried, so they are no longer kept.
    2: [
        "DROP TRIGGER IF EXISTS trg_image_data_delete_fts",
        "DROP TABLE IF EXISTS image_search_fts",
        "DROP TABLE IF EXISTS image_text_fts",
    ],
}


class DatabaseProblem(RuntimeError):
    """The database cannot be used; the message says why and what to do, in plain words."""


def describe_db_error(exc: Exception, path=None) -> str:
    """Explain an ``sqlite3.Error`` to the user, with what they can do about it (``path``: the database, default the usual one)."""
    text = str(exc).lower()
    path = path or config.DB_PATH
    if "locked" in text or "busy" in text:
        return "The Imagent database is busy because another Imagent program is writing to it. Wait a moment and try again."
    if "full" in text:
        return "The disk is full, so nothing more can be saved. Free up some space and run again; finished work is kept."
    if "readonly" in text or "read-only" in text:
        return f"The Imagent database is read-only: {path}. Check the permissions of that file and its folder."
    if "unable to open" in text:
        return f"Imagent cannot open its database at {path}. Check that the folder exists and that you can write to it."
    if "disk i/o" in text:
        return "The disk reported an input/output error. If the data lives on an external or network drive, check it is connected."
    if "not a database" in text or "malformed" in text or "corrupt" in text:
        return (f"The Imagent database is damaged: {path}. If you have a copy of it, restore the copy. "
                f"Otherwise rename or delete that file, run {config.INSTALL_COMMAND} again, and scan your folders again. "
                "Your image files are not affected.")
    return f"The Imagent database reported an error ({type(exc).__name__}: {exc})."


def _version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _statements(script: str) -> list[str]:
    """Split a SQL script into statements, so each can run inside one explicit transaction."""
    statements, buffer = [], ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statements.append(buffer.strip())
            buffer = ""
    return statements


def _upgrade(conn: sqlite3.Connection) -> None:
    """Create the schema in a new database, or bring an older one up to ``SCHEMA_VERSION``."""
    if _version(conn) == SCHEMA_VERSION:
        return
    # Python's own transaction handling is switched off while the schema changes, so the whole
    # change is one explicit transaction. IMMEDIATE takes the write lock first, so two programs
    # starting together cannot both migrate.
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = _version(conn)  # read again: another program may have finished the job meanwhile
            if version > SCHEMA_VERSION:
                raise DatabaseProblem(
                    f"This database was made by a newer version of Imagent (schema {version}; this one knows "
                    f"{SCHEMA_VERSION}). Update Imagent, or it could damage the database."
                )
            if version < SCHEMA_VERSION:
                has_tables = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'image_data'"
                ).fetchone()
                if version == 0 and not has_tables:
                    for statement in _statements(_SCHEMA_PATH.read_text(encoding="utf-8")):
                        conn.execute(statement)
                else:
                    for target in range(max(version, 1) + 1, SCHEMA_VERSION + 1):
                        for statement in _MIGRATIONS[target]:
                            conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.isolation_level = ""  # back to Python's default


def init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database and bring its schema up to date.

    Args:
        db_path: Absolute or relative path to the .db file.

    Returns:
        An open ``sqlite3.Connection`` with ``row_factory`` set to
        ``sqlite3.Row`` so columns are accessible by name.

    Raises:
        DatabaseProblem: the database cannot be used (damaged, locked, newer than this
            program, no space, SQLite too old); the message tells the user what to do.
    """
    if sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION:
        needed = ".".join(map(str, _MIN_SQLITE_VERSION))
        raise DatabaseProblem(
            f"SQLite {sqlite3.sqlite_version} is too old; Imagent needs {needed} or newer. "
            "Install a newer Python (python.org builds bundle a recent SQLite)."
        )

    data_dir = Path(db_path).parent
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            data_dir.chmod(0o700)  # the database holds file paths and GPS positions: this user only
    except OSError as exc:
        raise DatabaseProblem(f"Imagent cannot use its data folder {data_dir} ({exc.strerror or exc}). "
                              "Check that you can write to it.") from exc

    conn = None
    try:
        # `timeout` is how long a write waits for another process's lock before raising "database is locked".
        conn = sqlite3.connect(db_path, timeout=config.SQLITE_BUSY_TIMEOUT_S)
        conn.row_factory = sqlite3.Row
        # WAL mode allows concurrent reads during writes — better for long pipeline runs.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        _upgrade(conn)
    except sqlite3.Error as exc:
        if conn is not None:
            conn.close()
        raise DatabaseProblem(describe_db_error(exc, db_path)) from exc
    except BaseException:
        if conn is not None:
            conn.close()
        raise

    if os.name == "posix":
        Path(db_path).chmod(0o600)

    return conn
