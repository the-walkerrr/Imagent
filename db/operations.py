"""
db/operations.py — Low-level SQL helpers for model_settings, scan_roots, image_data, backlog,
text_embeddings and error_logs.

All functions accept an open ``sqlite3.Connection`` and operate on a single row
(or, where named so, on a list of rows).
Transaction management (BEGIN / COMMIT / ROLLBACK) is the caller's responsibility.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

# Rows deleted per statement. SQLite limits the number of ``?`` in one statement
# (999 before version 3.32), so long lists are processed in chunks below that.
_CHUNK = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# model_settings
# ---------------------------------------------------------------------------

def get_model_settings(conn: sqlite3.Connection) -> dict | None:
    """Return the locked model pair, or None when setup has not been completed."""
    row = conn.execute(
        """
        SELECT vlm_model, embedding_model, judge_model,
               vlm_digest, embedding_digest, judge_digest, embedding_dim
        FROM model_settings WHERE id = 1
        """
    ).fetchone()
    return dict(row) if row else None


def set_model_settings(conn: sqlite3.Connection, settings: dict) -> None:
    """Lock in the models. A plain INSERT: raises ``sqlite3.IntegrityError`` if they are already set."""
    conn.execute(
        """
        INSERT INTO model_settings
            (id, vlm_model, embedding_model, judge_model,
             vlm_digest, embedding_digest, judge_digest, embedding_dim, locked_at)
        VALUES (1, :vlm_model, :embedding_model, :judge_model,
                :vlm_digest, :embedding_digest, :judge_digest, :embedding_dim, :locked_at)
        """,
        {**settings, "locked_at": _now()},
    )


# ---------------------------------------------------------------------------
# scan_roots
# ---------------------------------------------------------------------------

def get_or_add_root(conn: sqlite3.Connection, path: str) -> int:
    """Return the root_id of a scanned folder, adding the folder if it is new."""
    conn.execute("INSERT OR IGNORE INTO scan_roots (path, added_at) VALUES (?, ?)", (path, _now()))
    return conn.execute("SELECT root_id FROM scan_roots WHERE path = ?", (path,)).fetchone()["root_id"]


def list_roots(conn: sqlite3.Connection) -> list[dict]:
    """Return every scanned folder, the most recently scanned first."""
    rows = conn.execute(
        "SELECT root_id, path FROM scan_roots ORDER BY COALESCE(last_scanned_at, added_at) DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def mark_root_scanned(conn: sqlite3.Connection, root_id: int) -> None:
    conn.execute("UPDATE scan_roots SET last_scanned_at = ? WHERE root_id = ?", (_now(), root_id))


# ---------------------------------------------------------------------------
# image_data
# ---------------------------------------------------------------------------

def lookup_by_path(conn: sqlite3.Connection, file_path: str) -> dict | None:
    """Return the stored record for a path, or None if the path is not indexed.

    Used at the start of each indexing cycle to decide whether to skip,
    update, or freshly index an image.
    """
    row = conn.execute(
        "SELECT file_id, file_path, hash, modified_at FROM image_data WHERE file_path = ?",
        (file_path,),
    ).fetchone()
    return dict(row) if row else None


def list_images_for_root(conn: sqlite3.Connection, root_id: int) -> list[dict]:
    """Return file_id, file_path, hash and modified_at of every image stored under a root."""
    rows = conn.execute(
        "SELECT file_id, file_path, hash, modified_at FROM image_data WHERE root_id = ?",
        (root_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def upsert_image_data(conn: sqlite3.Connection, record: dict) -> int:
    """Insert a new image_data row or update every column of the existing one for that path.

    ``record`` must contain all column keys expected by the table except ``file_id``.
    Returns the row's file_id (unchanged when the row already existed).
    """
    conn.execute(
        """
        INSERT INTO image_data (
            root_id, file_path, hash,
            short_description, description, vlm_metadata_json,
            width, height, file_size, mime_type,
            created_at, modified_at, indexed_at,
            processing_time_ms, metadata_version, vlm_model,
            taken_at, camera_make, camera_model, gps_lat, gps_lon
        ) VALUES (
            :root_id, :file_path, :hash,
            :short_description, :description, :vlm_metadata_json,
            :width, :height, :file_size, :mime_type,
            :created_at, :modified_at, :indexed_at,
            :processing_time_ms, :metadata_version, :vlm_model,
            :taken_at, :camera_make, :camera_model, :gps_lat, :gps_lon
        )
        ON CONFLICT(file_path) DO UPDATE SET
            root_id             = excluded.root_id,
            hash                = excluded.hash,
            short_description   = excluded.short_description,
            description         = excluded.description,
            vlm_metadata_json   = excluded.vlm_metadata_json,
            width               = excluded.width,
            height              = excluded.height,
            file_size           = excluded.file_size,
            mime_type           = excluded.mime_type,
            created_at          = excluded.created_at,
            modified_at         = excluded.modified_at,
            indexed_at          = excluded.indexed_at,
            processing_time_ms  = excluded.processing_time_ms,
            metadata_version    = excluded.metadata_version,
            vlm_model           = excluded.vlm_model,
            taken_at            = excluded.taken_at,
            camera_make         = excluded.camera_make,
            camera_model        = excluded.camera_model,
            gps_lat             = excluded.gps_lat,
            gps_lon             = excluded.gps_lon
        """,
        record,
    )
    return conn.execute(
        "SELECT file_id FROM image_data WHERE file_path = ?", (record["file_path"],)
    ).fetchone()["file_id"]


def update_image_location(conn: sqlite3.Connection, file_id: int, file_path: str, modified_at: str) -> None:
    """Point an already-indexed image at its new path (a move or rename of identical content)."""
    conn.execute(
        "UPDATE image_data SET file_path = ?, modified_at = ? WHERE file_id = ?",
        (file_path, modified_at, file_id),
    )


def update_image_modified_at(conn: sqlite3.Connection, file_id: int, modified_at: str) -> None:
    """Update the stored modification timestamp of an image whose content did not change."""
    conn.execute(
        "UPDATE image_data SET modified_at = ? WHERE file_id = ?",
        (modified_at, file_id),
    )


def delete_images(conn: sqlite3.Connection, file_ids: list[int]) -> int:
    """Delete image_data rows by file_id (embeddings and error logs follow); returns the count."""
    deleted = 0
    for start in range(0, len(file_ids), _CHUNK):
        chunk = file_ids[start:start + _CHUNK]
        placeholders = ", ".join("?" for _ in chunk)
        deleted += conn.execute(f"DELETE FROM image_data WHERE file_id IN ({placeholders})", chunk).rowcount
    return deleted


# ---------------------------------------------------------------------------
# backlog
# ---------------------------------------------------------------------------

def upsert_backlog(conn: sqlite3.Connection, file_path: str, root_id: int, modified_at: str) -> None:
    """Queue a file for indexing. If the file changed since it was last queued, its failed attempts are forgotten."""
    conn.execute(
        """
        INSERT INTO backlog (file_path, root_id, modified_at, queued_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            root_id     = excluded.root_id,
            attempts    = CASE WHEN backlog.modified_at != excluded.modified_at THEN 0 ELSE backlog.attempts END,
            last_error  = CASE WHEN backlog.modified_at != excluded.modified_at THEN NULL ELSE backlog.last_error END,
            modified_at = excluded.modified_at
        """,
        (file_path, root_id, modified_at, _now()),
    )


def get_backlog(conn: sqlite3.Connection, max_attempts: int) -> list[dict]:
    """Return the pending backlog rows that have failed fewer than ``max_attempts`` times."""
    rows = conn.execute(
        "SELECT file_path, root_id FROM backlog WHERE attempts < ? ORDER BY file_path",
        (max_attempts,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_exhausted_backlog(conn: sqlite3.Connection, max_attempts: int) -> list[dict]:
    """Return the backlog rows that were given up on, with their last error."""
    rows = conn.execute(
        "SELECT file_path, last_error FROM backlog WHERE attempts >= ? ORDER BY file_path",
        (max_attempts,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_backlog_paths_for_root(conn: sqlite3.Connection, root_id: int) -> list[str]:
    rows = conn.execute("SELECT file_path FROM backlog WHERE root_id = ?", (root_id,)).fetchall()
    return [row["file_path"] for row in rows]


def delete_backlog(conn: sqlite3.Connection, file_path: str) -> None:
    """Remove a backlog row once it has been processed (or no longer needs indexing)."""
    conn.execute("DELETE FROM backlog WHERE file_path = ?", (file_path,))


def delete_backlog_paths(conn: sqlite3.Connection, file_paths: list[str]) -> None:
    for start in range(0, len(file_paths), _CHUNK):
        chunk = file_paths[start:start + _CHUNK]
        placeholders = ", ".join("?" for _ in chunk)
        conn.execute(f"DELETE FROM backlog WHERE file_path IN ({placeholders})", chunk)


def record_backlog_failure(conn: sqlite3.Connection, file_path: str, message: str, count_attempt: bool) -> None:
    """Remember why indexing a file failed; ``count_attempt`` is False when the file is not to blame."""
    conn.execute(
        "UPDATE backlog SET attempts = attempts + ?, last_error = ? WHERE file_path = ?",
        (1 if count_attempt else 0, message, file_path),
    )


# ---------------------------------------------------------------------------
# text_embeddings
# ---------------------------------------------------------------------------

def upsert_text_embedding(conn: sqlite3.Connection, file_id: int, embedding_bytes: bytes) -> None:
    """Insert or update the serialised embedding for an image.

    ``embedding_bytes`` should be a packed array of little-endian float32
    (see ``pipeline.serialisation.serialize_embedding``).
    """
    conn.execute(
        """
        INSERT INTO text_embeddings (file_id, embedding, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            embedding  = excluded.embedding,
            created_at = excluded.created_at
        """,
        (file_id, embedding_bytes, _now()),
    )


def count_embeddings(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM text_embeddings").fetchone()["n"]


def iter_embeddings(conn: sqlite3.Connection):
    """Yield every stored (file_id, embedding blob) row without loading them all at once."""
    return conn.execute("SELECT file_id, embedding FROM text_embeddings")


def get_image_summary(conn: sqlite3.Connection, file_id: int) -> dict | None:
    """Return display/reconstruction fields for a single image.

    Used by search to present results (file_path, short_description).
    """
    row = conn.execute(
        """
        SELECT file_id, file_path, short_description, description, vlm_metadata_json
        FROM image_data WHERE file_id = ?
        """,
        (file_id,),
    ).fetchone()
    return dict(row) if row else None
