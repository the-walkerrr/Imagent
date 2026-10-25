"""
db/operations.py — Low-level SQL helpers for image_data, backlog, text_embeddings, and error_logs.

All functions accept an open ``sqlite3.Connection`` and operate on a single row.
Transaction management (BEGIN / COMMIT / ROLLBACK) is the caller's responsibility.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# image_data
# ---------------------------------------------------------------------------

def lookup_by_file_id(conn: sqlite3.Connection, file_id: str) -> dict | None:
    """Return the stored record for a given OS file ID, or None if not found.

    Used at the start of each indexing cycle to decide whether to skip,
    update, or freshly index an image.
    """
    row = conn.execute(
        "SELECT file_id, file_path, hash, modified_at FROM image_data WHERE file_id = ?",
        (file_id,),
    ).fetchone()
    return dict(row) if row else None


def upsert_image_data(conn: sqlite3.Connection, record: dict) -> None:
    """Insert a new image_data row or update every column of an existing one.

    ``record`` must contain all column keys expected by the table.  On
    conflict with ``file_id`` the row is updated in-place; the UNIQUE
    constraint on ``file_path`` is handled by the ON CONFLICT clause.
    """
    conn.execute(
        """
        INSERT INTO image_data (
            file_id, file_path, hash,
            short_description, description, vlm_metadata_json,
            width, height, file_size, mime_type,
            created_at, modified_at, indexed_at,
            processing_time_ms, metadata_version, vlm_model,
            taken_at, camera_make, camera_model, gps_lat, gps_lon
        ) VALUES (
            :file_id, :file_path, :hash,
            :short_description, :description, :vlm_metadata_json,
            :width, :height, :file_size, :mime_type,
            :created_at, :modified_at, :indexed_at,
            :processing_time_ms, :metadata_version, :vlm_model,
            :taken_at, :camera_make, :camera_model, :gps_lat, :gps_lon
        )
        ON CONFLICT(file_id) DO UPDATE SET
            file_path           = excluded.file_path,
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


def update_image_file_path(conn: sqlite3.Connection, file_id: str, file_path: str) -> None:
    """Update the stored file path for a file with a stable OS file ID."""
    conn.execute(
        "UPDATE image_data SET file_path = ? WHERE file_id = ?",
        (file_path, file_id),
    )


def update_image_modified_at(conn: sqlite3.Connection, file_id: str, modified_at: str) -> None:
    """Update the stored modification timestamp for a file with a stable OS file ID."""
    conn.execute(
        "UPDATE image_data SET modified_at = ? WHERE file_id = ?",
        (modified_at, file_id),
    )


# ---------------------------------------------------------------------------
# backlog
# ---------------------------------------------------------------------------

def upsert_backlog(conn: sqlite3.Connection, file_id: str, file_path: str) -> None:
    """Insert or update a pending backlog row."""
    conn.execute(
        """
        INSERT INTO backlog (file_id, file_path)
        VALUES (?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            file_path = excluded.file_path
        """,
        (file_id, file_path),
    )


def delete_image_data(conn: sqlite3.Connection, file_id: str) -> None:
    """Delete an image_data row by file_id."""
    conn.execute("DELETE FROM image_data WHERE file_id = ?", (file_id,))


def get_backlog(conn: sqlite3.Connection) -> list[dict]:
    """Return every pending backlog row as a dict with file_id and file_path."""
    rows = conn.execute("SELECT file_id, file_path FROM backlog ORDER BY file_id").fetchall()
    return [dict(row) for row in rows]


def delete_backlog(conn: sqlite3.Connection, file_id: str) -> None:
    """Remove a backlog row once it has been processed (or no longer needs indexing)."""
    conn.execute("DELETE FROM backlog WHERE file_id = ?", (file_id,))


def delete_image_data_not_in(conn: sqlite3.Connection, file_ids: set[str]) -> int:
    """Delete image_data rows whose file_id was not seen during a filesystem scan."""
    if not file_ids:
        cursor = conn.execute("DELETE FROM image_data")
        return cursor.rowcount

    placeholders = ", ".join("?" for _ in file_ids)
    cursor = conn.execute(
        f"DELETE FROM image_data WHERE file_id NOT IN ({placeholders})",
        tuple(file_ids),
    )
    return cursor.rowcount


# ---------------------------------------------------------------------------
# text_embeddings
# ---------------------------------------------------------------------------

def upsert_text_embedding(
    conn: sqlite3.Connection,
    file_id: str,
    embedding_bytes: bytes,
    model: str,
) -> None:
    """Insert or update the serialised embedding for an image.

    ``embedding_bytes`` should be a packed array of IEEE 754 floats
    (see ``pipeline.serialisation.serialize_embedding``).
    """
    conn.execute(
        """
        INSERT INTO text_embeddings (file_id, embedding, model, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            embedding  = excluded.embedding,
            model      = excluded.model,
            created_at = excluded.created_at
        """,
        (file_id, embedding_bytes, model, datetime.now(timezone.utc).isoformat()),
    )


def get_all_embeddings(conn: sqlite3.Connection, model: str) -> list[dict]:
    """Return every stored embedding whose model matches ``model``.

    Used by search to compare only embeddings generated with the currently
    configured embedding model — comparing vectors from different embedding
    spaces produces meaningless similarity scores.
    """
    rows = conn.execute(
        "SELECT file_id, embedding FROM text_embeddings WHERE model = ?",
        (model,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_embeddings_model_counts(conn: sqlite3.Connection) -> list[dict]:
    """Return the count of stored embeddings grouped by model name.

    Used by search to detect and report images indexed under a different
    embedding model than the one currently configured.
    """
    rows = conn.execute(
        "SELECT model, COUNT(*) AS n FROM text_embeddings GROUP BY model"
    ).fetchall()
    return [dict(row) for row in rows]


def get_image_summary(conn: sqlite3.Connection, file_id: str) -> dict | None:
    """Return display/reconstruction fields for a single image.

    Used by search to present results (file_path, short_description) and to
    rebuild an image's search text from stored VLM metadata when backfilling
    image_search_fts.
    """
    row = conn.execute(
        """
        SELECT file_id, file_path, short_description, description, vlm_metadata_json
        FROM image_data WHERE file_id = ?
        """,
        (file_id,),
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# image_search_fts
# ---------------------------------------------------------------------------

def upsert_fts_entry(conn: sqlite3.Connection, file_id: str, search_text: str) -> None:
    """Insert or replace the full-text search entry for an image.

    FTS5 virtual tables don't support ON CONFLICT upserts, so an existing
    entry is deleted before the new one is inserted.
    """
    conn.execute("DELETE FROM image_search_fts WHERE file_id = ?", (file_id,))
    conn.execute(
        "INSERT INTO image_search_fts (file_id, search_text) VALUES (?, ?)",
        (file_id, search_text),
    )


def upsert_partial_fts_entry(conn: sqlite3.Connection, file_id: str, partial_text: str) -> None:
    """Insert or replace the trigram (partial-word) entry for an image."""
    conn.execute("DELETE FROM image_text_fts WHERE file_id = ?", (file_id,))
    conn.execute(
        "INSERT INTO image_text_fts (file_id, partial_text) VALUES (?, ?)",
        (file_id, partial_text),
    )


def partial_fts_search(conn: sqlite3.Connection, query: str, limit: int) -> list[dict]:
    """Return up to ``limit`` file_ids whose visible text contains a fragment of ``query``.

    Ranked by SQLite's bm25 relevance (more negative = more relevant).
    """
    rows = conn.execute(
        """
        SELECT file_id, bm25(image_text_fts) AS rank
        FROM image_text_fts
        WHERE image_text_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def fts_search(conn: sqlite3.Connection, query: str, limit: int) -> list[dict]:
    """Return up to ``limit`` file_ids matching ``query`` via full-text search.

    Ranked by SQLite's bm25 relevance (more negative = more relevant).
    """
    rows = conn.execute(
        """
        SELECT file_id, bm25(image_search_fts) AS rank
        FROM image_search_fts
        WHERE image_search_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
    return [dict(row) for row in rows]
