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
            processing_time_ms, metadata_version, vlm_model
        ) VALUES (
            :file_id, :file_path, :hash,
            :short_description, :description, :vlm_metadata_json,
            :width, :height, :file_size, :mime_type,
            :created_at, :modified_at, :indexed_at,
            :processing_time_ms, :metadata_version, :vlm_model
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
            vlm_model           = excluded.vlm_model
        """,
        record,
    )


def update_image_file_path(conn: sqlite3.Connection, file_id: str, file_path: str) -> None:
    """Update the stored file path for a file with a stable OS file ID."""
    conn.execute(
        "UPDATE image_data SET file_path = ? WHERE file_id = ?",
        (file_path, file_id),
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
