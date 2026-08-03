"""
db/operations.py — Low-level SQL helpers for image_data, text_embeddings, and error_logs.

All functions accept an open ``sqlite3.Connection`` and operate on a single row.
Transaction management (BEGIN / COMMIT / ROLLBACK) is the caller's responsibility.
"""

import sqlite3
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# image_data
# ---------------------------------------------------------------------------

def lookup_by_path(conn: sqlite3.Connection, path: str) -> dict | None:
    """Return the stored record for a given file path, or None if not found.

    Used at the start of each indexing cycle to decide whether to skip,
    update, or freshly index an image.
    """
    row = conn.execute(
        "SELECT image_id, hash, modified_at FROM image_data WHERE path = ?",
        (path,),
    ).fetchone()
    return dict(row) if row else None


def upsert_image_data(conn: sqlite3.Connection, record: dict) -> None:
    """Insert a new image_data row or update every column of an existing one.

    ``record`` must contain all column keys expected by the table.  On
    conflict with ``image_id`` the row is updated in-place; the UNIQUE
    constraint on ``path`` is handled by the ON CONFLICT clause.
    """
    conn.execute(
        """
        INSERT INTO image_data (
            image_id, path, hash,
            short_description, description, vlm_metadata_json,
            width, height, file_size, mime_type,
            created_at, modified_at, indexed_at,
            processing_time_ms, metadata_version, vlm_model
        ) VALUES (
            :image_id, :path, :hash,
            :short_description, :description, :vlm_metadata_json,
            :width, :height, :file_size, :mime_type,
            :created_at, :modified_at, :indexed_at,
            :processing_time_ms, :metadata_version, :vlm_model
        )
        ON CONFLICT(image_id) DO UPDATE SET
            path                = excluded.path,
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


def update_image_path(conn: sqlite3.Connection, image_id: str, new_path: str) -> None:
    """Update the stored file path after an image has been moved to indexed/.

    Called as a best-effort step after a successful file move.
    """
    conn.execute(
        "UPDATE image_data SET path = ? WHERE image_id = ?",
        (new_path, image_id),
    )


# ---------------------------------------------------------------------------
# text_embeddings
# ---------------------------------------------------------------------------

def upsert_text_embedding(
    conn: sqlite3.Connection,
    image_id: str,
    embedding_bytes: bytes,
    model: str,
) -> None:
    """Insert or update the serialised embedding for an image.

    ``embedding_bytes`` should be a packed array of IEEE 754 floats
    (see ``pipeline.serialisation.serialize_embedding``).
    """
    conn.execute(
        """
        INSERT INTO text_embeddings (image_id, embedding, model, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(image_id) DO UPDATE SET
            embedding  = excluded.embedding,
            model      = excluded.model,
            created_at = excluded.created_at
        """,
        (image_id, embedding_bytes, model, datetime.now(timezone.utc).isoformat()),
    )
