"""
pipeline/error_logger.py — Insert structured error records into the ``error_logs`` table.

Designed to be called inside ``except`` blocks throughout the pipeline.  This
function is intentionally silent: it never raises an exception of its own, because
error logging must not interrupt processing of remaining images.
"""

import json
import sqlite3
import traceback
from datetime import datetime, timezone


def log_error(
    conn: sqlite3.Connection,
    *,
    stage: str,
    exc: Exception,
    image_id: str | None = None,
    path: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Insert a structured error record into ``error_logs``.

    Args:
        conn:     Open database connection.  The insert is committed immediately
                  using its own transaction so it is persisted even if the caller
                  later rolls back a broader transaction.
        stage:    Processing stage where the error occurred.
                  Valid values: ``file_scan``, ``metadata_extraction``,
                  ``embedding_generation``, ``database_write``, ``file_move``.
        exc:      The caught exception instance.
        image_id: The image's UUID if it was known when the error occurred.
        path:     The file path being processed at the time of the error.
        metadata: Optional dict of additional context (model names, timings, etc.).
                  Will be stored as JSON.
    """
    try:
        conn.execute(
            """
            INSERT INTO error_logs
                (image_id, path, stage, error_type, error_message, stack_trace, metadata, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                path,
                stage,
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
                json.dumps(metadata) if metadata else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception:
        # Swallow any failure here — logging must never crash the pipeline.
        pass
