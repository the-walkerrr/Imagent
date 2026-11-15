"""
pipeline/error_logger.py — Insert structured error records into the ``error_logs`` table.

Designed to be called inside ``except`` blocks throughout the pipeline.  This
function is intentionally silent: it never raises an exception of its own, because
error logging must not interrupt processing of remaining images.
"""

from __future__ import annotations

import json
import sqlite3
import traceback
from datetime import datetime, timezone

import config


def log_error(
    conn: sqlite3.Connection,
    *,
    stage: str,
    exc: Exception,
    file_id: int | None = None,
    file_path: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Insert a structured error record into ``error_logs``.

    Args:
        conn:     Open database connection.  The insert is committed immediately
                  using its own transaction so it is persisted even if the caller
                  later rolls back a broader transaction.
        stage:    Where the error occurred (e.g. ``pipeline``, ``search``,
                  ``query_understanding``, ``scan``).
        exc:      The caught exception instance.
        file_id:  The image_data file_id, only if the image already has a row (the column is a foreign key).
        file_path: The file path being processed at the time of the error.
        metadata: Optional dict of additional context (model names, timings, etc.).
                  Will be stored as JSON.
    """
    try:
        conn.execute(
            """
            INSERT INTO error_logs
                (file_id, file_path, stage, error_type, error_message, stack_trace, metadata, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                file_path,
                stage,
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
                json.dumps(metadata) if metadata else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        # Keep only the newest rows so repeated failures cannot grow the table without limit.
        conn.execute(
            "DELETE FROM error_logs WHERE id <= (SELECT MAX(id) FROM error_logs) - ?",
            (config.ERROR_LOG_MAX_ROWS,),
        )
        conn.commit()
    except Exception:
        # Swallow any failure here — logging must never crash the pipeline.
        pass
