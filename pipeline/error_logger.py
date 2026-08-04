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


def log_error(
    conn: sqlite3.Connection,
    *,
    stage: str,
    exc: Exception,
    file_id: str | None = None,
    file_path: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Insert a structured error record into ``error_logs``.

    Args:
        conn:     Open database connection.  The insert is committed immediately
                  using its own transaction so it is persisted even if the caller
                  later rolls back a broader transaction.
        stage:    Processing stage where the error occurred.
                  Valid values: ``file_scan``, ``metadata_extraction``,
                  ``embedding_generation``, ``database_write``.
        exc:      The caught exception instance.
        file_id:  The OS file ID if it was known when the error occurred.
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
        conn.commit()
    except Exception:
        # Swallow any failure here — logging must never crash the pipeline.
        pass
