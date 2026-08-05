#!/usr/bin/env python3
"""
clear_db.py — Permanently wipe all Imagent data from the database.

Deletes every row from image_data, backlog, text_embeddings, and error_logs
in ~/.imagent/imagent.db. This does NOT delete any image files on disk —
only the indexed metadata, embeddings, and logs. Re-running scan.py/analyze_and_index.py
afterwards will re-index everything from scratch.
"""

import sys

import config
from db.connection import init_db

CONFIRMATION_PHRASE = "YES"


def clear_all(conn) -> dict[str, int]:
    """Delete every row from all Imagent tables and return per-table row counts."""
    counts: dict[str, int] = {}
    with conn:
        for table in ("image_data", "backlog", "text_embeddings", "error_logs"):
            cursor = conn.execute(f"DELETE FROM {table}")
            counts[table] = cursor.rowcount
    return counts


def main() -> None:
    print("=" * 60)
    print("  WARNING: THIS WILL PERMANENTLY DELETE ALL IMAGENT DATA")
    print("=" * 60)
    print(f"  Database: {config.DB_PATH}")
    print("  This clears image_data, backlog, text_embeddings, and")
    print("  error_logs. Original image files on disk are NOT touched,")
    print("  but the entire index (descriptions, embeddings, history)")
    print("  will be lost and cannot be undone.")
    print("=" * 60)

    answer = input(f"\nType '{CONFIRMATION_PHRASE}' to confirm: ").strip()
    if answer != CONFIRMATION_PHRASE:
        print("\n[imagent] Confirmation not received. Aborting — no data was deleted.")
        sys.exit(1)

    conn = init_db(str(config.DB_PATH))
    try:
        counts = clear_all(conn)
    finally:
        conn.close()

    print("\n[imagent] Database cleared.")
    for table, count in counts.items():
        print(f"  {table:<16}: {count} row(s) deleted")


if __name__ == "__main__":
    main()
