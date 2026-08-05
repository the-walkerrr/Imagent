#!/usr/bin/env python3
"""Scan a folder and populate backlog rows for new or modified files."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
from analyze_and_index import run as run_indexing
from db.connection import init_db
from db.operations import (
    delete_image_data,
    delete_image_data_not_in,
    get_backlog,
    lookup_by_file_id,
    update_image_file_path,
    upsert_backlog,
)
from pipeline.discovery import discover_images
from pipeline.validation import validate_folder


def file_scan_metadata(file_path: Path) -> dict[str, str]:
    """Return the OS file ID, absolute path, and last modified timestamp."""
    stat = os.stat(file_path)
    return {
        "file_id": str(stat.st_ino),
        "file_path": str(file_path),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def scan(root: Path) -> dict[str, int]:
    """Scan root and synchronize image_data/backlog state."""
    conn = init_db(str(config.DB_PATH))
    counts = {
        "scanned": 0,
        "queued_new": 0,
        "queued_modified": 0,
        "paths_updated": 0,
        "unchanged": 0,
        "deleted": 0,
    }
    seen_file_ids: set[str] = set()

    try:
        for file_path in discover_images(root):
            metadata = file_scan_metadata(file_path)
            file_id = metadata["file_id"]
            current_file_path = metadata["file_path"]
            current_modified_at = metadata["modified_at"]
            seen_file_ids.add(file_id)
            counts["scanned"] += 1

            existing = lookup_by_file_id(conn, file_id)

            if existing:
                if existing["modified_at"] != current_modified_at:
                    with conn:
                        delete_image_data(conn, file_id)
                        upsert_backlog(conn, file_id, current_file_path)
                    counts["queued_modified"] += 1
                    continue

                if existing["file_path"] != current_file_path:
                    with conn:
                        update_image_file_path(conn, file_id, current_file_path)
                    counts["paths_updated"] += 1
                    continue

                counts["unchanged"] += 1
                continue

            with conn:
                upsert_backlog(conn, file_id, current_file_path)
            counts["queued_new"] += 1

        with conn:
            counts["deleted"] = delete_image_data_not_in(conn, seen_file_ids)

        return counts
    finally:
        conn.close()


def main() -> None:
    raw_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter the folder path to scan: ").strip()
    root = validate_folder(raw_path)
    counts = scan(root)

    print(f"\n[imagent] Root     : {root}")
    print(f"[imagent] Database : {config.DB_PATH}")
    print(f"[imagent] Scanned  : {counts['scanned']}")
    print(f"[imagent] New      : {counts['queued_new']}")
    print(f"[imagent] Modified : {counts['queued_modified']}")
    print(f"[imagent] Moved    : {counts['paths_updated']}")
    print(f"[imagent] Unchanged: {counts['unchanged']}")
    print(f"[imagent] Deleted  : {counts['deleted']}\n")

    conn = init_db(str(config.DB_PATH))
    try:
        backlog_size = len(get_backlog(conn))
    finally:
        conn.close()

    if backlog_size == 0:
        return

    answer = input(
        f"[imagent] {backlog_size} file(s) in the backlog are pending indexing. "
        "Run indexing now? [y/N]: "
    ).strip().lower()
    if answer in ("y", "yes"):
        run_indexing()
    else:
        print("[imagent] Skipping indexing. Run analyze_and_index.py later to process the backlog.")


if __name__ == "__main__":
    main()
