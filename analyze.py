#!/usr/bin/env python3
"""
analyze.py — Imagent image indexing pipeline entrypoint.

Prompts the user for a root folder, discovers all supported images, and runs
each one through the full indexing pipeline:

  file_scan → metadata_extraction → embedding_generation → database_write

All activity is stored in ``<root>/imagent.db``.
"""

from __future__ import annotations

import array
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from db.connection import init_db
from db.operations import (
    lookup_by_file_id,
    update_image_file_path,
    upsert_image_data,
    upsert_text_embedding,
)
from image_processing.hashing import compute_hash
from image_processing.metadata import extract_file_metadata
from pipeline.discovery import discover_images
from pipeline.error_logger import log_error
from pipeline.validation import validate_folder
from text_processing.formatters import build_search_text
from vlm.analyzer import analyse_image, embed_search_text

# Name of the SQLite database file, created inside the root folder.
DB_FILENAME = "imagent.db"

# Increment this when the stored metadata schema changes incompatibly.
METADATA_VERSION = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def serialize_embedding(embedding: list[float]) -> bytes:
    """Pack a float list into raw bytes (IEEE 754 single precision).

    Using ``array`` from the stdlib keeps the stored size at 4 bytes per
    dimension and avoids adding a numpy dependency.
    """
    return array.array("f", embedding).tobytes()


def build_vlm_metadata_json(payload) -> str:
    """Serialise all VLM payload fields except ``short_description`` and ``description``.

    Those two fields are stored in dedicated columns on ``image_data``, so they
    are excluded here to avoid duplication.
    """
    data = payload.model_dump()
    data.pop("short_description", None)
    data.pop("description", None)
    return json.dumps(data)


def is_unchanged(existing: dict | None, file_hash: str, modified_at: str) -> bool:
    """Return True if the stored hash and modification time match the current file.

    When True the image can be safely skipped — neither content nor metadata
    has changed since the last indexing run.
    """
    if existing is None:
        return False
    return existing["hash"] == file_hash and existing["modified_at"] == modified_at


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def index_image(conn, image_path: Path) -> str:
    """Run the complete indexing pipeline for a single image.

    Stages (in order):
      1. file_scan           — hash + file metadata
      2. metadata_extraction — VLM analysis
      3. embedding_generation — build search doc + embed
      4. database_write      — atomic upsert of image_data + text_embeddings

    Returns:
        ``'skipped'``  — file is unchanged since the last run.
        ``'updated'``  — file existed but content or metadata changed.
        ``'indexed'``  — new image, not previously seen.
    """
    path_str = str(image_path)
    file_id: str | None = None

    # ------------------------------------------------------------------
    # Stage 1: file_scan — gather hash and file-level metadata
    # ------------------------------------------------------------------
    file_hash = compute_hash(path_str)
    file_meta = extract_file_metadata(path_str)
    file_id = file_meta["file_id"]
    modified_at: str = file_meta["modified_at"]

    existing = lookup_by_file_id(conn, file_id)

    if is_unchanged(existing, file_hash, modified_at):
        if existing["file_path"] != path_str:
            update_image_file_path(conn, file_id, path_str)
            conn.commit()
        return "skipped"

    is_update = existing is not None

    # ------------------------------------------------------------------
    # Stage 2: metadata_extraction — VLM analysis
    # ------------------------------------------------------------------
    pipeline_start = time.perf_counter()
    payload = analyse_image(path_str)

    # ------------------------------------------------------------------
    # Stage 3: embedding_generation — search document + embedding vector
    # ------------------------------------------------------------------
    search_text = build_search_text(payload)
    embedding = embed_search_text(search_text)
    embedding_bytes = serialize_embedding(embedding)
    elapsed_ms = int((time.perf_counter() - pipeline_start) * 1000)

    vlm_metadata_json = build_vlm_metadata_json(payload)

    # ------------------------------------------------------------------
    # Stage 4: database_write — atomic transaction
    # ------------------------------------------------------------------
    record = {
        "file_id": file_id,
        "file_path": path_str,
        "hash": file_hash,
        "short_description": payload.short_description,
        "description": payload.description,
        "vlm_metadata_json": vlm_metadata_json,
        **file_meta,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "processing_time_ms": elapsed_ms,
        "metadata_version": METADATA_VERSION,
        "vlm_model": config.OLLAMA_MODEL,
    }

    # ``with conn`` starts a transaction; commits on success, rolls back on error.
    with conn:
        upsert_image_data(conn, record)
        upsert_text_embedding(conn, file_id, embedding_bytes, config.OLLAMA_EMBEDDING_MODEL)
    # ↑ Commit point — image is now considered fully indexed in the database.

    return "updated" if is_update else "indexed"


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run(root: Path) -> None:
    """Initialise the database, discover images, and index them one by one.

    Progress is printed for each image.  Failures are logged to ``error_logs``
    and do not stop processing of the remaining images.  A summary is printed
    at the end.
    """
    db_path = str(root / DB_FILENAME)
    conn = init_db(db_path)

    print(f"\n[imagent] Root     : {root}")
    print(f"[imagent] Database : {db_path}")

    images = discover_images(root)
    total = len(images)
    print(f"[imagent] Discovered {total} image(s)\n")

    if total == 0:
        print("[imagent] Nothing to index.")
        conn.close()
        return

    counts: dict[str, int] = {"indexed": 0, "updated": 0, "skipped": 0, "failed": 0}
    wall_start = time.perf_counter()

    for i, image_path in enumerate(images, start=1):
        rel = image_path.relative_to(root)
        # Print progress inline; result appended on the same line after processing.
        print(f"  [{i:>{len(str(total))}}/{total}] {rel} ... ", end="", flush=True)

        try:
            result = index_image(conn, image_path)
            counts[result] += 1
            print(result)
        except Exception as exc:
            counts["failed"] += 1
            print(f"FAILED  ({type(exc).__name__}: {exc})")
            log_error(
                conn,
                stage="pipeline",
                exc=exc,
                file_path=str(image_path),
                metadata={"root": str(root)},
            )

    elapsed = time.perf_counter() - wall_start

    # Final summary
    print(f"\n{'─' * 52}")
    print(f"  Total discovered  : {total}")
    print(f"  Indexed (new)     : {counts['indexed']}")
    print(f"  Updated           : {counts['updated']}")
    print(f"  Skipped           : {counts['skipped']}")
    print(f"  Failed            : {counts['failed']}")
    print(f"  Total elapsed     : {elapsed:.2f}s")
    print(f"{'─' * 52}\n")

    conn.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Prompt the user for a folder path and kick off the indexing pipeline."""
    raw_path = input("Enter the folder path to index: ").strip()
    root = validate_folder(raw_path)
    run(root)


if __name__ == "__main__":
    main()
