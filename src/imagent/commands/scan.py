#!/usr/bin/env python3
"""
imagent/commands/scan.py — Imagent: scan folders for images, then index them.

Scanning compares a folder with the index and queues the new images and those whose picture changed. A file
whose EXIF facts changed keeps its analysis and only has them refreshed, an image that moved or was renamed keeps
its record, and one that disappeared is forgotten. Indexing then takes every file queued
in the ``backlog`` table, including any left over from an interrupted run, and runs each one through the
full indexing pipeline:

  file_scan → metadata_extraction → embedding_generation → database_write

Successfully processed files (indexed/updated/skipped) are removed from the backlog; a file that fails stays
queued and is retried on the next run, up to ``config.BACKLOG_MAX_ATTEMPTS`` times (until it changes).

All activity is stored in the user-level database (``config.DB_PATH``).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import textwrap
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx
import ollama
from PIL import Image

from imagent import config
from imagent.db.connection import describe_db_error, init_db
from imagent.db.operations import (
    REUSED_COLUMNS,
    VERSION_COLUMNS,
    StoredImage,
    add_failure_history,
    delete_backlog,
    delete_backlog_paths,
    delete_images,
    find_analysis_by_pixels,
    get_backlog,
    get_backlog_paths_for_root,
    get_exhausted_backlog,
    get_exif_fields,
    get_keep_deleted,
    get_or_add_root,
    images_by_path,
    images_outside_root,
    keep_analyses_of,
    list_roots,
    lookup_by_path,
    mark_root_scanned,
    park_backlog,
    record_backlog_failure,
    refresh_image_file,
    refund_backlog_attempts,
    requeue_changed_parked,
    update_image_location,
    update_image_modified_at,
    upsert_backlog,
    upsert_image_data,
    upsert_text_embedding,
    use_deleted_analysis,
)
from imagent.image_processing import ocr
from imagent.image_processing.hashing import compute_hash, compute_pixel_hash
from imagent.image_processing.memory import check_image_size
from imagent.image_processing.metadata import extract_file_metadata, iso_utc
from imagent.image_processing.problems import BUSY, NOT_DOWNLOADED, UNREADABLE, UNSUPPORTED, ImageProblem, explain, problem
from imagent.os_specific.files import is_not_downloaded
from imagent.pipeline.console import file_link, make_output_safe
from imagent.pipeline.discovery import discover
from imagent.pipeline.error_messages import describe
from imagent.pipeline.error_logger import log_error
from imagent.pipeline.failure import IndexingFailure, RunStopped
from imagent.pipeline.folder_picker import ask_for_folder
from imagent.pipeline.locking import AlreadyRunning, single_instance
from imagent.pipeline.progress import Progress, format_duration
from imagent.pipeline.serialisation import serialize_embedding
from imagent.pipeline.versions import collect_versions, model_versions
from imagent.pipeline.validation import find_overlapping_root, validate_folder
from imagent.text_processing.formatters import build_description, build_search_text
from imagent.vlm.analyzer import OUT_OF_MEMORY, SERVER_CRASH, TIMEOUT, analyse_image, embed_search_text
from imagent.vlm.client import NotReady, check_ready, load_model, make_client, unload_model

try:
    import questionary

    STYLE = questionary.Style(
        [
            ("qmark", "fg:ansidefault bold"),
            ("question", "fg:ansidefault bold"),
            ("pointer", "fg:ansidefault bold"),
            ("highlighted", "underline"),
            ("selected", "underline"),
            ("answer", "fg:ansidefault bold"),
            ("instruction", "fg:ansibrightblack"),
        ]
    )
except ImportError:
    questionary = None
    STYLE = None

# Increment this when the stored metadata schema changes incompatibly.
METADATA_VERSION = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_vlm_metadata_json(analysis) -> str:
    """Serialise everything learned about the image: the model's reply exactly as it wrote it, the text read in the image
    and the facts read from the pixels. The summary and the description are also kept in their own columns on ``image_data``."""
    return json.dumps({"vlm": analysis.reply, "ocr_text": analysis.ocr_text, "features": analysis.features})


def is_unchanged(existing: dict | None, file_hash: str, modified_at: str) -> bool:
    """Return True if the stored hash and modification time match the current file.

    When True the image can be safely skipped — neither content nor metadata
    has changed since the last indexing run.
    """
    if existing is None:
        return False
    return existing["hash"] == file_hash and existing["modified_at"] == modified_at


def _changed_since_scan(image_path: Path, scanned_modified_at: str | None) -> bool:
    """True if the file's modification time is not the one the scan saw: it changed after the scan, so it may still be being written.

    A damaged-looking file that is still changing is not damaged yet; it is tried again later, and nothing is held against it."""
    if not scanned_modified_at:
        return False
    try:
        return iso_utc(os.stat(image_path).st_mtime) != scanned_modified_at
    except OSError:
        return False


def classify_failure(exc: Exception) -> tuple[bool, bool]:
    """Return ``(counts_as_attempt, ollama_problem)`` for an exception raised while indexing one image.

    Not being able to reach Ollama, or a model that is no longer installed, is not the file's fault,
    so it does not use up one of the file's attempts. Other trouble talking to Ollama (an error reply,
    a timeout, a connection cut off in the middle, as when Ollama crashes) does count, since a file can
    be what makes the model fail, but it also counts towards stopping the run when Ollama keeps failing.
    """
    if isinstance(exc, ollama.ResponseError) and exc.status_code == 404:
        return False, True
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError)):
        return False, True
    if isinstance(exc, (httpx.TransportError, ollama.ResponseError)):
        return True, True
    return True, False


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def index_image(conn, client: ollama.Client, settings: dict, root_id: int, image_path: Path,
                on_stage: Callable[[str], None] = lambda stage: None,
                on_retry: Callable[[str], None] = lambda kind: None) -> tuple[str, str]:
    """Run the complete indexing pipeline for a single image.

    Stages (in order):
      1. file_scan           — hash + file metadata
      2. metadata_extraction — VLM analysis
      3. embedding_generation — build search doc + embed
      4. database_write      — atomic upsert of image_data + text_embeddings

    ``on_stage`` is told what is being done, before each step; ``on_retry`` is told when the model's reply
    has to be asked for again, and why.

    Raises ``AnalysisFailed`` when the model's reply could not be put right.

    Returns ``(result, short_description)``, where the description is empty for a skipped file and
    ``result`` is one of:
        ``'skipped'``  — file is unchanged since the last run.
        ``'updated'``  — file existed but content or metadata changed.
        ``'indexed'``  — new image, not previously seen.
    """
    path_str = str(image_path)
    versions = settings.get("versions", {})
    on_stage("checking file")
    if is_not_downloaded(os.stat(path_str)):  # evicted to cloud storage since the scan: reading it would download it
        raise problem(NOT_DOWNLOADED, "not_downloaded",
                      step="checked the file", park=False)

    # ------------------------------------------------------------------
    # Stage 1: file_scan — gather hash and file-level metadata
    # ------------------------------------------------------------------
    file_hash = compute_hash(path_str)
    file_meta = extract_file_metadata(path_str)
    modified_at: str = file_meta["modified_at"]

    existing = lookup_by_path(conn, path_str)

    if is_unchanged(existing, file_hash, modified_at):
        return "skipped", ""

    is_update = existing is not None

    # ------------------------------------------------------------------
    # Stage 2: metadata_extraction — VLM analysis
    # ------------------------------------------------------------------
    pipeline_start = time.perf_counter()
    pixel_hash, source = _analysis_of_same_pixels(conn, settings, path_str)
    if source is not None:  # the same picture is indexed already, whatever its name and metadata: its analysis is used again
        on_stage("reusing the analysis of the same picture")
        pixel_hash = _pixel_hash_for_record(path_str, modified_at, pixel_hash)  # also checks the file did not change meanwhile
        record = {
            "root_id": root_id,
            "file_path": path_str,
            "hash": file_hash,
            "pixel_hash": pixel_hash,
            **{column: source[column] for column in REUSED_COLUMNS},
            **file_meta,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": int((time.perf_counter() - pipeline_start) * 1000),
        }
        on_stage("saving")
        with conn:
            file_id = upsert_image_data(conn, record)
            upsert_text_embedding(conn, file_id, source["embedding"], settings["embedding_model"], source["embedding_digest"],
                                  source["embedding_dim"])
            if source["from_deleted"]:  # the picture is indexed again: its kept analysis is no longer one of a deleted picture
                use_deleted_analysis(conn, pixel_hash)
        return ("updated" if is_update else "indexed"), source["short_description"]

    analysis = analyse_image(client, settings["vlm_model"], path_str, on_stage, on_retry)

    # ------------------------------------------------------------------
    # Stage 3: embedding_generation — search document + embedding vector
    # ------------------------------------------------------------------
    on_stage("embedding description")
    search_text = build_search_text(analysis.reply, analysis.ocr_text, config.SUMMARY_FIELD)
    if config.ONE_MODEL_AT_A_TIME:  # setup found the two models do not fit in memory together: the vision model goes before the embedding model comes
        unload_model(client, settings["vlm_model"])
    try:
        embedding = embed_search_text(client, settings["embedding_model"], search_text)
    finally:
        if config.ONE_MODEL_AT_A_TIME:  # ... and the embedding model goes before the next image's vision model comes
            unload_model(client, settings["embedding_model"])
    if len(embedding) != settings["embedding_dim"]:
        raise RunStopped(
            f"The embedding model returned {len(embedding)} values, but {settings['embedding_dim']} "
            "were measured at setup. The model has changed, so no image can be indexed with it."
        )
    embedding_bytes = serialize_embedding(embedding)
    elapsed_ms = int((time.perf_counter() - pipeline_start) * 1000)

    pixel_hash = _pixel_hash_for_record(path_str, modified_at, pixel_hash)

    vlm_metadata_json = build_vlm_metadata_json(analysis)
    short_description = str(analysis.reply[config.SUMMARY_FIELD]).strip()

    # ------------------------------------------------------------------
    # Stage 4: database_write — atomic transaction
    # ------------------------------------------------------------------
    record = {
        "root_id": root_id,
        "file_path": path_str,
        "hash": file_hash,
        "pixel_hash": pixel_hash,
        "short_description": short_description,
        "description": build_description(analysis.reply, config.SUMMARY_FIELD),
        "vlm_metadata_json": vlm_metadata_json,
        **file_meta,
        **{column: versions.get(column) for column in VERSION_COLUMNS},
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "processing_time_ms": elapsed_ms,
        "metadata_version": METADATA_VERSION,
        "vlm_model": settings["vlm_model"],
    }

    on_stage("saving")
    # ``with conn`` starts a transaction; commits on success, rolls back on error.
    with conn:
        file_id = upsert_image_data(conn, record)
        upsert_text_embedding(conn, file_id, embedding_bytes, settings["embedding_model"], versions.get("embedding_digest"), len(embedding))
        add_failure_history(conn, path_str, analysis.attempts)
    # ↑ Commit point — image is now considered fully indexed in the database.

    return ("updated" if is_update else "indexed"), short_description


def _analysis_of_same_pixels(conn, settings: dict, path: str) -> tuple[str | None, dict | None]:
    """``(pixel hash, analysis)`` for the picture at ``path``: the analysis is that of another indexed image with the same pixels
    (see ``find_analysis_by_pixels``), None when there is none, or when the picture's pixel hash or what an analysis made now
    would be made with is not known. The pixel hash is None when it could not be told; it is not the last word on the picture:
    ``_pixel_hash_for_record`` makes it final."""
    versions = settings.get("versions", {})
    wanted = {
        "vlm_model": settings["vlm_model"], "vlm_digest": versions.get("vlm_digest"), "prompt_hash": versions.get("prompt_hash"),
        "schema_hash": versions.get("schema_hash"), "search_text_version": versions.get("search_text_version"),
        "ocr_version": versions.get("ocr_version"), "metadata_version": METADATA_VERSION,
        "embedding_model": settings["embedding_model"], "embedding_digest": versions.get("embedding_digest"),
        "embedding_dim": settings["embedding_dim"],
    }
    if any(value is None for value in wanted.values()):
        return None, None
    pixel_hash = _pixel_hash_or_none(path)  # a picture that cannot be hashed here is analysed, and its failure (if any) reported, as always
    if pixel_hash is None:
        return None, None
    return pixel_hash, find_analysis_by_pixels(conn, pixel_hash, path, wanted)


def _pixel_hash_for_record(path: str, modified_at: str, known: str | None = None) -> str | None:
    """The pixel hash to store with an image that was just analysed (see ``compute_pixel_hash``; ``known`` is the one already
    worked out, which saves reading the picture again), or None for a picture Pillow cannot read, which was analysed through a
    converted copy: a scan then counts every change of such a file as an edit.

    Raises ``ImageProblem`` (``BUSY``, nothing held against the file) when the file changed since indexing began: its analysis
    would then belong to other pixels than the ones hashed here, and the next scan would not see that."""
    try:
        pixel_hash = known or compute_pixel_hash(path)
    except Exception as exc:
        found = explain(exc, path)
        if found is None:
            raise
        if found.kind != UNSUPPORTED:
            raise found from exc
        pixel_hash = None
    if iso_utc(os.stat(path).st_mtime) != modified_at:
        raise problem(BUSY, "changed_since_scan", step="checked the file did not change while it was analysed", park=False, report=False)
    return pixel_hash


def _index_with_tries(conn, client: ollama.Client, settings: dict, root_id: int, image_path: Path,
                      on_stage: Callable[[str], None], on_retry: Callable[[str], None]) -> tuple[str, str]:
    """``index_image``, tried again right away when the failure is Ollama's or the machine's, up to ``config.SERVICE_TRIES`` times.

    Each try runs the steps for its kind of trouble (waiting longer, starting Ollama again, freeing memory). What a failed
    try was is shown while the next one runs, and kept in the record of what was tried. The last failure is raised."""
    prefix = ""
    for attempt in range(1, config.SERVICE_TRIES + 1):
        try:
            return index_image(conn, client, settings, root_id, image_path, lambda stage: on_stage(prefix + stage), on_retry)
        except IndexingFailure as failure:
            if not failure.service or attempt == config.SERVICE_TRIES:
                raise
            with conn:
                add_failure_history(conn, str(image_path), failure.attempts)
            prefix = f"{failure.kind.replace('_', ' ')}: trying again, {attempt + 1} of {config.SERVICE_TRIES} · "
            on_stage(prefix + "starting")
    raise AssertionError("unreachable")  # config.SERVICE_TRIES is at least 1


# ---------------------------------------------------------------------------
# Trying an image again with lower settings
# ---------------------------------------------------------------------------

LOWER_SETTINGS_KINDS = (OUT_OF_MEMORY, SERVER_CRASH, TIMEOUT)  # the troubles that a lighter or a more patient way of asking can put right


def _long_side(image_path: Path) -> int | None:
    """The longer side of the picture in pixels, read from its header; None if it cannot be read."""
    try:
        with Image.open(image_path) as image:
            return max(image.size)
    except Exception:
        return None


def lower_settings_steps(kind: str, long_side: int | None) -> list[dict]:
    """What to change, one step after the other, to try an image again that failed with ``kind``.

    Each step is the changes to the usual device settings (see ``config.temporary_device_settings``) and builds on the one before:
      * no reply in time: start waiting where the failed tries stopped waiting, then ask for smaller pictures;
      * out of memory, or Ollama stopped (which is often what running out of graphics memory looks like): load the model paged in
        from disk, which needs less memory, if it is not already, then ask for smaller pictures.
    The sizes go down as calibration's do (``config.CALIBRATION_EDGE_STEP``) and stop where it stops (``CALIBRATION_MIN_EDGE``).
    They start below the picture's own long side: a size at or above it would change nothing. The context window stays as it is: it
    holds the largest picture, so a smaller one fits it too."""
    usual = config.device_settings()
    changes: dict = {}
    steps: list[dict] = []
    if kind == TIMEOUT:
        changes["timeout_seconds"] = usual["timeout_seconds"] * config.OLLAMA_TIMEOUT_STEP ** config.OLLAMA_TIMEOUT_RAISES
        steps.append(dict(changes))
    elif not usual["use_mmap"]:
        changes["use_mmap"] = True
        steps.append(dict(changes))
    edge = min(usual["image_long_edge"], long_side or usual["image_long_edge"])
    while True:
        edge = int(edge * config.CALIBRATION_EDGE_STEP)
        if edge < config.CALIBRATION_MIN_EDGE:
            return steps
        changes = {**changes, "image_long_edge": edge}
        steps.append(dict(changes))


def _describe_changes(changes: dict) -> str:
    parts = []
    if "timeout_seconds" in changes:
        parts.append(f"waiting up to {changes['timeout_seconds']} s")
    if "use_mmap" in changes:
        parts.append("model paged in from disk")
    if "image_long_edge" in changes:
        parts.append(f"pictures up to {changes['image_long_edge']} px")
    return ", ".join(parts)


def _index_with_lower_settings(conn, client: ollama.Client, settings: dict, row, kind: str,
                               on_stage: Callable[[str], None], on_retry: Callable[[str], None]) -> tuple[str, str, dict] | None:
    """``index_image`` for an image that failed with ``kind``, tried again with lower settings: each step of ``lower_settings_steps``
    in turn, until one works. The settings are lower only while that one try runs, so no other image and no later run sees them.

    Returns ``(result, short_description, what_was_changed)``, or None when no step worked (or there was none). A failure that is
    not the computer's (the model's reply, the picture) is raised: lower settings do not put it right."""
    path = Path(row["file_path"])
    for changes in lower_settings_steps(kind, _long_side(path)):
        label = f"lower settings ({_describe_changes(changes)}) · "
        with config.temporary_device_settings(**changes):
            if kind != TIMEOUT:
                for name in (settings["vlm_model"], settings["embedding_model"]):
                    unload_model(client, name)  # the memory is free again, and the models load with the settings of this try
            step_client = make_client(timeout=changes["timeout_seconds"]) if "timeout_seconds" in changes else client
            try:
                result, description = index_image(conn, step_client, settings, row["root_id"], path,
                                                  lambda stage: on_stage(label + stage), on_retry)
            except IndexingFailure as failure:
                if not failure.service:
                    raise
                with conn:
                    add_failure_history(conn, row["file_path"], failure.attempts)
                continue
        return result, description, changes
    return None


def _offer_lower_settings(conn, client: ollama.Client, settings: dict, failures: list, parked_versions: dict | None) -> None:
    """Tell the user which images were not indexed because of the computer (its memory or its speed, not the images) and why, and
    ask whether to try them again with lower settings. The settings are lower for these images only, and for each one by what went
    wrong with it: the usual settings stay as they are, and every other image is indexed as before.

    ``failures`` holds ``(backlog row, kind of trouble, message)`` for each of them."""
    width = max(shutil.get_terminal_size((80, 24)).columns - 1, 40)
    print(f"\n{len(failures)} image(s) were not indexed, and the trouble was the computer's, not the images':")
    for row, _, message in failures[:10]:
        print(f"\n  {file_link(row['file_path'], row['file_path'])}")
        print(textwrap.fill(message, width, initial_indent="    ", subsequent_indent="    "))
    if len(failures) > 10:
        print(f"\n  … and {len(failures) - 10} more.")
    if questionary is None or not sys.stdin.isatty():
        print("\nThey stay queued. To try them again with lower settings, scan again in a terminal.")
        return
    print("\nImagent can try each of them again with lower settings, chosen for what went wrong with that image: a longer wait, "
          "the model paged in from disk, a smaller picture. Your usual settings do not change and no other image is affected. "
          "A picture read at a smaller size is described in less detail.")
    if not questionary.confirm("Try these images again with lower settings?", default=False, style=STYLE).ask():
        print("They stay queued, and the next scan tries them with your usual settings.")
        return
    indexed = 0
    progress = Progress(len(failures))
    try:
        for i, (row, kind, _) in enumerate(failures, start=1):
            progress.start(i, row["file_path"])
            try:
                found = _index_with_lower_settings(conn, client, settings, row, kind, progress.stage, lambda kind: None)
            except IndexingFailure as exc:  # not the computer's: set aside or left queued, as in the run itself
                with conn:
                    add_failure_history(conn, row["file_path"], exc.attempts)
                    if exc.park:
                        park_backlog(conn, row["file_path"], exc.kind, exc.error_id, exc.retry_when, parked_versions, exc.params)
                    else:
                        record_backlog_failure(conn, row["file_path"], exc.error_id, count_attempt=False, kind=exc.kind if exc.report else None,
                                               params=exc.params)
                progress.done("FAILED", str(exc))
                continue
            except sqlite3.Error as exc:
                progress.done("FAILED", describe_db_error(exc))
                break
            except Exception as exc:
                progress.done("FAILED", f"{type(exc).__name__}: {exc}")
                continue
            if found is None:
                progress.done("FAILED", "Still not indexed, even with lower settings. It stays queued.")
                continue
            result, description, changes = found
            with conn:
                delete_backlog(conn, row["file_path"])
            indexed += 1
            progress.done(result, f"({_describe_changes(changes)}) {description}")
    except KeyboardInterrupt:
        progress.close()
        print("  interrupted")
    finally:
        progress.close()
    print(f"\n{indexed} of {len(failures)} image(s) indexed with lower settings; {len(failures) - indexed} stay queued.")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run(root_ids: set[int] | None = None) -> None:
    """Index the backlog (only the files of the folders ``root_ids`` when given); only one indexing run can be active at a time (a second one says so and stops)."""
    make_output_safe()
    try:
        with single_instance("indexing", "Indexing"):
            _index_backlog(root_ids)
    except AlreadyRunning as exc:
        sys.exit(f"{exc}")


def _index_backlog(root_ids: set[int] | None = None) -> None:
    """Initialise the database, pull pending files from the backlog (only those of the folders ``root_ids`` when given), and index them.

    Progress is printed for each image.  Failures are logged to ``error_logs``
    and their backlog row is kept (with the reason) so the file is retried on the
    next run; every other outcome (indexed/updated/skipped) removes the row from the
    backlog.  The run stops early if Ollama keeps failing, if the database cannot be
    written (disk full, for example; the files stay queued, none of their attempts used),
    or on Ctrl+C.  A summary is printed at the end.
    """
    db_path = str(config.DB_PATH)
    conn = init_db(db_path)

    try:
        client, settings = check_ready(conn, ("vlm", "embedding"))
    except NotReady as exc:
        conn.close()
        sys.exit(f"{exc}")

    settings = {**settings, "versions": collect_versions(client, settings)}
    parked_versions = model_versions(settings["versions"])
    if parked_versions:
        with conn:
            queued_again = requeue_changed_parked(conn, parked_versions)
        if queued_again:
            print(f"{queued_again} file(s) set aside because of the model's replies are queued again: the model or the prompt changed.")

    backlog_rows = get_backlog(conn, config.BACKLOG_MAX_ATTEMPTS)
    if root_ids is not None:
        backlog_rows = [row for row in backlog_rows if row["root_id"] in root_ids]
    total = len(backlog_rows)

    if total == 0:
        print("Nothing to index.")
        print_failed(conn)
        conn.close()
        return

    # Text in images is read by OCR. Without its models an image would be stored without its text and never
    # read again, so this is settled before the first image, not found out image by image.
    print()
    print("Loading the text-reading (OCR) engine…")
    try:
        ocr.check_available()
    except Exception as exc:
        conn.close()
        sys.exit(f"The text-reading (OCR) engine could not be loaded ({type(exc).__name__}: {exc}). "
                 f"Run: {config.SETUP_COMMAND}. Nothing was indexed; the backlog is kept.")
    # The models load here, so the time of the first image is not the time of loading them. When setup found that the two do not
    # fit in memory together, only the vision model is loaded: the embedding model comes after it, for each image (index_image).
    print("Loading the vision and embedding models…")
    load_model(client, settings["vlm_model"])
    if not config.ONE_MODEL_AT_A_TIME:
        load_model(client, settings["embedding_model"], embedding=True)
    print()

    counts: dict[str, int] = {"indexed": 0, "updated": 0, "skipped": 0, "failed": 0}
    wall_start = time.perf_counter()
    in_a_row = 0  # consecutive images that failed for a reason that is not theirs (Ollama, the machine)
    last_service_error = ""
    lower_settings_offer: list[tuple] = []  # (backlog row, kind, message) of images that failed for the computer's trouble: the user may try them again with lower settings
    can_offer = True  # not after Ctrl+C, a database that could not be written, or a model that changed
    io_failures: list[str] = []  # files that failed one after another with an error of the file system: a drive that went away, perhaps
    stopped = ""
    recent_retries: deque[bool] = deque(maxlen=config.RETRY_RATE_WINDOW)  # for the latest analysed images: did the model's reply need asking for again?
    progress = Progress(total)
    root_paths = {root["root_id"]: root["path"] for root in list_roots(conn)}

    try:
        for i, row in enumerate(backlog_rows, start=1):
            image_path = Path(row["file_path"])
            progress.start(i, image_path)

            retried: list[str] = []
            try:
                try:
                    result, short_description = _index_with_tries(conn, client, settings, row["root_id"], image_path, progress.stage,
                                                                  retried.append)
                    with conn:
                        delete_backlog(conn, row["file_path"])
                    counts[result] += 1
                    progress.done(result, short_description)
                    in_a_row = 0
                    io_failures.clear()
                    if result != "skipped":
                        recent_retries.append(bool(retried))
                except sqlite3.Error:
                    raise  # the database is the problem, not this image: handled below
                except IndexingFailure as exc:
                    if exc.kind == UNREADABLE and _changed_since_scan(image_path, row["modified_at"]):
                        exc = ImageProblem(BUSY, "changed_since_scan", exc.attempts, park=False, report=False)
                    # Every step for this kind of trouble was tried, so trying again would only repeat them. The file is set
                    # aside with the whole record of what was tried (it is queued again when the file changes), or, when the
                    # fault is not the file's, left queued with its attempts untouched.
                    counts["failed"] += 1
                    params = dict(exc.params)
                    if exc.service:  # the machine's trouble: it was tried several times, and the user is told what to do about it
                        params["tries"] = config.SERVICE_TRIES
                    message = describe(exc.error_id, params)
                    log_error(conn, stage="pipeline", exc=exc, file_path=str(image_path),
                              metadata={"root_id": row["root_id"], "kind": exc.kind})
                    with conn:
                        add_failure_history(conn, row["file_path"], exc.attempts)
                        if exc.park:
                            park_backlog(conn, row["file_path"], exc.kind, exc.error_id, exc.retry_when, parked_versions, params)
                        else:
                            record_backlog_failure(conn, row["file_path"], exc.error_id, count_attempt=False,
                                                   kind=exc.kind if exc.report else None, params=params)
                    progress.done("FAILED", message)
                    io_failures.clear()
                    if exc.service:
                        in_a_row += 1
                        last_service_error = str(exc)
                        if exc.kind in LOWER_SETTINGS_KINDS:
                            lower_settings_offer.append((row, exc.kind, message))
                    else:
                        in_a_row = 0
                    if exc.model_trouble:
                        recent_retries.append(True)
                except RunStopped as exc:
                    counts["failed"] += 1
                    progress.done("FAILED", str(exc))
                    stopped = f"Indexing stopped. {exc} Nothing was held against the files; the rest stays queued."
                    can_offer = False
                    break
                except Exception as exc:
                    if isinstance(exc, FileNotFoundError) and _is_gone(image_path, root_paths.get(row["root_id"])):
                        # Deleted since the scan, while its folder is still there: nothing to index. The next scan removes any stale record.
                        with conn:
                            delete_backlog(conn, row["file_path"])
                        progress.done("gone")
                        continue
                    counts["failed"] += 1
                    counts_as_attempt, ollama_problem = classify_failure(exc)
                    log_error(
                        conn,
                        stage="pipeline",
                        exc=exc,
                        file_path=str(image_path),
                        metadata={"root_id": row["root_id"]},
                    )
                    with conn:
                        record_backlog_failure(conn, row["file_path"], "unexpected", counts_as_attempt,
                                               params={"type": type(exc).__name__, "text": str(exc)})
                    progress.done("FAILED", f"{type(exc).__name__}: {exc}")
                    if ollama_problem:
                        in_a_row += 1
                        last_service_error = f"{type(exc).__name__}: {exc}"
                    else:
                        in_a_row = 0
                    if isinstance(exc, OSError) and not ollama_problem:
                        io_failures.append(row["file_path"])
                    else:
                        io_failures.clear()
                    if len(io_failures) >= config.MAX_CONSECUTIVE_SERVICE_FAILURES:
                        with conn:
                            refund_backlog_attempts(conn, io_failures)  # a drive that went away is not these files' fault
                        stopped = (f"Reading the images failed {len(io_failures)} times in a row (last error: {type(exc).__name__}: {exc}), "
                                   "so indexing stopped. Is the drive still connected and readable? "
                                   "The files keep all their attempts; the rest stays queued.")
                        break
                if in_a_row >= config.MAX_CONSECUTIVE_SERVICE_FAILURES:
                    stopped = (f"{in_a_row} images in a row failed for a reason that is not theirs, so indexing stopped "
                               f"(last error: {last_service_error}). Check that Ollama is running and has enough memory "
                               "and that the text-reading models are intact, then run it again; the rest stays queued.")
                    break
                if len(recent_retries) == recent_retries.maxlen and sum(recent_retries) / len(recent_retries) >= config.RETRY_RATE_STOP:
                    stopped = (f"The model's reply had to be asked for again for {sum(recent_retries)} of the last {len(recent_retries)} images, "
                               "so indexing stopped: this points at the model or the prompt, not at the images. "
                               "Nothing was held against the images. Close other apps that use a lot of memory, then scan again; "
                               "the rest stays queued.")
                    break
            except sqlite3.Error as exc:
                counts["failed"] += 1
                message = describe_db_error(exc)
                progress.done("FAILED", message)
                stopped = f"Indexing stopped. {message} Finished images are saved and the rest stays queued."
                can_offer = False
                break
    except KeyboardInterrupt:
        progress.close()
        print("  interrupted")
        can_offer = False
        stopped = "Interrupted. Finished images are saved; the rest stays queued."
    finally:
        progress.close()

    elapsed = time.perf_counter() - wall_start
    if settings["vlm_model"] != settings["judge_model"]:
        unload_model(client, settings["vlm_model"])  # searching does not need the VLM; free its memory for the judge model

    # Final summary
    print(f"\n{'─' * 52}")
    print(f"  Total in backlog  : {total}")
    print(f"  Indexed (new)     : {counts['indexed']}")
    print(f"  Updated           : {counts['updated']}")
    print(f"  Skipped           : {counts['skipped']}")
    print(f"  Failed            : {counts['failed']}")
    print(f"  Total elapsed     : {format_duration(elapsed, decimals=2)}")
    processed = counts["indexed"] + counts["updated"]
    if processed:
        print(f"  Average per image : {format_duration(elapsed / processed, decimals=2)}")
    print(f"{'─' * 52}")
    if stopped:
        print(f"\n{stopped}")
    try:
        print_failed(conn)
    except sqlite3.Error:
        pass  # the database is what stopped the run; its message is above
    if lower_settings_offer and can_offer:
        _offer_lower_settings(conn, client, settings, lower_settings_offer, parked_versions)
    print()

    conn.close()
    if stopped:
        sys.exit(1)


def _is_gone(image_path: Path, root_path: str | None) -> bool:
    """True if the file was deleted while the folder it was found in is still there.

    A missing file whose folder is missing too points at a drive that went away, or at a missing program file, and is
    not the file's doing: it is handled as a failure to read, which stops the run after a few in a row."""
    return root_path is not None and Path(root_path).is_dir() and not image_path.exists()


def print_failed(conn, limit: int | None = 10) -> int:
    """Tell the user which images could not be indexed, each with the reason and what can be done about it. Returns how many there are.

    Only images whose every try is over are listed; one that is indexed later (or whose file is gone) drops out of the list by itself.
    ``limit`` is how many to show; None shows them all."""
    failed = get_exhausted_backlog(conn, config.BACKLOG_MAX_ATTEMPTS)
    if not failed:
        return 0
    print(f"\n{len(failed)} image(s) could not be indexed:")
    width = max(shutil.get_terminal_size((80, 24)).columns - 1, 40)
    by_folder = defaultdict(list)
    for item in sorted(failed, key=lambda item: (os.path.dirname(item["file_path"]), os.path.basename(item["file_path"])))[:limit]:
        by_folder[os.path.dirname(item["file_path"])].append(item)
    bold = sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ
    for folder, items in by_folder.items():
        heading = f"\033[1m{folder}\033[0m" if bold else folder
        print(f"\n  {heading}")
        for item in items:
            print(f"    {file_link(os.path.basename(item['file_path']), item['file_path'])}")
            print(textwrap.fill(describe(item["error_id"], item["error_params"]) or "No reason was recorded.", width, initial_indent="      ", subsequent_indent="      "))
    if limit is not None and len(failed) > limit:
        print(f"\n  … and {len(failed) - limit} more. Choose \"Show the images that could not be indexed\" in the menu to see them all.")
    return len(failed)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

class RootOverlapError(ValueError):
    """The folder contains, or lies inside, a folder that was scanned before."""


def _inside(path: str, folders: list[str]) -> bool:
    return any(path.startswith(folder + os.sep) for folder in folders)


def _warn_about(items: list, problem: str, consequence: str, shown: int = 3) -> None:
    """Tell the user about folders or files the scan had trouble with: the first few by name, then how many more."""
    if not items:
        return
    print(f"{len(items)} {problem}; {consequence}")
    for item in items[:shown]:
        print(f"  {item}")
    if len(items) > shown:
        print(f"  … and {len(items) - shown} more")


def _pixel_hash_or_none(path: str) -> str | None:
    """The pixel hash of the picture at ``path`` (see ``compute_pixel_hash``), or None when it cannot be told: the picture is too
    large for the memory free, damaged, or in a format Pillow does not read. A scan never guesses from None: a file whose pixel
    hash is unknown counts as edited, and one that cannot be compared is not taken for a moved file."""
    try:
        check_image_size(path)
        return compute_pixel_hash(path)
    except Exception:
        return None


def _picture_size(path: str) -> tuple[int, int] | None:
    """The sides of the picture, the shorter first (a picture turned by its orientation keeps them), read from the file's header."""
    try:
        with Image.open(path) as image:
            return tuple(sorted(image.size))
    except Exception:
        return None


def _is_missing(path: str) -> bool:
    """True if nothing is at ``path``. A path that cannot be looked at (a folder without permission) is not missing, only unknown."""
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def _pair_moves(new_by_key: dict[str, list[str]], gone_by_key: dict[str, list[str]], gone: dict[str, StoredImage],
                on_disk: dict[str, str]) -> dict[str, str]:
    """Pair each new path with the path it was moved from: ``new_by_key`` and ``gone_by_key`` list the new and the vanished paths
    by the hash of what they hold, and each new path is paired with a vanished one of the same hash (``{new path: old path}``).

    Of identical copies, the pair that looks most like a move is made first: the same modified time and name, then the same time
    (a move keeps it), then the same name, then any. So when two copies of a picture exist and one was renamed, the one that was
    renamed is the one the report says moved."""
    ways = (lambda path, modified: (modified, os.path.basename(path)), lambda path, modified: modified,
            lambda path, modified: os.path.basename(path), lambda path, modified: None)
    pairs: dict[str, str] = {}
    for key, new_paths in new_by_key.items():
        old_paths, new_paths = list(gone_by_key.get(key, ())), list(new_paths)
        for way in ways:
            if not old_paths or not new_paths:
                break
            candidates = defaultdict(deque)
            for old in old_paths:
                candidates[way(old, gone[old].modified_at)].append(old)
            unpaired = []
            for new in new_paths:
                if candidates.get(way(new, on_disk[new])):
                    pairs[new] = candidates[way(new, on_disk[new])].popleft()
                else:
                    unpaired.append(new)
            used = set(pairs.values())
            old_paths = [old for old in old_paths if old not in used]
            new_paths = unpaired
    return pairs


def _find_moves(conn, new_paths: list[str], gone: dict[str, StoredImage], on_disk: dict[str, str], sizes: dict[str, int],
                file_hashes: dict[str, str | None], counts: dict[str, int]) -> dict[str, str]:
    """Which of ``new_paths`` are pictures of ``gone`` (indexed, but no longer at their path) at another path: ``{new path: old path}``.

    First by the bytes of the file (the hash of the whole file), then, for what is left, by the picture (the pixel hash), which also
    finds a file whose EXIF or other embedded data was edited as it was moved. A file is only read when something that could be its
    old self exists: by its size for the first, by the sides of the picture for the second, so a folder of new pictures is not read.
    ``file_hashes`` keeps the hash of each new file read (None when it could not be), for the caller to use."""
    if not gone or not new_paths:
        return {}

    def hash_of(path: str) -> str | None:
        if path not in file_hashes:
            try:
                file_hashes[path] = compute_hash(path)
            except OSError as exc:
                counts["unreadable"] += 1
                log_error(conn, stage="scan", exc=exc, file_path=path)
                file_hashes[path] = None
        return file_hashes[path]

    gone_by_hash: dict[str, list[str]] = defaultdict(list)
    for path, image in gone.items():
        gone_by_hash[image.hash].append(path)
    sizes_gone = {image.file_size for image in gone.values()}
    new_by_hash: dict[str, list[str]] = defaultdict(list)
    for path in new_paths:
        if None in sizes_gone or sizes[path] in sizes_gone:
            digest = hash_of(path)
            if digest is not None:
                new_by_hash[digest].append(path)
    moved = _pair_moves(new_by_hash, gone_by_hash, gone, on_disk)

    taken = set(moved.values())
    left_gone = {path: image for path, image in gone.items() if path not in taken and image.pixel_hash}
    left_new = [path for path in new_paths if path not in moved and file_hashes.get(path, "") is not None]
    if left_gone and left_new:
        gone_by_pixels: dict[str, list[str]] = defaultdict(list)
        for path, image in left_gone.items():
            gone_by_pixels[image.pixel_hash].append(path)
        sides_gone = {tuple(sorted((image.width, image.height))) if image.width and image.height else None
                      for image in left_gone.values()}
        new_by_pixels: dict[str, list[str]] = defaultdict(list)
        for path in left_new:
            if None not in sides_gone and _picture_size(path) not in sides_gone:
                continue
            pixel_hash = _pixel_hash_or_none(path)
            if pixel_hash is not None and hash_of(path) is not None:
                new_by_pixels[pixel_hash].append(path)
        moved.update(_pair_moves(new_by_pixels, gone_by_pixels, left_gone, on_disk))
    return moved


def _gone_from_other_folders(conn, root_id: int) -> dict[str, StoredImage]:
    """The indexed images of the other scanned folders whose file is not there any more, while their folder is: pictures that may
    have been moved into the folder being scanned. A folder that cannot be read now (a drive that is not connected) gives none:
    its images are unknown, not gone."""
    folders = {root["root_id"]: Path(root["path"]).is_dir() for root in list_roots(conn)}
    return {path: image for other, path, image in images_outside_root(conn, root_id)
            if folders.get(other) and _is_missing(path)}


def _apply_moves(conn, moved: dict[str, str], gone: dict[str, StoredImage], root_id: int, on_disk: dict[str, str],
                 file_hashes: dict[str, str | None], counts: dict[str, int]) -> None:
    """Point the record of each moved picture at its new path (and folder). When the file's bytes differ from before (the same
    picture with other embedded data) the facts stored about the file are refreshed too; the analysis is left as it is."""
    for new_path, old_path in moved.items():
        image = gone[old_path]
        details = None
        if file_hashes.get(new_path) != image.hash:
            try:
                details = {**extract_file_metadata(new_path), "modified_at": on_disk[new_path]}
            except OSError as exc:
                counts["unreadable"] += 1
                log_error(conn, stage="scan", exc=exc, file_path=new_path)
        with conn:
            update_image_location(conn, image.file_id, root_id, new_path, on_disk[new_path])
            if details is not None:
                refresh_image_file(conn, image.file_id, file_hashes[new_path], details)


def _register_root(conn, root: Path) -> int:
    """Add ``root`` to the scanned folders (nothing happens if it was added before) and return its id; raises ``RootOverlapError`` if
    it contains, or lies inside, another added folder."""
    conflict = find_overlapping_root(root, [r["path"] for r in list_roots(conn)])
    if conflict:
        raise RootOverlapError(f"{root} overlaps the folder already scanned: {conflict}")
    with conn:
        return get_or_add_root(conn, str(root))


def scan(root: Path, allow_mass_delete: bool = False) -> dict[str, int]:
    """Reconcile the index with the files under ``root``; the other folders' images are never deleted.

    Every file is in exactly one of these states (the counts of the result). A file whose path is in the index is:
      * up to date (``up_to_date``): nothing about its picture or its EXIF facts changed, even if its timestamp or other data in
        the file did (what is stored about the file is brought up to date);
      * details updated (``details_updated``): the picture is the same but its EXIF facts changed; they are refreshed in the
        index, nothing is analysed or embedded again (the embedded text holds no EXIF);
      * changed (``queued_modified``): its picture is not the one that was analysed, the pixels after the EXIF orientation
        (queued; the old record stays until the new one is ready).
    A file whose path is not in the index is:
      * moved (``moved``): a picture indexed under another path, in another folder (a subfolder of this one, or another scanned
        folder), that is gone from there. Its record follows it and nothing is analysed again. Found by the file's bytes, then by
        its pixels;
      * renamed (``renamed``): the same, but the old path is in the same folder as the new one (only the name changed). The
        report shows it with the details updated;
      * new (``queued_new``): anything else (queued).
    An indexed file that is gone from its path and was not moved is removed (``deleted``).
    Three things protect the index from a scan that
    cannot see what is really there:
      * what lies in a subfolder that could not be read is neither deleted nor treated as moved;
      * if the folder shows no images at all while some are indexed (a drive that is not
        mounted, for example), or at least ``config.SCAN_GUARD_MIN_FILES`` indexed images are gone
        and they are more than ``config.SCAN_GUARD_MAX_SHARE`` of the folder's indexed images,
        nothing is deleted unless ``allow_mass_delete`` is True; ``delete_blocked`` in the
        result tells the caller.
    Files that cannot be indexed are counted and left out: empty files and files whose name
    is not valid text (``empty``, ``bad_names``).
    """
    conn = init_db(str(config.DB_PATH))
    counts = {
        "scanned": 0,
        "queued_new": 0,
        "queued_modified": 0,
        "details_updated": 0,
        "moved": 0,
        "renamed": 0,
        "up_to_date": 0,
        "deleted": 0,
        "delete_blocked": 0,
        "unreadable": 0,
        "unreadable_dirs": 0,
        "empty": 0,
        "bad_names": 0,
        "not_downloaded": 0,
    }

    try:
        root_id = _register_root(conn, root)

        # path -> modification time, for every image on disk right now.
        found = discover(root)
        counts["unreadable_dirs"] = len(found.unreadable_dirs)
        counts["bad_names"] = len(found.bad_names)
        _warn_about(found.unreadable_dirs, "folder(s) could not be read", "their images were left in the index as they are:")
        _warn_about(found.bad_names, "file(s) have a name that is not valid text", "rename them to have them indexed:")
        unseen = [str(folder) for folder in found.unreadable_dirs]
        on_disk: dict[str, str] = {}
        sizes: dict[str, int] = {}  # path -> size in bytes, for telling which new files could be an indexed one that moved
        not_downloaded: list[Path] = []  # in cloud storage, not on this computer: neither read (it would be downloaded) nor forgotten
        for file_path in found.images:
            try:
                status = os.stat(file_path)
            except OSError as exc:  # removed or locked while scanning
                counts["unreadable"] += 1
                log_error(conn, stage="scan", exc=exc, file_path=str(file_path))
                continue
            if is_not_downloaded(status):
                not_downloaded.append(file_path)
                continue
            if status.st_size == 0:
                counts["empty"] += 1
                continue
            on_disk[str(file_path)] = iso_utc(status.st_mtime)
            sizes[str(file_path)] = status.st_size
        counts["scanned"] = len(on_disk)
        counts["not_downloaded"] = len(not_downloaded)
        _warn_about(not_downloaded, "image(s) are in cloud storage and not downloaded to this computer",
                    "they were not read, and what is indexed of them was kept:")
        away = {str(path) for path in not_downloaded}

        stored = images_by_path(conn, root_id)
        # Whatever lies in a folder that could not be read is unknown, not gone; so is a file that is in the cloud only.
        vanished = [path for path in stored if path not in on_disk and path not in away and not _inside(path, unseen)]
        new_paths = [path for path in on_disk if path not in stored]

        # A new path holding a picture that is gone from its own path is that picture, moved or renamed: from this folder, or
        # from another scanned one. Files are only read when something that could be their old self is gone (see _find_moves).
        file_hashes: dict[str, str | None] = {}
        gone_here = {path: stored[path] for path in vanished}
        moved = _find_moves(conn, new_paths, gone_here, on_disk, sizes, file_hashes, counts)
        _apply_moves(conn, moved, gone_here, root_id, on_disk, file_hashes, counts)
        left = [path for path in new_paths if path not in moved]
        gone_elsewhere = _gone_from_other_folders(conn, root_id) if left else {}
        moved_in = _find_moves(conn, left, gone_elsewhere, on_disk, sizes, file_hashes, counts)
        _apply_moves(conn, moved_in, gone_elsewhere, root_id, on_disk, file_hashes, counts)
        for new_path, old_path in {**moved, **moved_in}.items():  # same folder, other name: renamed; another folder: moved
            counts["renamed" if os.path.dirname(new_path) == os.path.dirname(old_path) else "moved"] += 1

        # Files present before and now: up to date, or changed in some way. Only a file whose timestamp changed is looked at.
        for path, modified_at in on_disk.items():
            row = stored.get(path)
            if row is None:
                continue
            if row.modified_at == modified_at:
                counts["up_to_date"] += 1
                continue
            try:
                file_hash = compute_hash(path)
            except OSError as exc:
                counts["unreadable"] += 1
                log_error(conn, stage="scan", exc=exc, file_path=path)
                continue
            if file_hash == row.hash:  # a touch or a sync changed the timestamp, not the file
                with conn:
                    update_image_modified_at(conn, row.file_id, modified_at)
                counts["up_to_date"] += 1
                continue
            pixel_hash = _pixel_hash_or_none(path)
            if pixel_hash is None or pixel_hash != row.pixel_hash:  # not the picture that was analysed (or not known to be)
                with conn:
                    upsert_backlog(conn, path, root_id, modified_at)
                counts["queued_modified"] += 1
                continue
            try:  # the same picture in other bytes: the EXIF facts may differ, nothing else needs to be done about it
                details = {**extract_file_metadata(path), "modified_at": modified_at}
            except OSError as exc:
                counts["unreadable"] += 1
                log_error(conn, stage="scan", exc=exc, file_path=path)
                continue
            with conn:
                stored_exif = get_exif_fields(conn, row.file_id)
                exif_changed = any(value != details[field] for field, value in stored_exif.items())
                refresh_image_file(conn, row.file_id, file_hash, details)
            counts["details_updated" if exif_changed else "up_to_date"] += 1

        moved_paths = set(moved) | set(moved_in)
        with conn:
            for path in new_paths:
                if path not in moved_paths:
                    upsert_backlog(conn, path, root_id, on_disk[path])
                    counts["queued_new"] += 1

        # Forget what is gone: stored images, and queued files that vanished before they were indexed.
        moved_from = set(moved.values())
        gone = [path for path in vanished if path not in moved_from]
        mass_delete = gone and (
            not on_disk
            or (len(gone) >= config.SCAN_GUARD_MIN_FILES and len(gone) > len(stored) * config.SCAN_GUARD_MAX_SHARE)
        )
        if mass_delete and not allow_mass_delete:
            counts["delete_blocked"] = len(gone)
            gone = []
        stale_queue = [path for path in get_backlog_paths_for_root(conn, root_id)
                       if path not in on_disk and path not in away and not _inside(path, unseen)]
        gone_ids = [stored[path].file_id for path in gone]
        with conn:
            keep_analyses_of(conn, gone_ids, get_keep_deleted(conn))  # what the user chose at setup: the analysis may be used again if the picture comes back
            counts["deleted"] = delete_images(conn, gone_ids)
            delete_backlog_paths(conn, stale_queue)

        with conn:
            mark_root_scanned(conn, root_id)
        return counts
    finally:
        conn.close()


def _ask_yes_no(question: str) -> bool:
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:  # no terminal attached: the safe answer is no
        return False


def _scannable(saved: list[str]) -> list[Path]:
    """The folders added before that can be read now; each of the others is skipped, with a warning."""
    folders = []
    for path in saved:
        folder = Path(path)
        if folder.is_dir() and os.access(folder, os.R_OK):
            folders.append(folder)
        else:
            print(f"Skipping {path}: the folder is missing or cannot be read. Its indexed images are kept.")
    return folders


def _scan_and_report(root: Path) -> None:
    """Scan ``root`` and print what changed."""
    try:
        counts = scan(root)
    except RootOverlapError as exc:
        sys.exit(f"Error: {exc}")

    if counts["delete_blocked"] and _ask_yes_no(
        f"{counts['delete_blocked']} indexed image(s) from this folder cannot be found any more, "
        "which is all or most of them (is a drive disconnected or a folder unmounted?). "
        "Remove them from the index? [y/N]: "
    ):
        counts = scan(root, allow_mass_delete=True)

    skip_reasons = (("empty", counts["empty"]), ("not downloaded", counts["not_downloaded"]),
                    ("unreadable", counts["unreadable"] + counts["bad_names"]))
    skipped = sum(number for _, number in skip_reasons)
    why = ", ".join(f"{number} {reason}" for reason, number in skip_reasons if number)
    states = (  # what the scan found, every state always shown: the files that are fine first, then what changed, then the problems
        ("Already indexed", counts["up_to_date"] + counts["moved"] + counts["details_updated"] + counts["renamed"]),
        ("To index", counts["queued_new"] + counts["queued_modified"]),
        ("Deleted files", counts["deleted"]),
        ("Skipped", f"{skipped} ({why})" if why else skipped),
    )  # images kept after a declined mass deletion have no row: the note below explains them
    bold = sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ  # as in Progress
    if bold:
        os.system("")  # makes the Windows console understand ANSI escape codes
    print(f"\n\033[1m{root}\033[0m" if bold else f"\n{root}")
    for label, value in states:
        print(f"  {label:<15} : {value}")
    if counts["delete_blocked"]:
        print(f"\nNote: {counts['delete_blocked']} image(s) are no longer in this folder. Their index was kept in case they come back.\n"
              f"  Moved them?       Scan their new folder ({config.ADD_COMMAND} \"folder path\"). They are recognised\n"
              f"                    and not analysed again, which can save hours.\n"
              f"  Drive unplugged?  Reconnect it and run {config.SCAN_COMMAND}.\n"
              f"  Deleted them?     Run {config.SCAN_COMMAND} and answer y to delete their index.")


def _scan_folders(roots: list[Path]) -> None:
    """Scan each folder and print what changed in it, then offer to index what is pending."""
    for root in roots:
        _scan_and_report(root)
    print()

    conn = init_db(str(config.DB_PATH))
    try:
        backlog = get_backlog(conn, config.BACKLOG_MAX_ATTEMPTS)
        failed = {row["file_path"] for row in get_exhausted_backlog(conn, config.BACKLOG_MAX_ATTEMPTS)}
    finally:
        conn.close()
    backlog_size = len([row for row in backlog if row["file_path"] not in failed])  # a file left queued after its tries is failed, not pending
    if backlog_size == 0:
        return
    if _ask_yes_no(
        f"{backlog_size} file(s) in the backlog are pending indexing. "
        "Run indexing now? [y/N]: "
    ):
        run()
    else:
        print(f"Skipping indexing. Run {config.SCAN_COMMAND} later to process the backlog.")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def added_folders() -> list[str]:
    """The folders that were added, the most recently scanned first."""
    conn = init_db(str(config.DB_PATH))
    try:
        return [row["path"] for row in list_roots(conn)]
    finally:
        conn.close()


def _ask_for_folder(added: list[str]) -> str:
    """The folder to add, asked for in a terminal with suggestions; ends the program if there is no terminal or nothing was entered."""
    if questionary is None or not sys.stdin.isatty():  # nobody to ask
        sys.exit(f'No folder was given. Run: {config.ADD_COMMAND} "folder path"')
    folder = ask_for_folder(added, STYLE)
    if folder is None:  # Ctrl+C; already said so
        sys.exit(1)
    if not folder.strip():
        sys.exit("No folder entered.")
    return folder.strip()


def main() -> None:
    """Scan every folder that was added, then offer to index what was found."""
    make_output_safe()
    saved = added_folders()
    if not saved:
        if questionary is None or not sys.stdin.isatty():  # nobody to ask
            sys.exit(f'No folders have been added yet. Run: {config.ADD_COMMAND} "folder path"')
        print("No folders have been added yet, so there is nothing to scan. Add one now and it is scanned and indexed.")
        add()
        return
    roots = _scannable(saved)
    if not roots:
        sys.exit("Error: none of the folders added before can be read.")
    _scan_folders(roots)


def add_folder(folder: str | None = None) -> Path:
    """Add ``folder`` (asked for, with suggestions, when not given) to the folders that are scanned, without scanning it; returns the folder.

    Ends the program with a message if the folder cannot be used or overlaps one added before."""
    if folder is None:
        folder = _ask_for_folder(added_folders())
    root = validate_folder(folder)
    conn = init_db(str(config.DB_PATH))
    try:
        _register_root(conn, root)
    except RootOverlapError as exc:
        sys.exit(f"Error: {exc}")
    finally:
        conn.close()
    return root


def scan_and_index(root: Path) -> None:
    """Scan the added folder ``root``, print what changed in it, then index the pictures it has pending."""
    _scan_and_report(root)
    print()
    conn = init_db(str(config.DB_PATH))
    try:
        with conn:
            root_id = get_or_add_root(conn, str(root))
    finally:
        conn.close()
    run({root_id})


def add(folder: str | None = None) -> None:
    """Add ``folder`` (asked for, with suggestions, when not given) to the folders that are scanned, then offer what to do next (see after_add.py).

    With no terminal to answer, nothing is offered: the folder is scanned and, as a script expects, the pictures found are indexed only if asked."""
    make_output_safe()
    if questionary is None or not sys.stdin.isatty():
        _scan_folders([validate_folder(_ask_for_folder([]) if folder is None else folder)])
        return
    from imagent.commands import after_add  # after_add uses this module's functions
    after_add.run(add_folder(folder))
