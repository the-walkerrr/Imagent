#!/usr/bin/env python3
"""
scan_and_index/scan_and_index.py — Imagent: scan folders for images, then index them.

Scanning compares a folder with the index and queues the new or modified images (an image that moved or
was renamed keeps its record, and one that disappeared is forgotten). Indexing then takes every file queued
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

# Make the project root importable when this file is run directly (``python scan_and_index/scan_and_index.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import httpx
    import ollama

    import config
    from db.connection import describe_db_error, init_db
    from db.operations import (
        VERSION_COLUMNS,
        add_failure_history,
        delete_backlog,
        delete_backlog_paths,
        delete_images,
        get_backlog,
        get_backlog_paths_for_root,
        get_exhausted_backlog,
        get_or_add_root,
        list_images_for_root,
        list_roots,
        lookup_by_path,
        mark_root_scanned,
        park_backlog,
        record_backlog_failure,
        refund_backlog_attempts,
        requeue_changed_parked,
        update_image_location,
        update_image_modified_at,
        upsert_backlog,
        upsert_image_data,
        upsert_text_embedding,
    )
    from image_processing import ocr
    from image_processing.hashing import compute_hash
    from image_processing.metadata import extract_file_metadata, iso_utc
    from pipeline.console import make_output_safe, run_cli
    from pipeline.discovery import discover
    from pipeline.error_logger import log_error
    from pipeline.failure import IndexingFailure, RunStopped
    from pipeline.locking import AlreadyRunning, single_instance
    from pipeline.progress import Progress
    from pipeline.serialisation import serialize_embedding
    from pipeline.versions import collect_versions, model_versions
    from pipeline.validation import find_overlapping_root, validate_folder
    from text_processing.formatters import build_description, build_search_text
    from vlm.analyzer import analyse_image, embed_search_text
    from vlm.client import NotReady, check_ready, unload_model
except ModuleNotFoundError as exc:
    sys.exit(f"Imagent is not set up yet (the Python package '{exc.name}' is missing). "
             f"Run: {'install.bat' if sys.platform == 'win32' else 'sh install.sh'}")

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

    vlm_metadata_json = build_vlm_metadata_json(analysis)
    short_description = str(analysis.reply[config.SUMMARY_FIELD]).strip()

    # ------------------------------------------------------------------
    # Stage 4: database_write — atomic transaction
    # ------------------------------------------------------------------
    record = {
        "root_id": root_id,
        "file_path": path_str,
        "hash": file_hash,
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
# Pipeline runner
# ---------------------------------------------------------------------------

def run() -> None:
    """Index the backlog; only one indexing run can be active at a time (a second one says so and stops)."""
    make_output_safe()
    try:
        with single_instance("indexing", "Indexing"):
            _index_backlog()
    except AlreadyRunning as exc:
        sys.exit(f"{exc}")


def _index_backlog() -> None:
    """Initialise the database, pull pending files from the backlog, and index them.

    Progress is printed for each image.  Failures are logged to ``error_logs``
    and their backlog row is kept (with the reason) so the file is retried on the
    next run; every other outcome (indexed/updated/skipped) removes the row from the
    backlog.  The run stops early if Ollama keeps failing, if the database cannot be
    written (disk full, for example; the files stay queued, none of their attempts used),
    or on Ctrl+C.  A summary is printed at the end.
    """
    db_path = str(config.DB_PATH)
    conn = init_db(db_path)

    print(f"\nDatabase : {db_path}")

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
    total = len(backlog_rows)
    print(f"Backlog  : {total} file(s) pending\n")

    if total == 0:
        print("Nothing to index.")
        print_failed(conn)
        conn.close()
        return

    # Text in images is read by OCR. Without its models an image would be stored without its text and never
    # read again, so this is settled before the first image, not found out image by image.
    print("Loading the text-reading (OCR) engine…")
    try:
        ocr.check_available()
    except Exception as exc:
        conn.close()
        sys.exit(f"The text-reading (OCR) engine could not be loaded ({type(exc).__name__}: {exc}). "
                 f"Run: {config.INSTALL_COMMAND}. Nothing was indexed; the backlog is kept.")

    counts: dict[str, int] = {"indexed": 0, "updated": 0, "skipped": 0, "failed": 0}
    wall_start = time.perf_counter()
    in_a_row = 0  # consecutive images that failed for a reason that is not theirs (Ollama, the machine)
    last_service_error = ""
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
                    # Every step for this kind of trouble was tried, so trying again would only repeat them. The file is set
                    # aside with the whole record of what was tried (it is queued again when the file changes), or, when the
                    # fault is not the file's, left queued with its attempts untouched.
                    counts["failed"] += 1
                    message = str(exc)
                    if exc.service:  # the machine's trouble: it was tried several times, and the user is told what to do about it
                        message += (f" Imagent tried {config.SERVICE_TRIES} times without success. "
                                    "Close other apps that use a lot of memory, then scan again.")
                    log_error(conn, stage="pipeline", exc=exc, file_path=str(image_path),
                              metadata={"root_id": row["root_id"], "kind": exc.kind})
                    with conn:
                        add_failure_history(conn, row["file_path"], exc.attempts)
                        if exc.park:
                            park_backlog(conn, row["file_path"], exc.kind, message, exc.retry_when, parked_versions)
                        else:
                            record_backlog_failure(conn, row["file_path"], message, count_attempt=False,
                                                   kind=exc.kind if exc.report else None)
                    progress.done("FAILED", message)
                    io_failures.clear()
                    if exc.service:
                        in_a_row += 1
                        last_service_error = str(exc)
                    else:
                        in_a_row = 0
                    if exc.model_trouble:
                        recent_retries.append(True)
                except RunStopped as exc:
                    counts["failed"] += 1
                    progress.done("FAILED", str(exc))
                    stopped = f"Indexing stopped. {exc} Nothing was held against the files; the rest stays queued."
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
                        record_backlog_failure(conn, row["file_path"], f"{type(exc).__name__}: {exc}", counts_as_attempt)
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
                message = describe_db_error(exc, db_path)
                progress.done("FAILED", message)
                stopped = f"Indexing stopped. {message} Finished images are saved and the rest stays queued."
                break
    except KeyboardInterrupt:
        progress.close()
        print("  interrupted")
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
    print(f"  Total elapsed     : {elapsed:.2f}s")
    processed = counts["indexed"] + counts["updated"]
    if processed:
        print(f"  Average per image : {elapsed / processed:.2f}s")
    print(f"{'─' * 52}")
    if stopped:
        print(f"\n{stopped}")
    try:
        print_failed(conn)
    except sqlite3.Error:
        pass  # the database is what stopped the run; its message is above
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
    for item in failed[:limit]:
        print(f"\n  {item['file_path']}")
        print(textwrap.fill(item["last_error"] or "No reason was recorded.", width, initial_indent="    ", subsequent_indent="    "))
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


def scan(root: Path, allow_mass_delete: bool = False) -> dict[str, int]:
    """Reconcile the index with the files under ``root``; files of other roots are never touched.

    A file is: new (queued), modified (queued; the old record stays until the new one is
    ready), moved or renamed (its record follows it, no re-analysis), touched only (timestamp
    updated) or gone (its record is deleted). Three things protect the index from a scan that
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
        "moved": 0,
        "unchanged": 0,
        "deleted": 0,
        "delete_blocked": 0,
        "unreadable": 0,
        "unreadable_dirs": 0,
        "empty": 0,
        "bad_names": 0,
    }

    try:
        known = list_roots(conn)
        conflict = find_overlapping_root(root, [r["path"] for r in known])
        if conflict:
            raise RootOverlapError(f"{root} overlaps the folder already scanned: {conflict}")
        with conn:
            root_id = get_or_add_root(conn, str(root))

        # path -> modification time, for every image on disk right now.
        found = discover(root)
        counts["unreadable_dirs"] = len(found.unreadable_dirs)
        counts["bad_names"] = len(found.bad_names)
        _warn_about(found.unreadable_dirs, "folder(s) could not be read", "their images were left in the index as they are:")
        _warn_about(found.bad_names, "file(s) have a name that is not valid text", "rename them to have them indexed:")
        unseen = [str(folder) for folder in found.unreadable_dirs]
        on_disk: dict[str, str] = {}
        for file_path in found.images:
            try:
                status = os.stat(file_path)
            except OSError as exc:  # removed or locked while scanning
                counts["unreadable"] += 1
                log_error(conn, stage="scan", exc=exc, file_path=str(file_path))
                continue
            if status.st_size == 0:
                counts["empty"] += 1
                continue
            on_disk[str(file_path)] = iso_utc(status.st_mtime)
        counts["scanned"] = len(on_disk)

        stored = {row["file_path"]: row for row in list_images_for_root(conn, root_id)}
        # Whatever lies in a folder that could not be read is unknown, not gone.
        vanished = [path for path in stored if path not in on_disk and not _inside(path, unseen)]
        new_paths = [path for path in on_disk if path not in stored]

        # A new path whose content equals that of a vanished one is the same image moved or renamed.
        # New files are only hashed when something vanished, so a first scan reads no file twice.
        moved: dict[str, str] = {}  # new path -> old path
        if vanished and new_paths:
            vanished_by_hash = defaultdict(list)
            for path in vanished:
                vanished_by_hash[stored[path]["hash"]].append(path)
            for path in new_paths:
                try:
                    candidates = vanished_by_hash.get(compute_hash(path))
                except OSError as exc:
                    counts["unreadable"] += 1
                    log_error(conn, stage="scan", exc=exc, file_path=path)
                    continue
                if candidates:
                    moved[path] = candidates.pop(0)
        with conn:
            for new_path, old_path in moved.items():
                update_image_location(conn, stored[old_path]["file_id"], new_path, on_disk[new_path])
        counts["moved"] = len(moved)

        # Files present before and now: unchanged, touched only, or modified.
        for path, modified_at in on_disk.items():
            row = stored.get(path)
            if row is None:
                continue
            if row["modified_at"] == modified_at:
                counts["unchanged"] += 1
                continue
            try:
                content_changed = compute_hash(path) != row["hash"]
            except OSError as exc:
                counts["unreadable"] += 1
                log_error(conn, stage="scan", exc=exc, file_path=path)
                continue
            with conn:
                if content_changed:
                    upsert_backlog(conn, path, root_id, modified_at)
                else:  # a touch or a sync changed the timestamp, not the picture
                    update_image_modified_at(conn, row["file_id"], modified_at)
            counts["queued_modified" if content_changed else "unchanged"] += 1

        with conn:
            for path in new_paths:
                if path not in moved:
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
                       if path not in on_disk and not _inside(path, unseen)]
        with conn:
            counts["deleted"] = delete_images(conn, [stored[path]["file_id"] for path in gone])
            delete_backlog_paths(conn, stale_queue)

        with conn:
            mark_root_scanned(conn, root_id)
        return counts
    finally:
        conn.close()


def _prompt_folder() -> str | None:
    """Prompt for the folder to scan, offering all folders and previously scanned folders as selectable options.

    Returns the raw (unvalidated) path string entered by the user, or None if "All folders" was chosen.
    """
    conn = init_db(str(config.DB_PATH))
    try:
        saved: list[str] = [root["path"] for root in list_roots(conn)]
    finally:
        conn.close()

    if not saved:
        if questionary is not None and sys.stdin.isatty():
            ans = questionary.text("Enter the folder path to scan:", style=STYLE).ask()
            if ans is None:
                sys.exit("\nScan cancelled.")
            return ans.strip()
        return input("Enter the folder path to scan: ").strip()

    if questionary is not None and sys.stdin.isatty():
        choices = [
            questionary.Choice("All folders", value="__all__"),
            questionary.Choice("Enter folder path...", value="__input__"),
        ]
        for i, folder in enumerate(saved):
            title = f"{folder} (last used)" if i == 0 else folder
            choices.append(questionary.Choice(title, value=folder))

        selected = questionary.select(
            "Select a folder to scan:",
            choices=choices,
            style=STYLE,
        ).ask()

        if selected is None:
            sys.exit("\nScan cancelled.")

        if selected == "__all__":
            return None

        if selected == "__input__":
            raw = questionary.text("Enter the folder path to scan:", style=STYLE).ask()
            if raw is None:
                sys.exit("\nScan cancelled.")
            return raw.strip()

        return selected

    # Fallback for non-interactive / non-TTY environments
    print("\nSelect a folder to scan:")
    print("  1. All folders")
    print("  2. Enter folder path...")
    for i, folder in enumerate(saved, 3):
        tag = " (last used)" if i == 3 else ""
        print(f"  {i}. {folder}{tag}")
    print()

    raw = input("Enter choice or folder path: ").strip()
    if raw == "1":
        return None
    if raw == "2":
        return input("Enter the folder path to scan: ").strip()
    if raw.isdigit():
        idx = int(raw) - 3
        if 0 <= idx < len(saved):
            return saved[idx]
    return raw


def _ask_yes_no(question: str) -> bool:
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:  # no terminal attached: the safe answer is no
        return False


def _ask_scope(root: Path, scanned_before: int) -> str:
    """Ask whether to scan only the new folder ``root`` or all folders. Returns "one", "all" or "back"."""
    one = f"Only this folder ({root})"
    everything = f"All folders (this one and the {scanned_before} scanned before)"

    if questionary is not None and sys.stdin.isatty():
        choice = questionary.select(
            "How do you want to scan?",
            choices=[
                questionary.Choice(one, value="one"),
                questionary.Choice(everything, value="all"),
                questionary.Choice("Back", value="back"),
            ],
            style=STYLE,
        ).ask()
        if choice is None:
            sys.exit("\nScan cancelled.")
        return choice

    # Fallback for non-interactive / non-TTY environments
    print("\nHow do you want to scan?")
    print(f"  1. {one}")
    print(f"  2. {everything}")
    print("  3. Back\n")
    while True:
        try:
            answer = input("Enter choice: ").strip()
        except EOFError:  # no terminal attached: the safe answer is to cancel
            sys.exit("\nScan cancelled.")
        if answer in ("1", "2", "3"):
            return ("one", "all", "back")[int(answer) - 1]


def _scannable(saved: list[str]) -> list[Path]:
    """The folders scanned before that can be read now; each of the others is skipped, with a warning."""
    folders = []
    for path in saved:
        folder = Path(path)
        if folder.is_dir() and os.access(folder, os.R_OK):
            folders.append(folder)
        else:
            print(f"Skipping {path}: the folder is missing or cannot be read. Its indexed images are kept.")
    return folders


def _choose_roots() -> list[Path]:
    """Prompt until the user has settled what to scan, and return those folders.

    "All folders" in the list scans every folder scanned before. A folder scanned before, or the very first folder,
    is scanned alone. For a new folder when others were scanned before, the user chooses between it alone and all
    folders, or goes back to the folder list. A folder scanned before that is now missing or unreadable is skipped,
    with a warning.
    """
    while True:
        raw = _prompt_folder()
        conn = init_db(str(config.DB_PATH))
        try:
            saved = [row["path"] for row in list_roots(conn)]
        finally:
            conn.close()
        if raw is None:
            roots = _scannable(saved)
            if not roots:
                sys.exit("Error: none of the folders scanned before can be read.")
            return roots

        root = validate_folder(raw)
        if not saved or str(root) in saved:
            return [root]

        conflict = find_overlapping_root(root, saved)
        if conflict:
            raise RootOverlapError(f"{root} overlaps the folder already scanned: {conflict}")

        scope = _ask_scope(root, len(saved))
        if scope == "back":
            continue
        if scope == "one":
            return [root]

        return [root, *_scannable(saved)]


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

    print(f"\nRoot     : {root}")
    print(f"Database : {config.DB_PATH}")
    print(f"Scanned  : {counts['scanned']}")
    print(f"New      : {counts['queued_new']}")
    print(f"Modified : {counts['queued_modified']}")
    print(f"Moved    : {counts['moved']}")
    print(f"Unchanged: {counts['unchanged']}")
    print(f"Deleted  : {counts['deleted']}")
    if counts["delete_blocked"]:
        print(f"Kept     : {counts['delete_blocked']} indexed file(s) not removed (they may be on a disconnected drive)")
    if counts["empty"]:
        print(f"Skipped  : {counts['empty']} empty file(s)")
    if counts["unreadable"]:
        print(f"Unreadable: {counts['unreadable']} file(s) skipped (see error_logs)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    make_output_safe()
    if len(sys.argv) > 1:
        roots = [validate_folder(sys.argv[1])]
    else:
        try:
            roots = _choose_roots()
        except RootOverlapError as exc:
            sys.exit(f"Error: {exc}")
    for root in roots:
        _scan_and_report(root)

    conn = init_db(str(config.DB_PATH))
    try:
        backlog_size = len(get_backlog(conn, config.BACKLOG_MAX_ATTEMPTS))
    finally:
        conn.close()

    if backlog_size == 0:
        return

    if _ask_yes_no(
        f"{backlog_size} file(s) in the backlog are pending indexing. "
        "Run indexing now? [y/N]: "
    ):
        run()
    else:
        print(f"Skipping indexing. Run {config.RUN_COMMAND} and choose \"Scan folders and index images\" later to process the backlog.")


if __name__ == "__main__":
    run_cli(main)