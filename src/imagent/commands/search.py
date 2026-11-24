#!/usr/bin/env python3
"""
imagent/commands/search.py — Imagent natural-language image search.

Given a natural language query, returns the three indexed images whose descriptions are
closest to it, best first.

Pipeline:
  1. Embed the query as typed.
  2. understand_query — an LLM drafts a hypothetical description of the image the query
     asks for (HyDE), and that description is embedded too. If this fails, the search
     continues with the query embedding alone.
  3. Cosine similarity of each embedding to every stored image embedding, read a batch at a
     time (so memory does not grow with the number of images); an image's score is the mean of
     the similarities it has.
  4. The top ``config.UNCERTAIN_TOP_K`` images by score whose file still exists are returned.
     There is no cut-off: the closest images are returned however weak the match.

All activity reads from the user-level database (``config.DB_PATH``).

``main`` is what ``imagent search`` runs. Given a query it answers it and ends. Without one it starts Ollama and loads
the models once, then keeps asking for queries until the user types ``exit`` (or presses Ctrl+C), so later searches start at once.

Each result shows its one-line summary, then its date, camera and size, then the full path of the file. With ``--full`` it shows
the full description instead of the summary, and ``--short`` the summary; with neither, what was chosen in settings (the summary until then). On a terminal, a line that is too long continues under its own indent.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
import time
from datetime import datetime

import numpy as np
import questionary

from imagent import config
from imagent.db.connection import init_db
from imagent.db.operations import (
    count_embeddings,
    get_image_summary,
    get_search_detail,
    iter_embedding_batches,
)
from imagent.pipeline.console import file_link, make_output_safe
from imagent.pipeline.error_logger import log_error
from imagent.pipeline.locking import is_running
from imagent.pipeline.progress import SearchProgress, format_duration
from imagent.vlm.analyzer import embed_search_text, understand_query
from imagent.vlm.client import NotReady, check_ready, load_model, other_models_running


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _unit(embedding: list[float]) -> np.ndarray:
    """An embedding as a unit-length float32 vector: cosine similarity is then a plain dot product."""
    vector = np.array(embedding, dtype=np.float32)
    vector /= np.linalg.norm(vector) + 1e-10
    return vector


def _best_matches(conn, dim: int, query_embeddings: list[list[float]], keep: int) -> tuple[list[int], np.ndarray, int]:
    """The ``keep`` stored images closest to the queries: ``(file_ids, similarities, skipped)``, best first.

    ``similarities`` has a row per image and a column per query (the cosine similarity of that query's embedding to the image's);
    the closest is the one with the highest mean over the columns. The embeddings are read ``config.DB_BATCH_ROWS`` at a time and
    only the best ``keep`` so far are held, so memory use is about ``DB_BATCH_ROWS * dim * 4`` bytes whatever the number of
    images. Rows whose size is not ``dim`` floats are skipped and counted (a defensive check: the models are locked, so none
    are expected).
    """
    queries = np.stack([_unit(embedding) for embedding in query_embeddings])
    best_ids = np.empty(0, dtype=np.int64)
    best_scores = np.empty((0, len(queries)), dtype=np.float32)
    skipped = 0
    for rows in iter_embedding_batches(conn, config.DB_BATCH_ROWS):
        batch = np.empty((len(rows), dim), dtype=np.float32)
        ids = []
        for row in rows:
            blob = row["embedding"]
            if len(blob) != dim * 4:
                skipped += 1
                continue
            batch[len(ids)] = np.frombuffer(blob, dtype="<f4")  # stored little-endian on every platform
            ids.append(row["file_id"])
        if not ids:
            continue
        batch = batch[: len(ids)]
        batch /= np.sqrt(np.einsum("ij,ij->i", batch, batch))[:, None] + 1e-10  # no full-size temporary, as np.linalg.norm would make
        ids_all = np.concatenate([best_ids, np.array(ids, dtype=np.int64)])
        scores_all = np.concatenate([best_scores, batch @ queries.T])
        if len(ids_all) > keep:
            closest = np.argpartition(-scores_all.mean(axis=1), keep - 1)[:keep]
            ids_all, scores_all = ids_all[closest], scores_all[closest]
        best_ids, best_scores = ids_all, scores_all
    order = np.argsort(-best_scores.mean(axis=1), kind="stable")
    return [int(file_id) for file_id in best_ids[order]], best_scores[order], skipped


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def _another_model_is_running(client, settings: dict, progress: SearchProgress) -> bool:
    """True if Ollama is running a model the search does not use; the user is warned (and told if indexing is the cause) and the search should close."""
    others = other_models_running(client, [settings["judge_model"], settings["embedding_model"]])
    if not others:
        return False
    progress.fail()
    progress.say(f"Ollama is running another model ({', '.join(others)}), so loading the search models now would throttle the device. "
                 + ("Indexing is running in the background; wait for it to finish, then search again."
                    if is_running("indexing") else "Close it or wait for it to finish, then search again."))
    return True


def _offer_scan() -> None:
    """With a terminal to answer in, offer to scan and index images now (or, when no folder has been added, to add one), or to exit; the search
    itself ends either way."""
    if not sys.stdin.isatty():
        return
    from imagent.commands import scan  # after_add (which scan uses) imports this module
    from imagent.commands.menu import ADD_FOLDER, STYLE, has_folders
    title, command = ("Scan and index images", "scan") if has_folders() else ADD_FOLDER  # with no folder there is nothing to scan: adding one scans and indexes it
    choice = questionary.select("What would you like to do?", choices=[questionary.Separator(" "), questionary.Choice(title, value=command),
                                questionary.Choice("Exit", value="exit")], qmark="▌", pointer="»", instruction=" ", style=STYLE).ask()
    if choice is None:  # Ctrl+C; questionary already said so
        sys.exit(1)
    if choice == "scan":
        scan.main()
    elif choice == "add":
        scan.add()


def start_session() -> tuple | None:
    """Start Ollama and load the judge and embedding models, shown as one step.

    Returns the client and the locked model settings to pass to ``search``, or None
    (after saying why) if the search cannot run.
    """
    make_output_safe()
    conn = init_db(str(config.DB_PATH))
    print()  # a gap above the live line
    progress = SearchProgress()
    progress.step("Loading the model and starting the server")
    try:
        try:
            client, settings = check_ready(conn, ("judge", "embedding"))
        except NotReady as exc:
            progress.fail()
            progress.say(str(exc))
            return None

        if count_embeddings(conn) == 0:
            progress.fail()
            progress.say("\nNo indexed images found.\n")  # a blank line above and below
            progress.close()
            conn.close()
            _offer_scan()
            return None

        if _another_model_is_running(client, settings, progress):
            return None
        load_model(client, settings["judge_model"])
        load_model(client, settings["embedding_model"], embedding=True)
        return client, settings
    finally:
        progress.close()
        conn.close()


def search(query: str, session: tuple | None = None) -> list[dict]:
    """Return the ``config.UNCERTAIN_TOP_K`` images closest to ``query``, best first.

    Each result dict has: file_id, file_path, short_description, description, the file's properties
    (width, height, file_size, mime_type, modified_at, root_path) and EXIF facts (taken_at, camera_make, camera_model,
    gps_lat, gps_lon; None when absent), score (the mean of the similarities below), query_score (similarity to the
    query as typed) and hyde_score (similarity to the HyDE description, None if that step failed).

    ``session`` is what ``start_session`` returned; without it the session is started here.
    """
    make_output_safe()
    if session is None:
        session = start_session()
        if session is None:
            return []
    client, settings = session
    conn = init_db(str(config.DB_PATH))
    progress = SearchProgress()
    try:
        judge_model, embedding_model = settings["judge_model"], settings["embedding_model"]
        progress.step("Searching")
        try:
            query_embedding = embed_search_text(client, embedding_model, query)
        except Exception as exc:
            log_error(conn, stage="search", exc=exc, metadata={"query": query})
            progress.fail()
            progress.say(f"Could not reach the embedding model ({type(exc).__name__}: {exc}).")
            return []
        if len(query_embedding) != settings["embedding_dim"]:
            progress.fail()
            progress.say(f"The embedding model returned {len(query_embedding)} values, but "
                         f"{settings['embedding_dim']} were measured at setup. The model has changed.")
            return []

        hyde_embedding = None
        progress.step("Analyzing")
        try:
            understanding = understand_query(client, judge_model, query)
            embedded = embed_search_text(client, embedding_model, understanding.hypothetical_description)
            if len(embedded) != settings["embedding_dim"]:
                raise ValueError(f"the embedding model returned {len(embedded)} values, "
                                 f"but {settings['embedding_dim']} were measured at setup")
            hyde_embedding = embedded
        except Exception as exc:
            log_error(conn, stage="query_understanding", exc=exc, metadata={"query": query})
            progress.fail("Could not analyze the query, so the search uses your words as typed.")
            # Query understanding is an enhancement, not a hard dependency —
            # fall back to the query embedding alone.

        progress.step("Matching images")
        queries = [query_embedding] if hyde_embedding is None else [query_embedding, hyde_embedding]
        # Enough of the closest to still find UNCERTAIN_TOP_K whose file exists after skipping up to SEARCH_MAX_MISSING_SKIPPED that do not.
        file_ids, similarities, skipped_dim = _best_matches(conn, settings["embedding_dim"], queries,
                                                            config.UNCERTAIN_TOP_K + config.SEARCH_MAX_MISSING_SKIPPED)
        if skipped_dim:
            progress.say(f"Skipped {skipped_dim} embedding(s) with an unexpected size.")
        if not file_ids:
            return []
        scores = similarities.mean(axis=1)

        results: list[dict] = []
        missing = 0
        for row, file_id in enumerate(file_ids):
            if len(results) >= config.UNCERTAIN_TOP_K or missing >= config.SEARCH_MAX_MISSING_SKIPPED:
                break
            summary = get_image_summary(conn, file_id)
            if summary is None:
                continue
            if not os.path.exists(summary["file_path"]):  # moved or deleted since the last scan, or on a drive that is not connected
                missing += 1
                continue
            results.append({
                "file_id": file_id,
                "file_path": summary["file_path"],
                "short_description": summary["short_description"],
                "description": summary["description"],
                **{key: summary[key] for key in _DETAIL_COLUMNS},
                "score": float(scores[row]),
                "query_score": float(similarities[row, 0]),
                "hyde_score": None if hyde_embedding is None else float(similarities[row, 1]),
            })
        if missing:
            progress.say(f"Skipped {missing} closer match(es) whose file is missing "
                         f"(is a drive disconnected?). Run {config.SCAN_COMMAND} to bring the index up to date.")
        return results
    finally:
        progress.close()
        conn.close()


_DETAIL_COLUMNS = ("width", "height", "file_size", "mime_type", "taken_at", "modified_at",
                   "camera_make", "camera_model", "gps_lat", "gps_lon", "root_path")


def _file_size(size: int) -> str:
    amount = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if amount < 1024 or unit == "GB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024


def _day(moment: datetime) -> str:
    """A date as people say it, e.g. ``9 May, 2024`` (the day is not padded, which strftime cannot do on every system)."""
    return f"{moment.day} {moment:%b}, {moment.year}"


def _when(result: dict) -> str | None:
    """When the picture was taken (its EXIF time), or else when the file was last changed."""
    if result["taken_at"]:
        try:
            taken = datetime.fromisoformat(result["taken_at"])
            return f"{taken:%I:%M %p} {_day(taken)}"
        except ValueError:
            return result["taken_at"]
    if result["modified_at"]:
        try:
            return f"modified {_day(datetime.fromisoformat(result['modified_at'][:10]))}"
        except ValueError:
            return f"modified {result['modified_at'][:10]}"
    return None


def _facts(result: dict) -> str:
    """The date, camera, size in pixels and size on disk of an image on one line, leaving out what is not known."""
    make, model = result["camera_make"], result["camera_model"]
    if make and model and model.lower().startswith(make.lower()):  # many cameras already put the make in the model ("iQOO", "iQOO Neo9 Pro")
        make = None
    camera = " ".join(part for part in (make, model) if part)
    pixels = f"{result['width']}\u00d7{result['height']}" if result["width"] and result["height"] else None
    size = _file_size(result["file_size"]) if result["file_size"] is not None else None
    return " \u00b7 ".join(part for part in (_when(result), camera, pixels, size) if part)


def _indented(text: str, break_long_words: bool = False) -> str:
    """``text`` under the file name (three spaces in). On a terminal a line that is too long continues with the same indent,
    instead of starting again at the left edge; piped output is not wrapped. A word longer than the line (a path) is
    only cut when ``break_long_words``."""
    lines = text.splitlines()
    if not sys.stdout.isatty():
        return "\n".join(f"   {line}".rstrip() for line in lines)
    width = max(shutil.get_terminal_size().columns - 1, 20)  # the last column is left empty so a line never wraps by itself
    return "\n".join(textwrap.fill(line, width, initial_indent="   ", subsequent_indent="   ", break_long_words=break_long_words,
                                    break_on_hyphens=False) if line.strip() else "" for line in lines)


def _description(result: dict, full: bool) -> str:
    """The one-line summary, or with ``full`` the whole description; empty when there is none."""
    return _indented((result["description"] if full else result["short_description"]) or "")


def _show(query: str, results: list[dict], full: bool = False, seconds: float | None = None) -> None:
    """Print the results; ``seconds`` is how long the search took, shown in the heading when given."""
    took = f" in {format_duration(seconds, decimals=1)}" if seconds is not None else ""
    if not results:
        print(f"\nNo results for: {query!r}{took}\n")
        return

    bold = sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ
    print(f"\nTop {len(results)} result(s) for: " + (f"\033[1m{query!r}\033[0m" if bold else f"{query!r}") + f"{took}\n")
    for position, result in enumerate(results, start=1):
        name = file_link(os.path.basename(result["file_path"]), result["file_path"])
        name = f"\033[1m{name}\033[0m" if bold else name
        lines = [f"{position}. {name}", _description(result, full), _indented(_facts(result)),
                 _indented(result["file_path"], break_long_words=True)]
        print("\n\n".join(line for line in lines if line))  # a blank line between the lines of a result, two between results
        print("\n")


def _answer(query: str, session: tuple, full: bool) -> None:
    started = time.monotonic()
    results = search(query, session)
    _show(query, results, full, time.monotonic() - started)


def _saved_detail_is_full() -> bool:
    """True when settings (or, before any choice, ``config.SEARCH_DETAIL_DEFAULT``) say a search shows the full description."""
    conn = init_db(str(config.DB_PATH))
    try:
        return (get_search_detail(conn) or config.SEARCH_DETAIL_DEFAULT) == "full"
    finally:
        conn.close()


DETAIL_HINT = "Tip: add --full or --short to change how much each result shows. You can also change it in Settings."


def _show_detail_hint() -> None:
    """The first thing a search says, in grey, on a terminal only (it is not part of the results): how to change what each result shows."""
    if not sys.stdout.isatty():
        return
    grey = os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ
    print("\n" + (f"\033[38;5;244m{DETAIL_HINT}\033[0m" if grey else DETAIL_HINT))  # a gap above, so it never touches the command typed


def main(query: str = "", full: bool | None = None) -> None:
    """Answer ``query`` and end; with no query, keep asking for queries until the user types ``exit``.

    ``full`` True shows every result's full description, False its summary; None (the default) uses what was chosen in settings. A flag
    given on the command line always wins over the setting."""
    make_output_safe()
    _show_detail_hint()
    if full is None:
        full = _saved_detail_is_full()
    query = query.strip()
    session = start_session()
    if session is None:
        return
    if query:
        _answer(query, session, full)
        return
    colour = sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ
    title = "▌Enter your search query (type exit to quit):"
    title = f"\033[1m{title}\033[0m" if colour else title
    above = ""  # the live line that loaded the models left the gap above the first step
    while True:
        print(f"{above}{title}\n")  # a step: a gap above and below it
        above = "\n"
        try:
            query = input().strip()
        except EOFError:  # Ctrl+D, or no terminal to type in
            return
        if query.lower() in ("exit", "quit"):
            return
        if not query:
            print('Type what you are looking for, for example: red car at night')
            continue
        print()  # a gap between the query and the step shown while it is answered
        _answer(query, session, full)
