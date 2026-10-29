#!/usr/bin/env python3
"""
search.py — Imagent natural-language image search.

Given a natural language query, returns the three indexed images whose descriptions are
closest to it, best first.

Pipeline:
  1. Embed the query as typed.
  2. understand_query — an LLM drafts a hypothetical description of the image the query
     asks for (HyDE), and that description is embedded too. If this fails, the search
     continues with the query embedding alone.
  3. Cosine similarity of each embedding to every stored image embedding; an image's
     score is the mean of the similarities it has.
  4. The top ``config.UNCERTAIN_TOP_K`` images by score whose file still exists are returned.
     There is no cut-off: the closest images are returned however weak the match.

All activity reads from the user-level database (``config.DB_PATH``).

Run from the command line it starts Ollama and loads the models once, then keeps asking for
queries until the user types ``exit`` (or presses Ctrl+C), so later searches start at once.
"""

from __future__ import annotations

import os
import sys

import numpy as np

import config
from db.connection import init_db
from db.operations import (
    count_embeddings,
    get_image_summary,
    iter_embeddings,
)
from pipeline.console import make_output_safe, run_cli
from pipeline.error_logger import log_error
from pipeline.locking import is_running
from pipeline.progress import SearchProgress
from vlm.analyzer import embed_search_text, understand_query
from vlm.client import NotReady, check_ready, load_model, other_models_running


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _load_embeddings(conn, dim: int) -> tuple[list[int], np.ndarray, int]:
    """Load every stored embedding into one unit-length float32 matrix, one row per image.

    Decoded straight into a preallocated array (never into Python lists), so memory use is
    about ``images * dim * 4`` bytes whatever the machine. Rows whose size is not ``dim``
    floats are skipped and counted (a defensive check: the models are locked, so none are expected).
    """
    matrix = np.empty((count_embeddings(conn), dim), dtype=np.float32)
    file_ids: list[int] = []
    skipped = 0
    for row in iter_embeddings(conn):
        blob = row["embedding"]
        if len(blob) != dim * 4 or len(file_ids) >= len(matrix):
            skipped += 1
            continue
        matrix[len(file_ids)] = np.frombuffer(blob, dtype="<f4")  # stored little-endian on every platform
        file_ids.append(row["file_id"])
    matrix = matrix[: len(file_ids)]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= norms + 1e-10  # cosine similarity is then a plain dot product
    return file_ids, matrix, skipped


def _similarities(matrix: np.ndarray, query_embedding: list[float]) -> np.ndarray:
    """Cosine similarity of a query embedding to every loaded embedding, in the matrix's row order."""
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_vec /= np.linalg.norm(query_vec) + 1e-10
    return matrix @ query_vec


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


def start_session() -> tuple | None:
    """Start Ollama and load the judge and embedding models, shown as one step.

    Returns the client and the locked model settings to pass to ``search``, or None
    (after saying why) if the search cannot run.
    """
    make_output_safe()
    conn = init_db(str(config.DB_PATH))
    progress = SearchProgress()
    progress.step("Loading the model and starting the server")
    try:
        try:
            client, settings = check_ready(conn, ("judge", "embedding"),
                                           say=lambda message: progress.say(message.removeprefix("[imagent] ")))
        except NotReady as exc:
            progress.fail()
            progress.say(str(exc))
            return None

        if count_embeddings(conn) == 0:
            progress.fail()
            progress.say("No indexed images found. Run scan_and_index.py first.")
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

    Each result dict has: file_id, file_path, short_description, description, score (the mean of the
    similarities below), query_score (similarity to the query as typed) and hyde_score
    (similarity to the HyDE description, None if that step failed).

    ``session`` is what ``start_session`` returned; without it the session is started here.
    """
    make_output_safe()
    if session is None:
        session = start_session()
        if session is None:
            return []
    client, settings = session
    conn = init_db(str(config.DB_PATH))
    progress = SearchProgress(query)
    try:
        judge_model, embedding_model = settings["judge_model"], settings["embedding_model"]
        progress.step("Embedding query")
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
        progress.step("Understanding query")
        try:
            understanding = understand_query(client, judge_model, query)
            progress.step("Embedding HyDE description")
            embedded = embed_search_text(client, embedding_model, understanding.hypothetical_description)
            if len(embedded) != settings["embedding_dim"]:
                raise ValueError(f"the embedding model returned {len(embedded)} values, "
                                 f"but {settings['embedding_dim']} were measured at setup")
            hyde_embedding = embedded
        except Exception as exc:
            log_error(conn, stage="query_understanding", exc=exc, metadata={"query": query})
            progress.fail("skipped")
            # Query understanding is an enhancement, not a hard dependency —
            # fall back to the query embedding alone.

        progress.step("Comparing images")
        file_ids, matrix, skipped_dim = _load_embeddings(conn, settings["embedding_dim"])
        if skipped_dim:
            progress.say(f"Skipped {skipped_dim} embedding(s) with an unexpected size.")
        if not file_ids:
            return []

        query_scores = _similarities(matrix, query_embedding)
        hyde_scores = _similarities(matrix, hyde_embedding) if hyde_embedding is not None else None
        scores = query_scores if hyde_scores is None else (query_scores + hyde_scores) / 2

        results: list[dict] = []
        missing = 0
        for row in np.argsort(-scores):
            if len(results) >= config.UNCERTAIN_TOP_K or missing >= config.SEARCH_MAX_MISSING_SKIPPED:
                break
            summary = get_image_summary(conn, file_ids[row])
            if summary is None:
                continue
            if not os.path.exists(summary["file_path"]):  # moved or deleted since the last scan, or on a drive that is not connected
                missing += 1
                continue
            results.append({
                "file_id": file_ids[row],
                "file_path": summary["file_path"],
                "short_description": summary["short_description"],
                "description": summary["description"],
                "score": float(scores[row]),
                "query_score": float(query_scores[row]),
                "hyde_score": None if hyde_scores is None else float(hyde_scores[row]),
            })
        if missing:
            progress.say(f"Skipped {missing} closer match(es) whose file is missing "
                         "(is a drive disconnected?). Run scan_and_index.py to bring the index up to date.")
        return results
    finally:
        progress.close()
        conn.close()


def _show(query: str, results: list[dict]) -> None:
    if not results:
        print(f"\nNo results for: {query!r}\n")
        return

    print(f"\nTop {len(results)} result(s) for: {query!r}\n")
    bold = sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ
    for position, result in enumerate(results, start=1):
        name = f"\033[1m{result['file_path']}\033[0m" if bold else result["file_path"]
        print(f"{position}. {name}")
        print(f"   {result['description']}")
        print(f"   score={result['score']:.3f}")
        print()


def main() -> None:
    make_output_safe()
    query = " ".join(sys.argv[1:]).strip()
    session = start_session()
    if session is None:
        return
    while True:
        if not query:
            try:
                query = input("Enter your search query (type exit to quit): ").strip()
            except EOFError:  # Ctrl+D, or no terminal to type in
                return
            if query.lower() in ("exit", "quit"):
                return
            if not query:
                print('Type what you are looking for, for example: red car at night')
                continue
        _show(query, search(query, session))
        query = ""


if __name__ == "__main__":
    run_cli(main, cancelled="Search cancelled.")
