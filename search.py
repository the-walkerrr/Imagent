#!/usr/bin/env python3
"""
search.py — Imagent natural-language image search.

Given a natural language query, returns up to three matching indexed images,
best first, or an empty list if nothing genuinely matches — never an
unfiltered list.

Pipeline:
  1. understand_query — an LLM rewrites the query (typo/abbreviation fixes),
     extracts a checklist of required details, and drafts a hypothetical
     image description (HyDE) to embed alongside the raw query. Falls back
     to the raw query alone on any failure — this stage is an enhancement,
     not a hard dependency.
  2. Rank candidates by four signals — semantic similarity to the raw
     query, semantic similarity to the HyDE description, FTS5 keyword
     match (with stemming), and partial-word match against visible text,
     brands and landmarks — fused via Reciprocal Rank Fusion (RRF).
  3. A cheap numeric gate (assess_confidence) checks the raw query's cosine
     similarity against the rest of its own candidate pool. If the best
     score is below an absolute floor, nothing in the library is even
     plausibly related, so the search returns "not found" immediately
     without spending an LLM call.
  4. Otherwise, an LLM judge reads each top candidate's stored record (the
     same text that was embedded and indexed) against the query and
     checklist, and decides — strictly, defaulting to "no match" when
     uncertain — whether it is genuinely the requested image. Every
     accepted candidate is returned, highest judge confidence first; if
     the judge rejects all of them, the result is "not found".

All activity reads from the user-level database at ``~/.imagent/imagent.db``
(overridable via the ``IMAGENT_HOME`` environment variable).
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys

import numpy as np

import config
from db.connection import init_db
from db.operations import (
    fts_search,
    get_all_embeddings,
    get_embeddings_model_counts,
    get_image_summary,
    partial_fts_search,
)
from pipeline.error_logger import log_error
from pipeline.serialisation import deserialize_embedding
from text_processing.formatters import build_search_text
from VLMResponseFormat import ImagePayload
from vlm.analyzer import embed_search_text, judge_candidate, understand_query

RRF_K = 60  # standard Reciprocal Rank Fusion damping constant
CANDIDATE_POOL = 50  # candidates each ranking method contributes before fusion
_FTS_TOKEN_RE = re.compile(r"[0-9A-Za-z]+")
_PARTIAL_MIN_TOKEN_LEN = 4  # shortest query word used for partial-word (trigram) matching


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _semantic_candidates(conn, query_embedding: list[float]) -> tuple[list[dict], int]:
    """Rank stored embeddings by cosine similarity to a query embedding.

    Only embeddings matching the currently configured embedding model are
    considered (see the model-mismatch reporting in ``search()``). Returns
    the top ``CANDIDATE_POOL`` matches plus a count of embeddings skipped
    for having an unexpected dimensionality (a defensive check — shouldn't
    happen given the model filter, but guards against silently-wrong math).
    """
    rows = get_all_embeddings(conn, model=config.OLLAMA_EMBEDDING_MODEL)
    if not rows:
        return [], 0

    query_vec = np.array(query_embedding, dtype=np.float32)
    expected_dim = query_vec.shape[0]

    file_ids: list[str] = []
    vectors: list[list[float]] = []
    skipped = 0
    for row in rows:
        vec = deserialize_embedding(row["embedding"])
        if len(vec) != expected_dim:
            skipped += 1
            continue
        file_ids.append(row["file_id"])
        vectors.append(vec)

    if not vectors:
        return [], skipped

    matrix = np.array(vectors, dtype=np.float32)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query_vec)
    # Full cosine formula (not a bare dot product): stored bge-m3 embeddings
    # are empirically unit-normalised, but that isn't guaranteed for every
    # embedding model, so normalise defensively rather than assume it.
    similarities = (matrix @ query_vec) / (matrix_norms * query_norm + 1e-10)

    order = np.argsort(-similarities)[:CANDIDATE_POOL]
    candidates = [{"file_id": file_ids[i], "score": float(similarities[i])} for i in order]
    return candidates, skipped


def _fts_match_expression(query: str, min_length: int = 3) -> str | None:
    """Build a safe FTS5 MATCH expression from a natural-language query.

    Bare multi-token strings are an implicit AND in FTS5, so a natural
    query like "a dog on the beach" matches nothing (every token,
    including "a"/"on"/"the", would need to appear in the same document).
    Tokenizing and OR-ing double-quoted terms fixes that; quoting also
    neutralizes FTS5 operators (AND/OR/NEAR/-/') as literal text, which
    doubles as sanitization against MATCH syntax errors. Tokens shorter than
    ``min_length`` are dropped as noise. Returns None if no usable tokens remain.
    """
    tokens = []
    seen = set()
    for match in _FTS_TOKEN_RE.finditer(query.lower()):
        token = match.group(0)
        if len(token) < min_length or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= 12:
            break
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


def _keyword_candidates(conn, query: str, partial: bool = False) -> list[dict]:
    """Rank images by full-text keyword match against the search query.

    With ``partial=True`` the match runs against the trigram index of visible
    text, brands and landmarks instead, so a fragment of a word is enough.
    Only tokens of ``_PARTIAL_MIN_TOKEN_LEN`` characters or more are used there,
    because short fragments match almost any text.

    Drops zero-IDF matches (bm25 ~ -0.0000, i.e. a term present in nearly
    every document) — those carry no discriminative signal and shouldn't
    earn rank-based fusion credit.
    """
    expression = _fts_match_expression(query, min_length=_PARTIAL_MIN_TOKEN_LEN if partial else 3)
    if expression is None:
        return []
    try:
        search_fn = partial_fts_search if partial else fts_search
        rows = search_fn(conn, expression, limit=CANDIDATE_POOL)
    except sqlite3.OperationalError as exc:
        log_error(conn, stage="search", exc=exc, metadata={"query": query, "expression": expression})
        return []
    return [row for row in rows if row["rank"] <= -1e-6]


def _fuse_rrf(*ranked_lists: list[dict]) -> list[dict]:
    """Combine any number of ranked candidate lists via Reciprocal Rank Fusion.

    Each list contributes 1 / (RRF_K + rank) to a file's fused score, so
    images that rank well on any signal (or several) rise to the top —
    rather than requiring one method to be correct on its own.
    """
    fused: dict[str, dict] = {}

    for source_name, ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            entry = fused.setdefault(
                item["file_id"],
                {"file_id": item["file_id"], "rrf_score": 0.0, "sources": set(), "cosine": None},
            )
            entry["rrf_score"] += 1.0 / (RRF_K + rank)
            entry["sources"].add(source_name)
            if source_name == "semantic_raw":
                entry["cosine"] = item["score"]

    return sorted(fused.values(), key=lambda e: e["rrf_score"], reverse=True)


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------

def assess_confidence(semantic_raw: list[dict]) -> dict:
    """Assess how confident the raw-query cosine ranking is in its own top hit.

    Compares the top candidate's cosine score against the background of
    the next ``config.Z_POOL_SIZE`` candidates (not the whole corpus —
    whole-corpus stats drift toward the background-similarity floor as the
    library grows, which would silently rot these numbers over time).

    Returns a diagnostics dict. Only ``decision == "NOT_FOUND"`` changes
    control flow (skips the LLM judge entirely, since nothing is even
    plausibly related); ``CONFIDENT`` vs ``UNCERTAIN`` is informational —
    the judge runs either way.
    """
    if not semantic_raw:
        return {"decision": "NOT_FOUND", "cos1": None, "margin": None, "z": None}

    cos1 = semantic_raw[0]["score"]
    if cos1 < config.COSINE_ABS_FLOOR:
        return {"decision": "NOT_FOUND", "cos1": cos1, "margin": None, "z": None}

    margin = cos1 - semantic_raw[1]["score"] if len(semantic_raw) > 1 else None

    pool = [c["score"] for c in semantic_raw[1 : 1 + config.Z_POOL_SIZE]]
    z = None
    if len(pool) >= config.Z_MIN_POOL:
        pool_mean = float(np.mean(pool))
        pool_std = max(float(np.std(pool)), config.Z_SIGMA_FLOOR)
        z = (cos1 - pool_mean) / pool_std

    decision = "UNCERTAIN"
    if margin is not None and margin >= config.MARGIN_CONFIDENT and z is not None and z >= config.Z_CONFIDENT:
        decision = "CONFIDENT"

    return {"decision": decision, "cos1": cos1, "margin": margin, "z": z}


# ---------------------------------------------------------------------------
# Judge input
# ---------------------------------------------------------------------------

def _build_document(summary: dict) -> str | None:
    """Reconstruct the search text for a stored image, for the judge to read.

    Rebuilds the same document ``build_search_text`` produced at indexing
    time (from the stored VLM metadata) rather than storing it twice.
    """
    try:
        payload_data = json.loads(summary["vlm_metadata_json"])
        payload_data["short_description"] = summary["short_description"]
        payload_data["description"] = summary["description"]
        payload = ImagePayload.model_validate(payload_data)
        return build_search_text(payload)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def search(query: str) -> list[dict]:
    """Return the images the judge accepts for ``query``, best first; empty if none.

    Combines semantic similarity (raw query + HyDE description), keyword
    relevance (full-text search), a numeric confidence gate, and an LLM
    relevance judge that reviews up to ``config.UNCERTAIN_TOP_K`` candidates.
    Each result dict has: file_id, file_path, short_description, is_match,
    confidence, reasoning, matched_details, sources, plus the gate
    diagnostics (cos1, margin, z, gate_decision).
    """
    conn = init_db(str(config.DB_PATH))
    try:
        model_counts = get_embeddings_model_counts(conn)
        if not model_counts:
            print("[imagent] No indexed images found. Run scan.py and analyze_and_index.py first.")
            return []

        other_models = [row for row in model_counts if row["model"] != config.OLLAMA_EMBEDDING_MODEL]
        if other_models:
            excluded = sum(row["n"] for row in other_models)
            names = ", ".join(f"{row['model']} ({row['n']})" for row in other_models)
            print(
                f"[imagent] Excluding {excluded} image(s) indexed with a different embedding "
                f"model ({names}) from semantic ranking — re-run analyze_and_index.py to "
                f"bring them up to date with {config.OLLAMA_EMBEDDING_MODEL}."
            )

        try:
            query_embedding = embed_search_text(query)
        except Exception as exc:
            log_error(conn, stage="search", exc=exc, metadata={"query": query})
            print(f"[imagent] Could not reach the embedding model ({type(exc).__name__}: {exc}).")
            return []

        required_details: list[str] = []
        keyword_query = query
        hyde_embedding = None
        try:
            understanding = understand_query(query)
            required_details = understanding.required_details
            keyword_query = understanding.normalized_query or query
            hyde_embedding = embed_search_text(understanding.hypothetical_description)
        except Exception as exc:
            log_error(conn, stage="query_understanding", exc=exc, metadata={"query": query})
            # Query understanding is an enhancement, not a hard dependency —
            # fall back to the raw query alone.

        semantic_raw, skipped_dim = _semantic_candidates(conn, query_embedding)
        if skipped_dim:
            print(f"[imagent] Skipped {skipped_dim} embedding(s) with an unexpected dimensionality.")

        gate = assess_confidence(semantic_raw)
        if gate["decision"] == "NOT_FOUND":
            return []

        ranked_lists = [("semantic_raw", semantic_raw)]
        if hyde_embedding is not None:
            semantic_hyde, _ = _semantic_candidates(conn, hyde_embedding)
            ranked_lists.append(("semantic_hyde", semantic_hyde))
        ranked_lists.append(("keyword", _keyword_candidates(conn, keyword_query)))
        ranked_lists.append(("partial_text", _keyword_candidates(conn, keyword_query, partial=True)))

        fused = _fuse_rrf(*ranked_lists)

        accepted: list[dict] = []
        for entry in fused[: config.UNCERTAIN_TOP_K]:
            summary = get_image_summary(conn, entry["file_id"])
            if summary is None:
                continue
            document = _build_document(summary)
            if document is None:
                continue
            try:
                verdict = judge_candidate(query, document, required_details)
            except Exception as exc:
                log_error(
                    conn,
                    stage="relevance_judge",
                    exc=exc,
                    file_id=entry["file_id"],
                    metadata={"query": query},
                )
                continue

            if verdict.is_match and verdict.confidence >= config.JUDGE_CONFIDENCE_MIN:
                accepted.append({
                    "file_id": entry["file_id"],
                    "file_path": summary["file_path"],
                    "short_description": summary["short_description"],
                    "is_match": True,
                    "confidence": verdict.confidence,
                    "reasoning": verdict.reasoning,
                    "matched_details": verdict.matched_details,
                    "sources": sorted(entry["sources"]),
                    "gate_decision": gate["decision"],
                    "cos1": gate["cos1"],
                    "margin": gate["margin"],
                    "z": gate["z"],
                })

        # Stable sort: candidates with equal confidence keep their fused ranking order.
        accepted.sort(key=lambda result: result["confidence"], reverse=True)
        return accepted
    finally:
        conn.close()


def main() -> None:
    raw_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your search query: ").strip()
    results = search(raw_query)

    if not results:
        print(f"\n[imagent] No matching image found for: {raw_query!r}\n")
        return

    print(f"\n[imagent] {len(results)} match(es) for: {raw_query!r}\n")
    for position, result in enumerate(results, start=1):
        print(f"[imagent] {position}. {result['file_path']}")
        print(f"[imagent]    {result['short_description']}")
        print(f"[imagent]    confidence={result['confidence']:.2f}  sources={'+'.join(result['sources'])}")
        print(f"[imagent]    reasoning: {result['reasoning']}")
        print()


if __name__ == "__main__":
    main()
