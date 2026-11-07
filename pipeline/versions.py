"""
pipeline/versions.py — The versions of everything that produces an indexed record.

Stored with each record (see db/schema.sql), they show which records were made with something older than what is
installed now, and decide when a file set aside for its model's replies is worth another try.
"""

from __future__ import annotations

import hashlib
import json
from importlib import metadata

import config
from text_processing.formatters import SEARCH_TEXT_VERSION
from vlm.client import server_version

# The versions that decide whether a file parked for its model's replies could do better now.
MODEL_KEYS = ("vlm_digest", "prompt_hash", "schema_hash")


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_versions(client, settings: dict) -> dict:
    """The versions in use right now: the models' digests as Ollama reports them, fingerprints of the prompt and
    the reply schema, and the versions of the search-text format, the OCR library, Ollama and Imagent."""
    try:
        digests = {model.model: model.digest for model in client.list().models}
    except Exception:
        digests = {}
    ollama_version = server_version(config.OLLAMA_HOST) if config.OLLAMA_HOST else None
    return {
        "vlm_digest": digests.get(settings["vlm_model"]),
        "embedding_digest": digests.get(settings["embedding_model"]),
        "prompt_hash": _fingerprint(config.ANALYSIS_PROMPT),
        "schema_hash": _fingerprint(json.dumps(config.ANALYSIS_SCHEMA, sort_keys=True)),
        "search_text_version": SEARCH_TEXT_VERSION,
        "ocr_version": _package_version("rapidocr"),
        "ollama_version": ".".join(map(str, ollama_version)) if ollama_version else None,
        "app_version": config.APP_VERSION,
    }


def model_versions(versions: dict) -> dict | None:
    """The part of ``versions`` that decides whether a parked file is worth another try; None if any of it is not known."""
    picked = {key: versions.get(key) for key in MODEL_KEYS}
    return picked if all(picked.values()) else None
