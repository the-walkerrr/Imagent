"""Shared helpers for the single-image test scripts in this folder."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the project root importable when a script is run as ``python tests/<script>.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from vlm.analyzer import analyse_image


def ask_image_path() -> str:
    """Prompt for a local image file path and return it once it is confirmed to exist."""
    value = input("Image path: ").strip().strip("'\"")
    if not value:
        sys.exit("No image given.")
    path = Path(value).expanduser()
    if not path.is_file():
        sys.exit(f"File not found: {path}")
    return str(path)


def run_analysis(image_path: str) -> None:
    """Run the production ``analyse_image`` on one image and print the result and timing."""
    if not config.OLLAMA_HOST or not config.OLLAMA_MODEL:
        sys.exit("Ollama host/model not configured — run setup.py first.")

    start = time.perf_counter()
    payload = analyse_image(image_path)
    elapsed = time.perf_counter() - start

    print(payload.model_dump_json(indent=2))
    print(f"\nModel: {config.OLLAMA_MODEL} @ {config.OLLAMA_HOST}")
    print(f"Analysis time: {elapsed:.2f}s")
