#!/usr/bin/env python3
"""
retry_failed/retry_failed.py — Imagent: queue again the images that were set aside, then offer to index them.

An image is set aside (see ``backlog.status``) when every step for its kind of failure was tried; it is queued again by
itself when the file changes, or, for a model's replies, when the model or the prompt changes. This does it on request:
for every image set aside, or only for one kind of failure (``python retry_failed/retry_failed.py unreadable``).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Make the project root importable when this file is run directly (``python retry_failed/retry_failed.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import config
    from db.connection import init_db
    from db.operations import list_parked, requeue_parked
    from pipeline.console import make_output_safe, run_cli
    from scan_and_index.scan_and_index import _ask_yes_no, run
except ModuleNotFoundError as exc:
    sys.exit(f"Imagent is not set up yet (the Python package '{exc.name}' is missing). "
             f"Run: {'install.bat' if sys.platform == 'win32' else 'sh install.sh'}")


def _choose(kinds: Counter) -> str | None:
    """Ask which kind to retry. Returns the kind, or None for all of them."""
    names = sorted(kinds)
    print("\n  0) All of them")
    for number, kind in enumerate(names, start=1):
        print(f"  {number}) {kind} ({kinds[kind]})")
    while True:
        try:
            answer = input("Which do you want to try again? [0]: ").strip() or "0"
        except EOFError:
            sys.exit("\n[imagent] An answer is needed here but there is no terminal to type it in. "
                     "Give the kind on the command line, or run this from a terminal.")
        if answer.isdigit() and int(answer) <= len(names):
            return None if answer == "0" else names[int(answer) - 1]
        print(f"Please type a number from 0 to {len(names)}.")


def main() -> None:
    make_output_safe()
    conn = init_db(str(config.DB_PATH))
    try:
        parked = list_parked(conn)
        if not parked:
            print("[imagent] No images are set aside. There is nothing to try again.")
            return
        kinds = Counter(item["park_kind"] for item in parked)
        print(f"[imagent] {len(parked)} image(s) are set aside:")
        for kind in sorted(kinds):
            example = next(item for item in parked if item["park_kind"] == kind)
            print(f"\n  {kind}: {kinds[kind]} image(s). For example, {Path(example['file_path']).name}:\n    {example['last_error']}")

        kind = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "all" else None
        if len(sys.argv) > 1 and kind is not None and kind not in kinds:
            sys.exit(f"[imagent] No image is set aside for '{kind}'. The kinds are: {', '.join(sorted(kinds))}.")
        if len(sys.argv) <= 1:
            kind = _choose(kinds) if len(kinds) > 1 else None
        with conn:
            queued = requeue_parked(conn, kind)
        print(f"\n[imagent] {queued} image(s) queued again.")
    finally:
        conn.close()

    if queued and _ask_yes_no("[imagent] Run indexing now? [y/N]: "):
        run()


if __name__ == "__main__":
    run_cli(main)
