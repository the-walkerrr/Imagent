#!/usr/bin/env python3
"""
menu/menu.py — Imagent: the menu that run.sh and run.bat open. Pick an entry with the arrow keys and Enter.

It runs the chosen command and ends with that command's exit code. "Show the images that could not be indexed" is only
offered when there are some (images are retried by themselves; this only tells the user which ones could not be indexed and why).
"Re-embed images" is only offered when the embedding
model is no longer the one the stored vectors were made with (it was updated or deleted in Ollama); otherwise there is
nothing to re-embed. If that cannot be found out (Ollama cannot be reached, setup is not finished), the entry is left
out: the command that is chosen reports the problem itself.

Uninstall is not run from here: it deletes .venv, which this program runs from. The menu ends with UNINSTALL_EXIT_CODE
and the launcher starts the uninstaller.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Make the project root importable when this file is run directly (``python menu/menu.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import questionary

    import config
    from db.connection import init_db
    from db.operations import get_exhausted_backlog
    from pipeline.console import make_output_safe, run_cli
    from vlm.client import check_ready
except ModuleNotFoundError as exc:
    sys.exit(f"Imagent is not set up yet (the Python package '{exc.name}' is missing). "
             f"Run: {'install.bat' if sys.platform == 'win32' else 'sh install.sh'}")

PROJECT_DIR = Path(__file__).resolve().parent.parent
UNINSTALL_EXIT_CODE = 75  # tells run.sh / run.bat to start the uninstaller (they compare with this number)
UNINSTALL = "uninstall"

SEARCH = ("Search your images", "search/search.py")
SCAN = ("Scan folders and index images", "scan_and_index/scan_and_index.py")
SHOW_FAILED_SCRIPT = "show_failed/show_failed.py"
REEMBED = ("Re-embed images (the embedding model changed)", "reembed/reembed.py")

STYLE = questionary.Style([
    ("qmark", "fg:ansidefault bold"),
    ("question", "fg:ansidefault bold"),
    ("pointer", "fg:ansidefault bold"),
    ("highlighted", "fg:ansidefault bold underline"),
    ("answer", "fg:ansidefault bold"),
    ("instruction", "fg:ansibrightblack"),
])


def embedding_model_changed() -> bool:
    """True if the embedding model the vectors were made with is not installed in Ollama, or is another version of it now."""
    try:
        conn = init_db(str(config.DB_PATH))
        try:
            client, settings = check_ready(conn, ())  # starts Ollama if it is down; it is stopped again when the menu ends
        finally:
            conn.close()
        digest = {model.model: model.digest for model in client.list().models}.get(settings["embedding_model"])
    except Exception:  # best effort: the menu must always appear, and the chosen command reports a real problem
        return False
    stored = settings["embedding_digest"]
    return digest is None or bool(stored) and digest != stored


def failed_count() -> int:
    """How many images could not be indexed after every try; 0 if that cannot be found out (the menu must always appear)."""
    try:
        conn = init_db(str(config.DB_PATH))
        try:
            return len(get_exhausted_backlog(conn, config.BACKLOG_MAX_ATTEMPTS))
        finally:
            conn.close()
    except Exception:
        return 0


def choices(with_reembed: bool, failed: int = 0) -> list:
    shown = [(f"Show the {failed} image(s) that could not be indexed", SHOW_FAILED_SCRIPT)] if failed else []
    entries = [SEARCH, SCAN] + shown + ([REEMBED] if with_reembed else [])
    return [questionary.Choice(title, value=script) for title, script in entries] + [
        questionary.Choice([("fg:ansired", "Uninstall Imagent")], value=UNINSTALL)]


def main() -> None:
    make_output_safe()
    print("Checking the models…")
    choice = questionary.select("What would you like to do?", choices=choices(embedding_model_changed(), failed_count()), qmark="?",
                                pointer="»", style=STYLE).ask()
    if choice is None:  # Ctrl+C; questionary already said so
        sys.exit(1)
    if choice == UNINSTALL:
        sys.exit(UNINSTALL_EXIT_CODE)
    process = subprocess.Popen([sys.executable, str(PROJECT_DIR / choice)])
    while True:
        try:
            sys.exit(process.wait())
        except KeyboardInterrupt:
            continue  # the command got the same Ctrl+C and is ending by itself: wait for it, do not cut it short


if __name__ == "__main__":
    run_cli(main)
