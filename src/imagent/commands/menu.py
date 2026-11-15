#!/usr/bin/env python3
"""
imagent/commands/menu.py — Imagent: the menu that run.sh and run.bat open. Pick an entry with the arrow keys and Enter.

It runs the chosen command and ends with that command's exit code. "Show the images that could not be indexed" is only
offered when there are some (images are retried by themselves; this only tells the user which ones could not be indexed and why).

Uninstall is not run from here: it deletes .venv, which this program runs from. The menu ends with UNINSTALL_EXIT_CODE
and the launcher starts the uninstaller.
"""

from __future__ import annotations

import subprocess
import sys

try:
    import questionary

    from imagent import config
    from imagent.db.connection import init_db
    from imagent.db.operations import get_exhausted_backlog
    from imagent.pipeline.console import make_output_safe, run_cli
except ModuleNotFoundError as exc:
    sys.exit(f"Imagent is not set up yet (the Python package '{exc.name}' is missing). "
             f"Run: {'install.bat' if sys.platform == 'win32' else 'sh install.sh'}")

UNINSTALL_EXIT_CODE = 75  # tells run.sh / run.bat to start the uninstaller (they compare with this number)
UNINSTALL = "uninstall"

SEARCH = ("Search your images", "imagent.commands.search")
SCAN = ("Scan folders and index images", "imagent.commands.scan")
FAILED_MODULE = "imagent.commands.failed"

STYLE = questionary.Style([
    ("qmark", "fg:ansidefault bold"),
    ("question", "fg:ansidefault bold"),
    ("pointer", "fg:ansidefault bold"),
    ("highlighted", "fg:ansidefault bold underline"),
    ("answer", "fg:ansidefault bold"),
    ("instruction", "fg:ansibrightblack"),
])


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


def choices(failed: int = 0) -> list:
    shown = [(f"Show the {failed} image(s) that could not be indexed", FAILED_MODULE)] if failed else []
    entries = [SEARCH, SCAN] + shown
    return [questionary.Choice(title, value=script) for title, script in entries] + [
        questionary.Choice([("fg:ansired", "Uninstall Imagent")], value=UNINSTALL)]


def main() -> None:
    make_output_safe()
    choice = questionary.select("What would you like to do?", choices=choices(failed_count()), qmark="?",
                                pointer="»", style=STYLE).ask()
    if choice is None:  # Ctrl+C; questionary already said so
        sys.exit(1)
    if choice == UNINSTALL:
        sys.exit(UNINSTALL_EXIT_CODE)
    process = subprocess.Popen([sys.executable, "-m", choice])
    while True:
        try:
            sys.exit(process.wait())
        except KeyboardInterrupt:
            continue  # the command got the same Ctrl+C and is ending by itself: wait for it, do not cut it short


if __name__ == "__main__":
    run_cli(main)
