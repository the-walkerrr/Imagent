#!/usr/bin/env python3
"""
imagent/commands/menu.py — Imagent: the menu that ``imagent`` opens when it is run with no command. Pick an entry with the arrow keys and Enter.

It only opens once setup is complete: when it is not, ``cli.py`` runs setup first (``needs_setup``), then opens the menu.
It runs the chosen command and ends with that command's exit code; "Exit" just closes the menu. "Show the images that could not be indexed" is only
offered when there are some (images are retried by themselves; this only tells the user which ones could not be indexed and why).
"Forget the deleted pictures Imagent remembers" is only offered when it remembers some. "Scan folders and index images" and
"Delete a folder" are only offered once a folder has been added (until then there is nothing to scan or delete). "Search your images" is only
offered once some image is indexed (until then there is nothing to search). "Help" (``imagent help``) is always offered, just before "Exit".

"""

from __future__ import annotations

import subprocess
import sys

import questionary

from imagent import config
from imagent.db.connection import init_db
from imagent.db.operations import count_deleted_analyses, count_embeddings, get_device_settings, get_exhausted_backlog, get_model_settings, list_roots
from imagent.pipeline import console
from imagent.pipeline.console import make_output_safe

UNINSTALL = "uninstall"
EXIT = "exit"

SEARCH = ("Search your images", "search")
ADD_FOLDER = ("Add a folder", "add")
SCAN = ("Scan folders and index images", "scan")
DELETE_FOLDER = ("Delete a folder", "delete")
FAILED_COMMAND = "failed"
FORGET_COMMAND = "forget"
HELP = ("Help: what each command does", "help")

STYLE = questionary.Style([
    ("qmark", "fg:ansidefault bold"),
    ("question", "fg:ansidefault bold"),
    ("pointer", "fg:ansidefault bold"),
    ("highlighted", "fg:ansidefault bold underline"),
    ("answer", "fg:ansidefault bold"),
    ("instruction", "fg:ansibrightblack"),
])


LOGO = """\
█████  █   █   ███    ████  █████  █   █  █████
  █    ██ ██  █   █  █      █      ██  █    █
  █    █ █ █  █████  █  ██  ████   █ █ █    █
  █    █   █  █   █  █   █  █      █  ██    █
█████  █   █  █   █   ████  █████  █   █    █"""
TAGLINE = "Local image indexing and search tool."


def show_header() -> None:
    """The name in big letters with the one-line description under it, after a blank line so it never touches the command typed above."""
    questionary.print("")
    questionary.print(LOGO, style="bold")
    questionary.print("")
    questionary.print(TAGLINE)
    questionary.print("")


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


def remembered_count() -> int:
    """How many deleted pictures Imagent remembers; 0 if that cannot be found out (the menu must always appear)."""
    try:
        conn = init_db(str(config.DB_PATH))
        try:
            return count_deleted_analyses(conn)
        finally:
            conn.close()
    except Exception:
        return 0


def has_folders() -> bool:
    """True when at least one folder has been added; True if that cannot be found out (the menu must always appear, complete)."""
    try:
        conn = init_db(str(config.DB_PATH))
        try:
            return bool(list_roots(conn))
        finally:
            conn.close()
    except Exception:
        return True


def has_indexed_images() -> bool:
    """True when at least one image is indexed (searchable, as search itself counts them); True if that cannot be found out (the menu must always appear, complete)."""
    try:
        conn = init_db(str(config.DB_PATH))
        try:
            return count_embeddings(conn) > 0
        finally:
            conn.close()
    except Exception:
        return True


def needs_setup() -> bool:
    """True when setup has not been completed: the models are not locked or this computer has not been measured.

    False if that cannot be found out, so the menu still appears (it offers Uninstall, which a damaged database may need).
    """
    try:
        conn = init_db(str(config.DB_PATH))
        try:
            return get_model_settings(conn) is None or get_device_settings(conn) is None
        finally:
            conn.close()
    except Exception:
        return False


def choices(failed: int = 0, remembered: int = 0, folders: bool = True, indexed: bool = True) -> list:
    shown = [(f"Show the {failed} image(s) that could not be indexed", FAILED_COMMAND)] if failed else []
    forget = [(f"Forget the {remembered} deleted picture(s) Imagent remembers", FORGET_COMMAND)] if remembered else []
    entries = ([*([SEARCH] if indexed else []), ADD_FOLDER, SCAN, DELETE_FOLDER] if folders else [ADD_FOLDER]) + shown + forget + [HELP, ("Exit", EXIT)]
    return [questionary.Choice(title, value=command) for title, command in entries] + [
        questionary.Choice([("fg:ansired", "Uninstall Imagent")], value=UNINSTALL)]


def main() -> None:
    make_output_safe()
    show_header()
    # The blank separator is skipped by the arrow keys; it only leaves a gap between the question and the entries.
    choice = questionary.select("What would you like to do?", choices=[questionary.Separator(" ")] + choices(failed_count(), remembered_count(), has_folders(), has_indexed_images()), qmark="▌",
                                pointer="»", instruction=" ", style=STYLE).ask()
    if choice is None:  # Ctrl+C; questionary already said so
        sys.exit(1)
    if choice == EXIT:
        return
    command = choice.split()
    process = subprocess.Popen([sys.executable, "-m", "imagent", *command])
    while True:
        try:
            code = process.wait()
        except KeyboardInterrupt:
            continue  # the command got the same Ctrl+C and is ending by itself: wait for it, do not cut it short
        console.gap_shown = True  # the command left the blank line when it ended
        sys.exit(code)

