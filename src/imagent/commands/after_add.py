#!/usr/bin/env python3
"""
imagent/commands/after_add.py — What to do after a folder was added: a menu that comes back after each choice, until Exit.

Shown by ``imagent add`` (and by ``imagent scan`` when no folder was added yet) when there is a terminal to answer in. The folder
has only been added, nothing is scanned yet. From here the user can add another folder, scan and index the folder just added, scan every
folder and index what is pending (as ``imagent scan`` does), delete folders (as ``imagent delete`` does) or search the pictures (once some are indexed).
A choice that ends with a message or Ctrl+C brings the user back to the menu instead of ending the program.
"""

from __future__ import annotations

import sys
from pathlib import Path

import questionary

from imagent.commands import delete, scan, search
from imagent.commands.menu import STYLE, has_indexed_images
from imagent.os_specific.files import comparable_path

ADD, SCAN_THIS, SCAN_ALL, DELETE, SEARCH, EXIT = "add", "scan this", "scan all", "delete", "search", "exit"


def choices(folder: Path | None, indexed: bool = True) -> list:
    """The menu; ``folder`` is the folder added last (None if it is not there any more), which "scan and index" is about.
    "Search your images" is only offered when some image is indexed (``indexed``)."""
    entries = [("Add another folder", ADD)]
    if folder is not None:
        entries.append((f"Scan and index this folder ({folder})", SCAN_THIS))
    entries += [("Scan all folders and index the pictures waiting", SCAN_ALL), ("Delete folders", DELETE),
                *([("Search your images", SEARCH)] if indexed else []), ("Exit", EXIT)]
    return [questionary.Choice(title, value=value) for title, value in entries]


def _attempt(action):
    """Run ``action`` and return what it returns. Whatever ends it early (a message, Ctrl+C) is shown and ends only the action; None is returned."""
    try:
        return action()
    except SystemExit as stop:
        if isinstance(stop.code, str):
            print(stop.code, file=sys.stderr)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
    return None


def _scan_all() -> None:
    if not scan.added_folders():  # scan.main would ask for a folder to add, which would open this menu again inside itself
        print("No folders have been added, so there is nothing to scan.")
        return
    scan.main()


def _is_added(folder: Path) -> bool:
    return comparable_path(str(folder)) in {comparable_path(path) for path in scan.added_folders()}


def run(folder: Path) -> None:
    """Offer what to do next, again after each choice, until Exit; ``folder`` is the folder that was just added."""
    current: Path | None = folder
    while True:
        if current is not None and not _is_added(current):  # deleted from this menu
            current = None
        print()
        # The blank separator is skipped by the arrow keys; it only leaves a gap between the question and the entries.
        choice = questionary.select("What would you like to do next?", choices=[questionary.Separator(" ")] + choices(current, has_indexed_images()), qmark="▌",
                                    pointer="»", instruction=" ", style=STYLE).ask()
        if choice is None:  # Ctrl+C; questionary already said so
            sys.exit(1)
        if choice == EXIT:
            return
        if choice == ADD:
            current = _attempt(scan.add_folder) or current
        elif choice == SCAN_THIS:
            _attempt(lambda: scan.scan_and_index(current))
        elif choice == SCAN_ALL:
            _attempt(_scan_all)
        elif choice == DELETE:
            _attempt(lambda: delete.main([]))
        elif choice == SEARCH:
            _attempt(search.main)
