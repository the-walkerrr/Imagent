#!/usr/bin/env python3
"""
imagent/commands/settings.py — Imagent: the settings menu, run by ``imagent settings`` and offered by the main menu.

Every setting is picked from a list of values; nothing is typed. The settings are kept in the database (``index_preferences``) and a
setting never chosen uses its default from ``config.py``:
  - how many deleted pictures Imagent remembers (asked at the end of setup too). Choosing fewer than are remembered now forgets the
    oldest ones, so it asks first;
  - whether a search shows the one-line summary or the full description of each picture (``imagent search --full`` / ``--short`` change it
    for one search).
"Uninstall Imagent" is here too. It is offered even when the database cannot be read, as the damage may be why the user is uninstalling.
The models and the measurements of this computer are not settings: they are locked at setup.
"""

from __future__ import annotations

import sys

import questionary

from imagent import config
from imagent.commands.menu import STYLE
from imagent.commands.setup import KEEP_DELETED_CHOICES, KEEP_DELETED_DEFAULT, keep_deleted_label
from imagent.db.connection import init_db
from imagent.db.operations import count_deleted_analyses, get_keep_deleted, get_search_detail, set_keep_deleted, set_search_detail
from imagent.pipeline.console import make_output_safe, run_cli

CANCELLED = "Settings closed."
KEEP_DELETED = "keep_deleted"
SEARCH_DETAIL = "search_detail"
UNINSTALL = "uninstall"
EXIT = "exit"

NOTE_STYLE = questionary.Style([*STYLE.style_rules, ("separator", "fg:#808080")])  # a fixed mid grey, as the terminal's "bright black" can look like normal text
SEARCH_DETAIL_SHOWS = {"short": "the one-line summary", "full": "the whole description"}


def read_settings() -> dict | None:
    """The settings as they are now (the defaults where none was chosen) and how many deleted pictures are remembered, or None if the
    database cannot be read."""
    try:
        conn = init_db(str(config.DB_PATH))
        try:
            return {KEEP_DELETED: get_keep_deleted(conn),  # None: not chosen yet, nothing is remembered
                    SEARCH_DETAIL: get_search_detail(conn) or config.SEARCH_DETAIL_DEFAULT,
                    "remembered": count_deleted_analyses(conn)}
        finally:
            conn.close()
    except Exception:
        return None


def choices(values: dict | None) -> list:
    """The entries of the settings menu. With no readable database only Exit and Uninstall are offered."""
    rows = []
    if values is not None:
        keep = values[KEEP_DELETED]
        rows = [(f"Deleted pictures Imagent remembers: {'not chosen yet (none)' if keep is None else keep}", KEEP_DELETED),
                (f"Search result detail: {values[SEARCH_DETAIL].capitalize()} ({SEARCH_DETAIL_SHOWS[values[SEARCH_DETAIL]]})", SEARCH_DETAIL)]
    return [questionary.Choice(title, value=value) for title, value in rows + [("Exit", EXIT)]] + [
        questionary.Choice([("fg:ansired", "Uninstall Imagent")], value=UNINSTALL)]


def _select(question: str, entries: list, default=None, note: str | None = None):
    """The value of the entry picked from ``entries``; Ctrl+C ends the program. ``note`` is a small grey description under the question, one line per ``\n``."""
    questionary.print("")  # a gap above every question, so it never touches the command typed or the line before it
    top = [questionary.Separator(line) for line in note.split("\n")] if note else []  # separators are skipped by the arrow keys
    prompt = questionary.select(question, choices=[*top, questionary.Separator(" ")] + entries, default=default, qmark="▌", pointer="»",
                                instruction=" ", style=NOTE_STYLE)
    # questionary also marks the default as "selected", which it colours. Keep only the pointer on it, as in the main menu.
    control = next(w.content for w in prompt.application.layout.find_all_windows() if hasattr(w.content, "selected_options"))
    control.selected_options.clear()
    answer = prompt.ask()
    if answer is None:  # Ctrl+C; questionary already said so
        sys.exit(1)
    return answer


def _save(setter, value) -> None:
    conn = init_db(str(config.DB_PATH))
    try:
        with conn:
            setter(conn, value)
    finally:
        conn.close()


def _done(text: str) -> None:
    questionary.print(f"\n✓ {text}", style="fg:green")


def change_keep_deleted(values: dict) -> None:
    current = values[KEEP_DELETED]
    numbers = sorted({*KEEP_DELETED_CHOICES, *([] if current is None else [current])})  # a number chosen before stays in the list
    new = _select("How many deleted pictures should Imagent remember?",
                  [questionary.Choice(keep_deleted_label(number), value=number) for number in numbers] + [questionary.Choice("Exit", value=EXIT)],
                  default=KEEP_DELETED_DEFAULT if current is None else current, note="After you delete a picture, Imagent can remember what it learned about it.\nIf the picture comes back, it does not need to be analysed again.\nOnce the limit is reached, the oldest are forgotten first.")
    if new == EXIT:  # nothing is changed (setup asks the same question without this entry)
        return
    forgotten = max(values["remembered"] - new, 0)
    if forgotten:
        keep = _select(f"Imagent remembers {values['remembered']} deleted pictures. Forget the {forgotten} oldest?",
                       [questionary.Choice("No, leave it as it is", value=False), questionary.Choice(f"Yes, forget {forgotten}", value=True)])
        if not keep:
            return
    _save(set_keep_deleted, new)
    _done(f"Imagent will remember the last {new} deleted pictures." if new else "Imagent will not remember deleted pictures.")


def change_search_detail(values: dict) -> None:
    new = _select("What should a search show for each picture?",
                  [questionary.Choice(f"{detail.capitalize()}: {SEARCH_DETAIL_SHOWS[detail]}", value=detail) for detail in config.SEARCH_DETAILS]
                  + [questionary.Choice("Exit", value=EXIT)], default=values[SEARCH_DETAIL])
    if new == EXIT:  # nothing is changed
        return
    _save(set_search_detail, new)
    _done(f"A search shows {SEARCH_DETAIL_SHOWS[new]} of each picture.")


CHANGES = {KEEP_DELETED: change_keep_deleted, SEARCH_DETAIL: change_search_detail}


def main() -> None:
    make_output_safe()
    if not sys.stdin.isatty():
        sys.exit("Settings need an interactive terminal. Run: imagent settings")
    chosen = None
    while True:
        values = read_settings()
        if values is None:
            questionary.print("Imagent's database cannot be read, so no setting can be shown or changed.")
        chosen = _select("Settings", choices(values), default=chosen if values is not None else None)
        if chosen == EXIT:
            return
        if chosen == UNINSTALL:
            from imagent.commands import uninstall
            run_cli(uninstall.main, cancelled=uninstall.CANCELLED)
            return  # uninstalled, or it said why not: either way there is nothing more to set
        CHANGES[chosen](values)
