#!/usr/bin/env python3
"""
imagent/commands/delete.py — Imagent: remove folders that were added, and what was indexed from them, after two questions.

Run by ``imagent delete``, with the folders to remove or, without any, choosing them from the folders that were added. It shows what will be
lost (the pictures indexed, the time spent analysing them, their size), asks whether to keep the analysis of the pictures (so adding a
folder again does not analyse its pictures again: the same kept analyses a scan keeps for deleted pictures, up to the number chosen at
setup), asks once more whether to delete, then deletes. The pictures themselves are never touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import questionary
from prompt_toolkit.keys import Keys
from questionary.prompts.common import InquirerControl

from imagent import config
from imagent.commands.menu import STYLE
from imagent.commands.uninstall import _fmt_bytes
from imagent.db.connection import init_db
from imagent.db.operations import (
    count_deleted_analyses, count_keepable_analyses, delete_roots, get_keep_deleted, image_ids_in_roots, keep_analyses_of,
    list_roots, root_stats,
)
from imagent.os_specific.files import comparable_path
from imagent.pipeline.console import make_output_safe
from imagent.pipeline.locking import AlreadyRunning, is_running, single_instance
from imagent.pipeline.progress import format_duration

# The menu style without the highlight on a row: no underline or bold under the pointer, and no reverse video on the row a question
# starts on (questionary draws the ``default`` row as "selected", which is reverse video unless a style says otherwise).
PICKER_STYLE = questionary.Style([rule for rule in STYLE.style_rules if rule[0] not in ("highlighted", "selected")]
                                 + [("highlighted", "noreverse nobold nounderline"), ("selected", "noreverse nobold nounderline")])
CANCELLED = "Cancelled. Nothing was deleted."
KEEP, DISCARD, DELETE, EXIT, CONFIRM = "keep", "discard", "delete", "exit", "confirm"


def match_folders(paths: list[str], roots: list[dict]) -> tuple[list[dict], list[str]]:
    """The added folders (rows of ``list_roots``) that ``paths`` name, each once, and the paths that are not an added folder."""
    by_path = {comparable_path(root["path"]): root for root in roots}
    chosen, unknown = [], []
    for raw in paths:
        root = by_path.get(comparable_path(str(Path(raw).expanduser().resolve())))
        if root is None:
            unknown.append(raw)
        elif root not in chosen:
            chosen.append(root)
    return chosen, unknown


def _ask(question):
    answer = question.ask()
    if answer is None:  # Ctrl+C; questionary already said so
        sys.exit(f"\n{CANCELLED}")
    return answer


def choose_folders(conn, roots: list[dict], **prompt_options) -> list[dict]:
    """Let the user tick the folders to remove from those that were added: Enter ticks or unticks the folder under the pointer, and
    Confirm, below the list, ends the choice (it is only offered once a folder is ticked). Exit, below Confirm, deletes nothing.

    It is one prompt that stays open: Enter redraws the row in place, so nothing flickers (a new prompt for each Enter would erase and
    draw the whole list again). ``prompt_options`` go to the prompt (the tests give it a fake keyboard and screen)."""
    indexed = {root["root_id"]: root_stats(conn, root["root_id"])["images"] for root in roots}
    ticked: set[int] = set()
    rows = {root["root_id"]: questionary.Choice("", value=root["root_id"]) for root in roots}
    confirm = questionary.Choice("Confirm", value=CONFIRM, disabled="tick a folder first")
    leave = questionary.Choice("Exit: delete nothing", value=EXIT)

    def draw() -> None:
        for root in roots:
            mark = "x" if root["root_id"] in ticked else " "
            rows[root["root_id"]].title = f"[{mark}] {root['path']}   ({indexed[root['root_id']]:,} indexed)"
        confirm.disabled = None if ticked else "tick a folder first"

    draw()
    questionary.print("")
    questionary.print("▌Which folders do you want to remove from the scan list?", style="bold")  # the message below is blank: it is the gap under this line
    question = questionary.select("", choices=[*rows.values(), questionary.Separator(" "), confirm, leave], qmark=" ", pointer="»", instruction=" ",
                                  style=PICKER_STYLE, **prompt_options)
    application = question.application
    control = next(control for control in application.layout.find_all_controls() if isinstance(control, InquirerControl))

    @application.key_bindings.add(Keys.ControlM, eager=True)  # added last, so it is used instead of the prompt's own Enter
    def press_enter(event):
        choice = control.get_pointed_at()
        if choice.value in (CONFIRM, EXIT):
            control.is_answered = True
            event.app.exit(result=choice.value)
            return
        ticked.symmetric_difference_update({choice.value})
        draw()
        event.app.invalidate()

    application.erase_when_done = True
    if _ask(question) == EXIT:
        sys.exit(f"\n{CANCELLED}")
    return [root for root in roots if root["root_id"] in ticked]


def show_what_is_lost(conn, folders: list[dict]) -> dict:
    """Print what will be lost, folder by folder; returns the totals."""
    total = {"images": 0, "bytes": 0, "analysis_ms": 0, "queued": 0}
    print("\n  This removes the folders from the scan list and what was indexed from them.")
    for folder in folders:
        stats = root_stats(conn, folder["root_id"])
        for key in total:
            total[key] += stats[key]
        print(f"\n  {folder['path']}")
        _print_stats(stats)
    if len(folders) > 1:
        print(f"\n  All {len(folders)} folders")
        _print_stats(total)
    print("\n  Your pictures are not touched, only what Imagent stored about them (descriptions and search data).")
    return total


def _print_stats(stats: dict) -> None:
    print(f"      Pictures indexed    {stats['images']:,}")
    print(f"      Analysis time       {format_duration(stats['analysis_ms'] / 1000)}   (the time spent describing them)")
    print(f"      Pictures' size      {_fmt_bytes(stats['bytes'])}   (on your disk, they stay there)")
    if stats["queued"]:
        print(f"      Waiting to index    {stats['queued']:,}")


def ask_keep(keepable: int, limit: int | None) -> bool | None:
    """Ask whether to keep the analysis of the pictures; True to keep it, False not to, None when the user chose to exit.

    Nothing is asked when nothing can be kept: ``limit`` is how many deleted pictures Imagent was set up to remember."""
    if not keepable:
        return False
    if not limit:
        print("\n  Imagent was set up to remember no deleted pictures, so none of this can be kept.")
        return False
    kept = min(keepable, limit)
    reach = (f"{kept:,} of the {keepable:,} pictures (the most Imagent was set up to remember)" if kept < keepable
             else f"all {keepable:,} pictures")  # what adding the folders again will recognise
    questionary.print("")  # a gap above the question
    choice = _ask(questionary.select(
        "Keep the analysis of these pictures?\n",
        choices=[
            questionary.Choice(f"Keep it: {reach} are not analysed again if you add the folders back", value=KEEP),
            questionary.Choice("Don't keep it: every picture is analysed again if you add the folders back", value=DISCARD),
            questionary.Choice("Exit: delete nothing", value=EXIT),
        ], qmark="▌", pointer="»", instruction=" ", style=PICKER_STYLE))
    return None if choice == EXIT else choice == KEEP


def main(paths: list[str] | None = None) -> None:
    make_output_safe()
    if not sys.stdin.isatty():
        sys.exit('Deleting needs an interactive terminal. Run: imagent delete "folder path"')
    if is_running("indexing"):  # said now, not after the questions: the lock is checked again when it is taken, in case one starts meanwhile
        sys.exit("Indexing is already running in another window. Wait for it to finish, then try again. Nothing was deleted.")
    conn = init_db(str(config.DB_PATH))
    try:
        roots = list_roots(conn)
        if not roots:
            print("No folders have been added, so there is nothing to delete.")
            return
        if paths:
            folders, unknown = match_folders(paths, roots)
            if unknown:
                sys.exit(f"These are not folders you added: {', '.join(unknown)}\n"
                         'If the path has spaces, put it in quotes: imagent delete "folder path"\n'
                         "Run imagent delete without a folder to choose from the folders you added.\n"
                         "Nothing was deleted.")
        else:
            folders = choose_folders(conn, roots)
        show_what_is_lost(conn, folders)
        ids = [folder["root_id"] for folder in folders]
        limit = get_keep_deleted(conn)
        keep = ask_keep(count_keepable_analyses(conn, ids), limit)
        if keep is None:
            sys.exit(f"\n{CANCELLED}")
        noun = "folder" if len(folders) == 1 else "folders"
        questionary.print("")  # a gap above the question
        choice = _ask(questionary.select(
            f"Are you sure you want to remove {len(folders)} {noun} from the scan list and delete what was indexed from {'it' if len(folders) == 1 else 'them'}?\n",
            choices=[questionary.Choice("Yes, delete", value=DELETE), questionary.Choice("Exit: delete nothing", value=EXIT)],
            default=EXIT, qmark="▌", pointer="»", instruction=" ", style=PICKER_STYLE))
        if choice != DELETE:
            sys.exit(f"\n{CANCELLED}")
        try:
            with single_instance("indexing", "Indexing"):  # a scan or an indexing run could be adding to these folders meanwhile
                with conn:
                    image_ids = image_ids_in_roots(conn, ids)
                    keep_analyses_of(conn, image_ids, limit if keep else 0)
                    delete_roots(conn, ids)
        except AlreadyRunning as exc:
            sys.exit(f"\n{exc} Nothing was deleted.")
        print(f"\n  ✓ Removed {len(folders)} {noun} from the scan list and the index of {len(image_ids):,} picture(s). Your pictures were not touched.")
        if keep:
            print(f"  ✓ Imagent remembers {count_deleted_analyses(conn):,} deleted picture(s). Add a folder again with "
                  f'{config.ADD_COMMAND} "folder path" and the ones it recognises are not analysed again.')
    finally:
        conn.close()
