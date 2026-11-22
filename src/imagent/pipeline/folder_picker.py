"""
imagent/pipeline/folder_picker.py — Ask which folder to add: type part of a folder name and pick it from the folders that match.

The folders are listed by walking the home folder (pipeline/discovery.py, with the same pruning as a scan) in a background thread that
``start_listing`` can start as soon as the command starts, so the list is usually ready by the time the user has typed a few letters.

The matches are ordered from the end of the path: a folder whose own name matches comes first, then one whose parent matches, and so
on, because a folder is remembered by its own name more than by what holds it.
"""

from __future__ import annotations

import os
import threading
import unicodedata
from pathlib import Path
from typing import NamedTuple

from imagent.os_specific.files import comparable_path, is_hidden_folder
from imagent.pipeline.discovery import _skip_folder, list_folders
from imagent.pipeline.validation import find_overlapping_root

ROWS = 8  # matches shown at once
MIN_QUERY = 3  # characters typed before any folder is suggested: one or two match most of the folders
PLACEHOLDER = f"enter at least {MIN_QUERY} characters of a folder name to get suggestions"
PLACEHOLDER_STYLE = "fg:#808080"  # a fixed mid grey: the terminal's own "bright black" can look the same as normal text in some themes


class Candidate(NamedTuple):
    path: str  # as the system gave it
    shown: str  # "~/Pictures/Trips"
    key: str  # what the typed text is compared with: below the home folder, "/" between folders, any upper or lower case


class _Listing:
    """The folders below one folder, being listed in a thread that starts at once."""

    def __init__(self, home: str):
        self.home = home
        self.folders: list[str] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        self.folders = list_folders(self.home)

    def wait(self) -> list[str]:
        self.thread.join()
        return self.folders


_listing: _Listing | None = None


def _listing_of(home: str) -> _Listing:
    """The listing of ``home`` that is under way, or a new one."""
    global _listing
    if _listing is None or _listing.home != home:
        _listing = _Listing(home)
    return _listing


def start_listing() -> None:
    """Start listing the folders below the home folder now, so the picker finds them ready."""
    _listing_of(os.path.expanduser("~"))


def _folders_below(home: str) -> list[str]:
    """The folders below ``home``, waiting for the listing if it is not done. A listing is used once: the next call lists again, as folders may have changed."""
    global _listing
    listing = _listing_of(home)
    folders = listing.wait()
    if _listing is listing:
        _listing = None
    return folders


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def candidates(folders: list[str], home: str, added: list[str]) -> list[Candidate]:
    """The folders of ``folders`` below ``home`` that can be added.

    Left out: a folder the scan itself skips, and any folder inside one (hidden by a leading dot or by the operating system,
    ``node_modules`` and the like: see pipeline/discovery.py); and a folder that is one of ``added`` or overlaps one,
    since adding it would be refused or be a rescan."""
    skipped: dict[str, bool] = {}  # a folder -> it is one the scan skips
    added_keys = {comparable_path(path) for path in added}
    found = []
    for path in folders:
        relative = os.path.relpath(path, home)
        if relative == "." or relative.startswith(".."):
            continue
        names = relative.split(os.sep)
        here = home
        for name in names:
            here = os.path.join(here, name)
            if here not in skipped:
                skipped[here] = _skip_folder(name) or is_hidden_folder(here)
            if skipped[here]:
                break
        else:
            if added and (comparable_path(path) in added_keys or find_overlapping_root(Path(path), added)):
                continue
            found.append(Candidate(path, "~" + os.sep + relative, _normalize("/".join(names))))
    return found


def rank(found: list[Candidate], query: str) -> list[Candidate]:
    """The folders of ``found`` that contain ``query``, the best match first.

    The deeper in the path the last match lies the further back a folder goes, so what matches in the folder's own name comes
    before what matches only in its parent. Within the same depth a name that is the query comes first, then one that starts with it,
    then one that contains it; after that the shorter path, then the path in alphabetical order.
    Fewer than ``MIN_QUERY`` characters match nothing."""
    wanted = _normalize(query).strip().replace(os.sep, "/")
    if len(wanted) < MIN_QUERY:
        return []
    scored = []
    for candidate in found:
        key = candidate.key
        start = key.rfind(wanted)
        if start < 0:
            continue
        end = start + len(wanted)
        starts_name = start == 0 or key[start - 1] == "/"
        ends_name = end == len(key) or key[end] == "/"
        kind = 0 if starts_name and ends_name else 1 if starts_name else 2
        scored.append(((key.count("/", end), kind, len(key), key), candidate))
    scored.sort()
    return [candidate for _, candidate in scored]


def _looks_like_path(text: str) -> bool:
    return text.startswith("~") or os.path.isabs(text)


class _Picker:
    """The prompt: a line to type in and, under it, the matching folders to choose from."""

    def __init__(self, added: list[str]):
        self.home = os.path.expanduser("~")
        self.added = added
        self.found: list[Candidate] | None = None  # None until the folders have been listed
        self.matches: list[Candidate] = []
        self.selected = 0
        self.app = None
        from prompt_toolkit.buffer import Buffer
        self.buffer = Buffer(on_text_changed=lambda _: self._refresh())

    def _refresh(self) -> None:
        text = self.buffer.text.strip()
        self.matches = rank(self.found, text) if self.found and not _looks_like_path(text) else []
        self.selected = 0

    def _load(self) -> None:
        self.found = candidates(_folders_below(self.home), self.home, self.added)
        self._refresh()
        if self.app is not None:
            self.app.invalidate()  # safe from this thread

    def _render(self) -> list[tuple[str, str]]:
        text = self.buffer.text.strip()
        if not text:
            return []
        if _looks_like_path(text):
            return [("class:instruction", "Press Enter to add this path.")]
        if len(text) < MIN_QUERY:
            return []
        if self.found is None:
            return [("class:instruction", "Looking for folders...")]
        if not self.matches:
            return [("class:instruction", "No folder matches. You can paste its path instead.")]
        lines: list[tuple[str, str]] = []
        for number, candidate in enumerate(self.matches[:ROWS]):
            chosen = number == self.selected
            lines.append(("class:highlighted" if chosen else "", ("» " if chosen else "  ") + candidate.shown + "\n"))
        if len(self.matches) > ROWS:
            lines.append(("class:instruction", f"  ...and {len(self.matches) - ROWS} more: keep typing to narrow them down"))
        return lines

    def ask(self, style) -> str | None:
        """Run the prompt; the folder chosen or typed, or None on Ctrl+C."""
        from prompt_toolkit.application import Application
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.processors import AfterInput, BeforeInput, ConditionalProcessor
        from prompt_toolkit.shortcuts import print_formatted_text

        typing = Window(BufferControl(self.buffer, input_processors=[
            BeforeInput([("class:qmark", "? "), ("class:question", "Folder to add: ")]),
            ConditionalProcessor(AfterInput(PLACEHOLDER, style=PLACEHOLDER_STYLE), Condition(lambda: not self.buffer.text)),
        ]), height=1)
        keys = KeyBindings()

        def move(step: int) -> None:
            shown = min(len(self.matches), ROWS)
            if shown:
                self.selected = (self.selected + step) % shown

        @keys.add("down")
        @keys.add("c-n")
        def _(event):
            move(1)

        @keys.add("up")
        @keys.add("c-p")
        def _(event):
            move(-1)

        @keys.add("enter")
        def _(event):
            text = self.buffer.text.strip()
            if self.matches:
                event.app.exit(result=self.matches[self.selected].path)
            elif text and (self.found is not None or _looks_like_path(text)):  # not while the folders are still being listed
                event.app.exit(result=text)

        @keys.add("c-c")
        def _(event):
            event.app.exit(result=None)

        self.app = Application(
            layout=Layout(HSplit([typing, Window(FormattedTextControl(self._render), height=ROWS + 1)]), focused_element=typing),
            key_bindings=keys, style=style, erase_when_done=True)
        threading.Thread(target=self._load, daemon=True).start()
        folder = self.app.run()
        if folder is None:
            print("Cancelled by user")
            return None
        print_formatted_text(FormattedText([("class:qmark", "? "), ("class:question", "Folder to add: "), ("class:answer", folder)]),
                             style=style)
        return folder


def ask_for_folder(added: list[str], style=None) -> str | None:
    """Ask which folder to add, suggesting the folders that match what is typed.

    ``added`` are the folders added before (they are not suggested). Returns what was chosen or typed, or None on Ctrl+C."""
    return _Picker(added).ask(style)
