"""
imagent/pipeline/progress.py — The progress displays of an indexing run (``Progress``) and of a search (``SearchProgress``).

Indexing:

While an image is processed, two lines are shown on a terminal: its file name, and what it is doing with the time so far.
When it is finished they are replaced by four lines (file name, description, status, time taken), then a blank line.
When output is not a terminal (a pipe or a file) only the finished blocks are printed.

Search: one live line (spinner and what is being done, such as "Searching…") that changes as the search moves on
and disappears when it is over. It is shown on a terminal only; nothing about the steps is printed when output is piped.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import textwrap
import threading
import time
from pathlib import Path

from imagent.pipeline.console import file_link

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_REFRESH_SECONDS = 0.2
_TIMED = ("indexed", "updated", "FAILED")  # results that took real work, so their time is shown
_STATUS_COLOURS = {"indexed": "32", "updated": "32", "FAILED": "31"}  # anything else is dim
_MIN_WIDTH = 40
_PIPED_WIDTH = 120


def format_duration(seconds: float, decimals: int = 0) -> str:
    """12s, 2m 30s, 1h 05m 30s or 2d 03h 05m 30s: minutes join once it is 60 seconds, hours at 60 minutes, days at 24 hours.
    With ``decimals`` a time below a minute shows that many digits after the point: 12.3s."""
    if decimals and seconds < 60:
        return f"{seconds:.{decimals}f}s"
    total = int(seconds)
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m {secs:02d}s"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(width - 1, 0)] + "…"


class Progress:
    """Shows which image is being processed and what it is doing, and what became of each one."""

    def __init__(self, total: int, stream=None):
        self._stream = stream or sys.stdout
        self._total = total
        self._digits = len(str(total))
        self._live = self._stream.isatty() and os.environ.get("TERM") != "dumb"
        self._colour = self._live and "NO_COLOR" not in os.environ
        self._lock = threading.Lock()
        self._index = 0
        self._name = ""
        self._path = ""
        self._stage = ""
        self._started = 0.0
        self._active = False  # an image is being processed, so its two lines are shown
        self._live_shown = False
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        if self._live and platform.system() == "Windows":
            os.system("")  # makes the Windows console understand ANSI escape codes

    def start(self, index: int, path: str | Path) -> None:
        """Begin the ``index``-th image (counting from 1)."""
        with self._lock:
            self._index = index
            self._name = Path(path).name
            self._path = path
            self._stage = "starting"
            self._started = time.monotonic()
            self._active = True
            if self._live:
                self._redraw()
                if self._ticker is None:  # keeps the spinner and the elapsed time moving during a long model call
                    self._ticker = threading.Thread(target=self._tick, daemon=True)
                    self._ticker.start()

    def stage(self, text: str) -> None:
        """Say what the current image is doing now."""
        with self._lock:
            self._stage = text
            if self._live and self._active:
                self._redraw()

    def done(self, result: str, detail: str = "") -> None:
        """Finish the current image: print its block. ``detail`` is a description or an error."""
        with self._lock:
            seconds = time.monotonic() - self._started
            self._active = False
            self._erase()
            timing = format_duration(seconds, decimals=1) if result in _TIMED else "–"
            self._write_block(" ".join(detail.split()) or "–", timing, result)

    def close(self) -> None:
        """Remove the two live lines. Safe to call more than once."""
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join()
        with self._lock:
            self._active = False
            self._erase()

    def _tick(self) -> None:
        while not self._stop.wait(_REFRESH_SECONDS):
            with self._lock:
                if self._active:
                    self._redraw()

    def _width(self) -> int:
        if not self._live:
            return _PIPED_WIDTH
        # The last column is left empty so a line never wraps; some terminals report a width of 0.
        return max(shutil.get_terminal_size((80, 24)).columns - 1, _MIN_WIDTH)

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self._colour and text and code else text

    def _write_block(self, detail: str, timing: str, result: str) -> None:
        """Write the four lines of the current image (a long description wraps), then a blank line."""
        width = self._width()
        lines = [self._name_line(width)]
        lines += [self._paint(text, "31" if result == "FAILED" else "") for text in textwrap.wrap(detail, width)]
        lines += [self._paint(result, _STATUS_COLOURS.get(result, "2")), timing]
        self._stream.write("\n".join(lines) + "\n\n")
        self._stream.flush()

    def _name_line(self, width: int) -> str:
        """The file name, with its place in the run after it."""
        position = f"({self._index:>{self._digits}}/{self._total})"
        name = file_link(_clip(self._name, width - len(position) - 2), self._path, self._stream)
        return f"{self._paint(name, '1')}  {self._paint(position, '2')}"

    def _erase(self) -> None:
        """Remove the two live lines; the cursor sits at the end of the second."""
        if self._live_shown:
            self._stream.write("\r\033[1A\033[J")
            self._live_shown = False

    def _redraw(self) -> None:
        self._erase()
        width = self._width()
        spinner = _SPINNER[int(time.monotonic() / _REFRESH_SECONDS) % len(_SPINNER)]
        elapsed = format_duration(time.monotonic() - self._started)
        self._stream.write(self._name_line(width) + "\n" + _clip(f"{spinner} {self._stage}…  {elapsed}", width))
        self._stream.flush()
        self._live_shown = True


class SearchProgress:
    """Shows what a search is doing as one live line that is replaced by the next step and removed at the end."""

    def __init__(self, stream=None):
        self._stream = stream or sys.stdout
        self._live = self._stream.isatty() and os.environ.get("TERM") != "dumb"
        self._lock = threading.Lock()
        self._text: str | None = None  # the step in progress
        self._live_shown = False
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        if self._live and platform.system() == "Windows":
            os.system("")  # makes the Windows console understand ANSI escape codes

    def step(self, text: str) -> None:
        """Show ``text`` as what is being done now, in place of the step before it."""
        with self._lock:
            self._text = text
            if self._live:
                self._redraw()
                if self._ticker is None:  # keeps the spinner moving during a long model call
                    self._ticker = threading.Thread(target=self._tick, daemon=True)
                    self._ticker.start()

    def fail(self, note: str = "") -> None:
        """Remove the live line because the step failed; ``note``, if given, says what happened instead."""
        with self._lock:
            self._finish(note)

    def say(self, message: str) -> None:
        """Print a message without breaking the live line."""
        with self._lock:
            self._erase()
            self._stream.write(message + "\n")
            if self._live and self._text is not None:
                self._redraw()
            self._stream.flush()

    def close(self) -> None:
        """Remove the live line. Safe to call more than once."""
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join()
        with self._lock:
            self._finish()

    def _tick(self) -> None:
        while not self._stop.wait(_REFRESH_SECONDS):
            with self._lock:
                if self._text is not None:
                    self._redraw()

    def _width(self) -> int:
        if not self._live:
            return _PIPED_WIDTH
        return max(shutil.get_terminal_size((80, 24)).columns - 1, _MIN_WIDTH)

    def _finish(self, note: str = "") -> None:
        """Remove the live line, and print ``note`` in its place if there is one."""
        if self._text is None:
            return
        self._erase()
        if note:
            self._stream.write(note + "\n")
            self._stream.flush()
        self._text = None

    def _erase(self) -> None:
        """Remove the live line; the cursor sits at its end."""
        if self._live_shown:
            self._stream.write("\r\033[J")
            self._live_shown = False

    def _redraw(self) -> None:
        self._erase()
        spinner = _SPINNER[int(time.monotonic() / _REFRESH_SECONDS) % len(_SPINNER)]
        self._stream.write(_clip(f"{spinner} {self._text}…", self._width()))
        self._stream.flush()
        self._live_shown = True
