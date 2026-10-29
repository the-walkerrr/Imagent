"""
pipeline/progress.py — The progress displays of an indexing run (``Progress``) and of a search (``SearchProgress``).

Indexing:

While an image is processed, two lines are shown on a terminal: its file name, and what it is doing with the time so far.
When it is finished they are replaced by four lines (file name, description, status, time taken), then a blank line.
When output is not a terminal (a pipe or a file) only the finished blocks are printed.

Search: the query, then one line per step. A finished step stays on screen with its time; the current step
is a live line below them (spinner, name, time so far), shown on a terminal only.
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

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_REFRESH_SECONDS = 0.2
_TIMED = ("indexed", "updated", "FAILED")  # results that took real work, so their time is shown
_STATUS_COLOURS = {"indexed": "32", "updated": "32", "FAILED": "31"}  # anything else is dim
_MIN_WIDTH = 40
_STEP_WIDTH = 44
_STEP_TIME_WIDTH = 6
_PIPED_WIDTH = 120


def _format_duration(seconds: float, decimals: bool = False) -> str:
    """12s, 2m 30s or 1h 05m; with ``decimals`` a time below a minute reads 12.3s."""
    if decimals and seconds < 60:
        return f"{seconds:.1f}s"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {total % 3600 // 60:02d}m"


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
            timing = _format_duration(seconds, decimals=True) if result in _TIMED else "–"
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
        return f"{self._paint(_clip(self._name, width - len(position) - 2), '1')}  {self._paint(position, '2')}"

    def _erase(self) -> None:
        """Remove the two live lines; the cursor sits at the end of the second."""
        if self._live_shown:
            self._stream.write("\r\033[1A\033[J")
            self._live_shown = False

    def _redraw(self) -> None:
        self._erase()
        width = self._width()
        spinner = _SPINNER[int(time.monotonic() / _REFRESH_SECONDS) % len(_SPINNER)]
        elapsed = _format_duration(time.monotonic() - self._started)
        self._stream.write(self._name_line(width) + "\n" + _clip(f"{spinner} {self._stage}…  {elapsed}", width))
        self._stream.flush()
        self._live_shown = True


class SearchProgress:
    """Shows the steps of a search: finished ones stay on screen with their time, the current one is a live line."""

    def __init__(self, query: str = "", stream=None):
        self._stream = stream or sys.stdout
        self._live = self._stream.isatty() and os.environ.get("TERM") != "dumb"
        self._colour = self._live and "NO_COLOR" not in os.environ
        self._lock = threading.Lock()
        self._text: str | None = None  # the step in progress
        self._started = 0.0
        self._live_shown = False
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        if self._live and platform.system() == "Windows":
            os.system("")  # makes the Windows console understand ANSI escape codes
        if query:
            lines = textwrap.wrap(query, self._width()) or [""]
            self._stream.write("\n".join(self._paint(line, "1") for line in lines) + "\n\n")
            self._stream.flush()

    def step(self, text: str) -> None:
        """Finish the step in progress (as done) and begin ``text``."""
        with self._lock:
            self._finish("✓", "32")
            self._text = text
            self._started = time.monotonic()
            if self._live:
                self._redraw()
                if self._ticker is None:  # keeps the spinner and the time moving during a long model call
                    self._ticker = threading.Thread(target=self._tick, daemon=True)
                    self._ticker.start()

    def fail(self, note: str = "") -> None:
        """Finish the step in progress as failed; ``note`` says what happened instead."""
        with self._lock:
            self._finish("✗", "31", note)

    def say(self, message: str) -> None:
        """Print a message without breaking the live line."""
        with self._lock:
            self._erase()
            self._stream.write(message + "\n")
            if self._live and self._text is not None:
                self._redraw()
            self._stream.flush()

    def close(self) -> None:
        """Finish the step in progress (as done) and remove the live line. Safe to call more than once."""
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join()
        with self._lock:
            self._finish("✓", "32")

    def _tick(self) -> None:
        while not self._stop.wait(_REFRESH_SECONDS):
            with self._lock:
                if self._text is not None:
                    self._redraw()

    def _width(self) -> int:
        if not self._live:
            return _PIPED_WIDTH
        return max(shutil.get_terminal_size((80, 24)).columns - 1, _MIN_WIDTH)

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self._colour and text and code else text

    def _finish(self, mark: str, code: str, note: str = "") -> None:
        if self._text is None:
            return
        self._erase()
        seconds = _format_duration(time.monotonic() - self._started, decimals=True)
        line = f"{self._paint(mark, code)} {_clip(self._text, _STEP_WIDTH):<{_STEP_WIDTH}}{seconds:>{_STEP_TIME_WIDTH}}"
        self._stream.write(line + (f"  {self._paint(note, '2')}" if note else "") + "\n")
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
        elapsed = _format_duration(time.monotonic() - self._started)
        text = _clip(self._text, _STEP_WIDTH - 1) + "…"
        self._stream.write(f"{spinner} {text:<{_STEP_WIDTH}}{elapsed:>{_STEP_TIME_WIDTH}}")
        self._stream.flush()
        self._live_shown = True
