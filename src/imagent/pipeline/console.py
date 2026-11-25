"""imagent/pipeline/console.py — Make printing safe on every terminal, and end a command-line program politely."""

from __future__ import annotations

import errno
import os
import sqlite3
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

from imagent import config
from imagent.db.connection import DatabaseProblem, describe_db_error


def make_output_safe() -> None:
    """Replace characters the terminal cannot show instead of raising UnicodeEncodeError.

    Windows consoles with a legacy code page (and redirected output) cannot encode symbols
    such as ✓ or ─, and file names can hold any character.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def file_link(text: str, path: str | Path, stream=None) -> str:
    """``text`` as a link that opens the file ``path`` in the system's default viewer when clicked
    (Cmd+click or Ctrl+click in most terminals). Plain ``text`` when ``stream`` is not a terminal.

    The terminal is told exactly which file the text stands for, so a click does not depend on it guessing
    where a path starts and ends (spaces, non-ASCII names and wrapped lines defeat that guess)."""
    stream = stream or sys.stdout
    if not stream.isatty() or os.environ.get("TERM") == "dumb":
        return text
    try:
        uri = Path(path).absolute().as_uri()
    except ValueError:
        return text
    return f"\033]8;;{uri}\033\\{text}\033]8;;\033\\"


def _save_traceback() -> str | None:
    """Keep the traceback of an unforeseen error in a file the user can attach to a report; None if it cannot be written."""
    path = config.IMAGENT_HOME / "last_error.log"
    try:
        config.IMAGENT_HOME.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{datetime.now().isoformat()}\n{traceback.format_exc()}", encoding="utf-8")
        return str(path)
    except OSError:
        return None


gap_shown = False  # set once the blank line that ends a program has been printed, so it is never printed twice


def leave_a_gap() -> None:
    """End a program on a terminal with a blank line, so the shell's prompt does not touch the last line of output. Only once, and not when
    the output is redirected."""
    global gap_shown
    if gap_shown or not sys.stdout.isatty():
        return
    gap_shown = True
    print()


def run_cli(main: Callable[[], None], cancelled: str = "Cancelled.") -> None:
    """Run a command-line ``main`` so that whatever goes wrong ends as a short, plain message, not a traceback.

    Ctrl+C, a question with no terminal to answer it, a database that cannot be used, and anything
    unforeseen (its traceback is saved to a file) each get their own message and a non-zero exit code.
    """
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(f"\n{cancelled}")
    except EOFError:
        sys.exit("\nAn answer is needed here but there is no terminal to type it in. "
                 "Run this from a terminal, or give the value on the command line.")
    except DatabaseProblem as exc:
        sys.exit(f"{exc}")
    except sqlite3.Error as exc:
        sys.exit(f"{describe_db_error(exc)}")
    except OSError as exc:
        if exc.errno in _TERMINAL_ERRORS and not _terminal_is_there():  # the window was closed (or there never was one): not a fault of Imagent's
            sys.exit("\nThe terminal is not there any more (it was closed, or this was not run in a terminal). Nothing more was done; "
                     "run the command again from a terminal.")
        _exit_unexpected(exc)
    except Exception as exc:
        _exit_unexpected(exc)


_TERMINAL_ERRORS = (errno.EIO, errno.ENOTTY, errno.EBADF, errno.EINVAL)  # what reading a closed or missing terminal fails with


def _terminal_is_there() -> bool:
    try:
        return os.isatty(sys.stdin.fileno())
    except (OSError, ValueError, AttributeError):  # the descriptor is closed, or stdin was replaced by something with none
        return False


def _exit_unexpected(exc: BaseException) -> None:
    """End with the plain message for an error nobody foresaw; its traceback is saved to a file."""
    saved = _save_traceback()
    where = "Details were saved to last_error.log in Imagent's data folder." if saved else "The details could not be saved."
    sys.exit(f"Something unexpected went wrong ({type(exc).__name__}: {exc}). {where} "
             "Running it again may work; if it keeps happening, please report it with that file.")
