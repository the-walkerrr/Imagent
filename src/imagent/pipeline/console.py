"""pipeline/console.py — Make printing safe on every terminal, and end a command-line program politely."""

from __future__ import annotations

import os
import sqlite3
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import config
from db.connection import DatabaseProblem, describe_db_error


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
    except Exception as exc:
        saved = _save_traceback()
        where = f"Details were saved to {saved}." if saved else "The details could not be saved."
        sys.exit(f"Something unexpected went wrong ({type(exc).__name__}: {exc}). {where} "
                 "Running it again may work; if it keeps happening, please report it with that file.")
