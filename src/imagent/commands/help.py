#!/usr/bin/env python3
"""
imagent/commands/help.py — Imagent: explain each command briefly.

Run by ``imagent help``, and offered by the menu. It needs no setup, no database and no model, so it works on a computer where Imagent is not set up yet.
The text is written here by hand: when a command is added to ``cli.py`` it must be added to ``COMMANDS`` too (a test checks that none is missing).
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap

from imagent.pipeline.console import make_output_safe

INTRO = "Imagent indexes your pictures and finds them again when you describe them. Everything runs on this computer."
MENU_NOTE = "Run {} with no command to open a menu of the common ones."
MORE_NOTE = "For the details of one command, run: {}"

# (name, example of how to run it, what it does): shown as a table, the name on the left, the example and the explanation on the right
COMMANDS = [
    ("search", 'imagent search "red car at night" --full',
     "Show the pictures that best match a description. With no description it keeps asking until you type exit. Add --full for each picture's full description, or --short for its one-line summary; settings holds the usual choice."),
    ("add", 'imagent add "folder path"',
     "Add a folder of pictures to Imagent. You then choose what to do next, usually to scan the folder so its pictures are indexed. Leave out the folder to pick one from a list."),
    ("scan", "imagent scan",
     "Scan every folder you added for new, changed, moved or removed pictures, then index what is pending."),
    ("delete", 'imagent delete "folder path"',
     "Remove folders from the scan list and what was indexed from them. Your pictures are never touched. You can keep the analysis, so adding the folder again is quick."),
    ("failed", "imagent failed",
     "List the pictures that could not be indexed, with the reason for each."),
    ("forget", "imagent forget",
     "When you delete a picture, Imagent keeps its description, so if the picture comes back it is not analysed again. This command erases those kept descriptions, after two confirmations. Your indexed pictures are not touched."),
    ("settings", "imagent settings",
     "Change how many deleted pictures Imagent remembers, whether a search shows the one-line summary or the full description (add --full or --short to one search to change it just once), and how long the scan waits before it runs again by itself. Uninstall is here too."),
    ("setup", "imagent setup",
     "Check Ollama, download and lock the three models, and measure this computer. It runs by itself the first time it is needed; run it again to see or restore the models."),
    ("uninstall", "imagent uninstall",
     "Delete the index, Imagent's data folder and the models it downloaded, after three confirmations."),
    ("help", "imagent help",
     "Show this explanation."),
]

WIDTH = 100           # the widest a line is, on a wide terminal
MIN_TEXT = 30        # the least room kept for the descriptions on a narrow one
INDENT = 2           # before the names
GAP = 4              # between the names and the commands and descriptions
NAME_WIDTH = max(len(name) for name, _, _ in COMMANDS)
RIGHT = INDENT + NAME_WIDTH + GAP  # where the right column starts

BOLD = "1"           # the heading, the names and what to type to run a command


def _style(text: str, code: str) -> str:
    """``text`` with the ANSI style ``code`` when printing to a terminal that understands it, otherwise unchanged."""
    if not sys.stdout.isatty() or os.environ.get("TERM") == "dumb" or "NO_COLOR" in os.environ:
        return text
    return f"\033[{code}m{text}\033[0m"


def _width() -> int:
    """How wide a line may be: the terminal's width (its last column is left empty so a line never wraps by itself), at most ``WIDTH``."""
    return max(min(shutil.get_terminal_size((WIDTH, 24)).columns - 1, WIDTH), RIGHT + MIN_TEXT)


def render() -> str:
    width = _width()
    lines = ["", textwrap.fill(INTRO, width), "", MENU_NOTE.format(_style("imagent", BOLD)), "", _style("imagent <command>", BOLD)]
    for name, example, explanation in COMMANDS:
        # break_on_hyphens is off so a flag such as --full is never split between two lines
        right = [_style(example, BOLD), *textwrap.wrap(explanation, width - RIGHT, break_on_hyphens=False)]
        left = " " * INDENT + _style(name, BOLD) + " " * (NAME_WIDTH - len(name) + GAP)  # padded by the plain length: the style codes show nothing
        lines += [""] + [((left if row == 0 else " " * RIGHT) + line).rstrip() for row, line in enumerate(right)]
    lines += ["", MORE_NOTE.format(_style("imagent <command> --help", BOLD))]
    return "\n".join(lines)


def main() -> None:
    make_output_safe()
    print(render())
