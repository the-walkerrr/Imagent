"""
image_processing/convert.py — Make a PNG copy of a picture Pillow cannot open, with the tool the operating system has.

The tool is the operating system's own (os_specific/convert.py): macOS has ``sips``, Windows has its imaging components, and
Linux has none, so there nothing is converted. The copy is lossless, lives in a temporary folder and is deleted as soon as the picture
has been analysed (or the analysis failed). The original file is never touched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Iterator

import config
from os_specific import system_name
from os_specific.convert import command as command_for


class ConversionFailed(Exception):
    """The picture could not be converted; the message says why, for the record."""


@contextmanager
def converted_copy(path: str) -> Iterator[str]:
    """Give the path of a PNG copy of the picture at ``path``; the copy is deleted when the ``with`` block ends.

    Raises ``ConversionFailed`` when this computer has no tool to convert with or the tool could not read the picture."""
    folder = tempfile.mkdtemp(prefix="imagent-")
    try:
        target = os.path.join(folder, "converted.png")
        prepared = command_for(os.path.abspath(path), target)
        if prepared is None:
            raise ConversionFailed(f"{system_name()} has no built-in tool that converts pictures")
        command, extra_environment = prepared
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=config.CONVERT_TIMEOUT,
                                  env={**os.environ, **extra_environment})
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ConversionFailed(f"{command[0]} could not be run ({type(exc).__name__}: {exc})") from exc
        if done.returncode != 0 or not os.path.isfile(target) or os.path.getsize(target) == 0:
            raise ConversionFailed(f"{command[0]} could not convert the picture: {(done.stderr or done.stdout).strip()[:300] or 'no output'}")
        yield target
    finally:
        shutil.rmtree(folder, ignore_errors=True)
