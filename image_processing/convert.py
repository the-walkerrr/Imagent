"""
image_processing/convert.py — Make a PNG copy of a picture Pillow cannot open, with the tool the operating system has.

macOS has ``sips``; Windows can convert with the imaging components built into it (through PowerShell); Linux has no such
tool, so there nothing is converted. The copy is lossless, lives in a temporary folder and is deleted as soon as the picture
has been analysed (or the analysis failed). The original file is never touched.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Iterator

import config

# Reads the first frame with Windows' own decoders (the same ones Photos uses) and writes it as PNG. The paths arrive in
# environment variables, so no file name can end up being read as part of the script.
_WINDOWS_SCRIPT = (
    "$ErrorActionPreference = 'Stop'; "
    "Add-Type -AssemblyName PresentationCore; "
    "$decoder = [System.Windows.Media.Imaging.BitmapDecoder]::Create([Uri]$env:IMAGENT_IN, 'None', 'OnLoad'); "
    "$encoder = New-Object System.Windows.Media.Imaging.PngBitmapEncoder; "
    "$encoder.Frames.Add($decoder.Frames[0]); "
    "$stream = [System.IO.File]::Create($env:IMAGENT_OUT); "
    "try { $encoder.Save($stream) } finally { $stream.Close() }"
)


class ConversionFailed(Exception):
    """The picture could not be converted; the message says why, for the record."""


def _command(source: str, target: str) -> tuple[list[str], dict] | None:
    """The command that converts ``source`` to a PNG at ``target`` on this operating system, and its environment; None if there is none."""
    system = platform.system()
    if system == "Darwin" and shutil.which("sips"):
        return ["sips", "-s", "format", "png", source, "--out", target], {}
    if system == "Windows" and shutil.which("powershell"):
        return (["powershell", "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_SCRIPT],
                {"IMAGENT_IN": source, "IMAGENT_OUT": target})
    return None


@contextmanager
def converted_copy(path: str) -> Iterator[str]:
    """Give the path of a PNG copy of the picture at ``path``; the copy is deleted when the ``with`` block ends.

    Raises ``ConversionFailed`` when this computer has no tool to convert with or the tool could not read the picture."""
    folder = tempfile.mkdtemp(prefix="imagent-")
    try:
        target = os.path.join(folder, "converted.png")
        prepared = _command(os.path.abspath(path), target)
        if prepared is None:
            raise ConversionFailed(f"{platform.system()} has no built-in tool that converts pictures")
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
