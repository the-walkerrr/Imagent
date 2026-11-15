"""
os_specific/convert.py — The command that turns a picture Pillow cannot open into a PNG, with what the operating system has.

macOS has ``sips``; Windows can convert with the imaging components built into it (through PowerShell); Linux has no such
tool, so there nothing is converted.
"""

from __future__ import annotations

import platform
import shutil

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


def command(source: str, target: str) -> tuple[list[str], dict] | None:
    """The command that converts ``source`` to a PNG at ``target`` on this operating system, and its extra environment; None if there is none."""
    system = platform.system()
    if system == "Darwin" and shutil.which("sips"):
        return ["sips", "-s", "format", "png", source, "--out", target], {}
    if system == "Windows" and shutil.which("powershell"):
        return (["powershell", "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_SCRIPT],
                {"IMAGENT_IN": source, "IMAGENT_OUT": target})
    return None
