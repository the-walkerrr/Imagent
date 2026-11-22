"""
imagent/os_specific/folder_index.py — The folders the operating system already knows about, so a folder can be picked without walking the disk.

Only macOS is covered, through Spotlight (``mdfind``). Everywhere else ``can_list_folders`` is False and the caller asks for the folder
as plain text.
"""

from __future__ import annotations

import platform
import shutil
import subprocess


def can_list_folders() -> bool:
    """True if this system can list its folders from an index it keeps."""
    return platform.system() == "Darwin" and shutil.which("mdfind") is not None


def list_folders(under: str) -> list[str]:
    """Every folder at or below ``under`` that the system's index knows, in no particular order; [] if the index cannot be asked or is empty.

    The index can be a little out of date: a folder it lists may be gone, and a new one may be missing."""
    try:
        result = subprocess.run(["mdfind", "-0", "-onlyin", under, 'kMDItemContentType == "public.folder"'],
                                capture_output=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    return [path.decode("utf-8", "replace") for path in result.stdout.split(b"\0") if path]
