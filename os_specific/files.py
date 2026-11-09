"""
os_specific/files.py — What each operating system says about a file or a folder that Python's ``os`` module does not.

  * a folder the user hid (Windows' hidden attribute, macOS's hidden flag): it holds the system's own caches, not pictures;
  * a file whose content is in cloud storage and not on this computer (an iCloud or OneDrive placeholder): reading it makes
    the system download it, and Imagent never uses the internet while it runs;
  * how two folder paths are compared: whether upper and lower case, or two ways of writing an accented letter, name the same folder.
"""

from __future__ import annotations

import os
import platform
import stat
import unicodedata

# Python's ``stat`` module has no names for these.
_UF_HIDDEN = 0x00008000  # macOS: the "hidden" flag of a file or folder (chflags hidden); ~/Library has it
_SF_DATALESS = 0x40000000  # macOS: the file's content is in iCloud (sys/stat.h)
_FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000  # Windows: the content is fetched from the cloud when the file is opened
_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000  # Windows: ... when its content is read (OneDrive "online-only")


def is_hidden_folder(path: str) -> bool:
    """True if the folder at ``path`` is hidden by the operating system's own attribute (not by a leading dot, which is the caller's to check).

    False where the system has no such attribute, or when the folder cannot be examined."""
    system = platform.system()
    if system not in ("Windows", "Darwin"):
        return False
    try:
        status = os.stat(path)
    except OSError:
        return False
    if system == "Windows":
        return bool(getattr(status, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_HIDDEN)
    return bool(getattr(status, "st_flags", 0) & _UF_HIDDEN)


def is_not_downloaded(status: os.stat_result) -> bool:
    """True if the file's content is kept in cloud storage and is not on this computer, from the ``os.stat`` of the file.

    Examining the file does not download it; opening or reading it does. False where the system has no such files."""
    system = platform.system()
    if system == "Windows":
        recalled = _FILE_ATTRIBUTE_RECALL_ON_OPEN | _FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS | stat.FILE_ATTRIBUTE_OFFLINE
        return bool(getattr(status, "st_file_attributes", 0) & recalled)
    if system == "Darwin":
        return bool(getattr(status, "st_flags", 0) & _SF_DATALESS)
    return False


def comparable_path(path: str) -> str:
    """``path`` in the form in which this operating system's file system tells two folder paths apart, for comparing them.

    Windows and macOS file systems do not tell upper from lower case; macOS also treats an accented letter written as one
    character or as two as the same. Linux's file systems tell them all apart, so nothing is changed there. Never use the result
    to open a file: only the spelling the file system gave is certain to open it."""
    system = platform.system()
    if system == "Darwin":
        path = unicodedata.normalize("NFC", path).casefold()
    elif system == "Windows":
        path = os.path.normcase(path).casefold()
    return path.rstrip(os.sep)
