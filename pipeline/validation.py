"""
pipeline/validation.py — Validate the user-supplied root folder before indexing begins.

Exits the process immediately with a clear error message if the path is unusable,
so downstream code can assume the root is valid and readable.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def validate_folder(raw_path: str) -> Path:
    """Validate *raw_path* and return a resolved ``Path``.

    Checks (in order):
      1. The path is not empty.
      2. The path exists on disk.
      3. The path is a directory (not a file or symlink to a file).
      4. The current process has read permission on the directory.

    On any failure, prints a human-readable error to stdout and calls
    ``sys.exit(1)``.  On success, returns the resolved ``Path``.
    """
    if not raw_path:
        print("[imagent] Error: No folder path provided.")
        sys.exit(1)

    path = Path(raw_path).expanduser().resolve()

    if not path.exists():
        print(f"[imagent] Error: Path does not exist: {path}")
        sys.exit(1)

    if not path.is_dir():
        print(f"[imagent] Error: Path is not a directory: {path}")
        sys.exit(1)

    if not os.access(path, os.R_OK):
        print(f"[imagent] Error: Directory is not readable (permission denied): {path}")
        sys.exit(1)

    return path


def _comparable(path: str) -> str:
    """Normalise a path string for comparing two folders on this OS."""
    # Windows and macOS file systems are case-insensitive by default; Linux's are case-sensitive.
    if platform.system() in ("Windows", "Darwin"):
        path = os.path.normcase(path).casefold()
    return path.rstrip(os.sep)


def find_overlapping_root(root: Path, known_roots: list[str]) -> str | None:
    """Return a known root that contains *root* or lies inside it, or None.

    The same folder is not an overlap (that is a rescan). Nested folders are rejected
    because one file would then belong to two roots.
    """
    new = _comparable(str(root))
    for known in known_roots:
        old = _comparable(known)
        if new != old and (new.startswith(old + os.sep) or old.startswith(new + os.sep)):
            return known
    return None
