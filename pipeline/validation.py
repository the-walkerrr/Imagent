"""
pipeline/validation.py — Validate the user-supplied root folder before indexing begins.

Exits the process immediately with a clear error message if the path is unusable,
so downstream code can assume the root is valid and readable.
"""

import os
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
