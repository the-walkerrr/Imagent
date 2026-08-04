"""
image_processing/metadata.py — Extract file-level metadata from an image path.

Reads OS-level stat data (size, timestamps) and opens the image with Pillow to
determine pixel dimensions and MIME type.  Does not call the VLM.
"""

from __future__ import annotations

import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def extract_file_metadata(path: str) -> dict:
    """Return a dict of file-level metadata for the image at *path*.

    Keys returned:
        file_id (str)           — OS-provided file identifier.
        width (int | None)       — pixel width; None if Pillow cannot open the file.
        height (int | None)      — pixel height; None if Pillow cannot open the file.
        file_size (int)          — file size in bytes.
        mime_type (str | None)   — guessed MIME type (e.g. 'image/jpeg').
        created_at (str | None)  — ISO 8601 UTC timestamp of file birth (macOS/Windows only).
        modified_at (str)        — ISO 8601 UTC timestamp of last modification.
    """
    stat = os.stat(path)

    # --- Dimensions ---
    width: int | None = None
    height: int | None = None
    try:
        with Image.open(path) as img:
            width, height = img.size
    except (UnidentifiedImageError, Exception):
        # Non-fatal: dimensions stay None; the rest of the metadata is still valid.
        pass

    # --- MIME type ---
    mime_type, _ = mimetypes.guess_type(path)

    # --- Timestamps ---
    modified_at = _iso_utc(stat.st_mtime)

    # st_birthtime is macOS/BSD-specific; st_ctime on Linux is inode-change time.
    created_at: str | None = None
    if hasattr(stat, "st_birthtime"):
        created_at = _iso_utc(stat.st_birthtime)

    return {
        "file_id": str(stat.st_ino),
        "width": width,
        "height": height,
        "file_size": stat.st_size,
        "mime_type": mime_type,
        "created_at": created_at,
        "modified_at": modified_at,
    }


def _iso_utc(timestamp: float) -> str:
    """Convert a POSIX timestamp to an ISO 8601 UTC string."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
