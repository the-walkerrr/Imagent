"""
image_processing/metadata.py — Extract file-level metadata from an image path.

Reads OS-level stat data (size, timestamps) and opens the image with Pillow to
determine pixel dimensions and MIME type.  Does not call the VLM.
"""

from __future__ import annotations

import mimetypes
import os
from datetime import datetime, timezone

from PIL import Image

_EXIF_IFD = 0x8769  # pointer to the Exif sub-directory (DateTimeOriginal lives here)
_GPS_IFD = 0x8825   # pointer to the GPS sub-directory
_NO_EXIF = {"taken_at": None, "camera_make": None, "camera_model": None, "gps_lat": None, "gps_lon": None}


def extract_file_metadata(path: str) -> dict:
    """Return a dict of file-level metadata for the image at *path*.

    Keys returned:
        width (int | None)       — pixel width; None if Pillow cannot open the file.
        height (int | None)      — pixel height; None if Pillow cannot open the file.
        file_size (int)          — file size in bytes.
        mime_type (str | None)   — guessed MIME type (e.g. 'image/jpeg').
        created_at (str | None)  — ISO 8601 UTC timestamp of file birth (None where the filesystem doesn't report it).
        modified_at (str)        — ISO 8601 UTC timestamp of last modification.
        taken_at (str | None)    — ISO 8601 local capture time from EXIF (no timezone); None if absent.
        camera_make (str | None), camera_model (str | None) — from EXIF; None if absent.
        gps_lat (float | None), gps_lon (float | None)      — decimal degrees from EXIF; None if absent.
    """
    stat = os.stat(path)

    # --- Dimensions and EXIF ---
    width: int | None = None
    height: int | None = None
    exif_fields = dict(_NO_EXIF)
    try:
        with Image.open(path) as img:
            width, height = img.size
            exif_fields = _extract_exif(img)
    except Exception:
        # Non-fatal: dimensions stay None; the rest of the metadata is still valid.
        pass

    # --- MIME type ---
    mime_type, _ = mimetypes.guess_type(path)

    # --- Timestamps ---
    modified_at = iso_utc(stat.st_mtime)

    # st_birthtime exists on macOS and (Python 3.12+) Windows, not on Linux; st_ctime is not a creation time, so don't substitute it.
    created_at: str | None = None
    if hasattr(stat, "st_birthtime"):
        created_at = iso_utc(stat.st_birthtime)

    return {
        "width": width,
        "height": height,
        "file_size": stat.st_size,
        "mime_type": mime_type,
        "created_at": created_at,
        "modified_at": modified_at,
        **exif_fields,
    }


def _extract_exif(img: Image.Image) -> dict:
    """Read capture time, camera and GPS from EXIF. Fields the file lacks stay None."""
    fields = dict(_NO_EXIF)
    try:
        exif = img.getexif()
        if not exif:
            return fields
        fields["camera_make"] = _clean_text(exif.get(0x010F))
        fields["camera_model"] = _clean_text(exif.get(0x0110))
        raw = exif.get_ifd(_EXIF_IFD).get(0x9003) or exif.get(0x0132)  # DateTimeOriginal, else DateTime
        if raw:
            try:
                fields["taken_at"] = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S").isoformat()
            except ValueError:
                pass
        gps = exif.get_ifd(_GPS_IFD)
        fields["gps_lat"] = _gps_degrees(gps.get(2), gps.get(1))
        fields["gps_lon"] = _gps_degrees(gps.get(4), gps.get(3))
    except Exception:
        # Non-fatal: EXIF is optional extra metadata; keep whatever was read.
        pass
    return fields


def _clean_text(value) -> str | None:
    text = str(value).replace("\x00", "").strip() if value is not None else ""
    return text or None


def _gps_degrees(dms, ref) -> float | None:
    """Convert EXIF (degrees, minutes, seconds) plus an N/S/E/W reference to signed decimal degrees."""
    try:
        degrees, minutes, seconds = (float(part) for part in dms)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    value = degrees + minutes / 60 + seconds / 3600
    return -value if ref in ("S", "W") else value


def iso_utc(timestamp: float) -> str:
    """Convert a POSIX timestamp to an ISO 8601 UTC string (also used by scan_and_index.py, so both agree on the format)."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
