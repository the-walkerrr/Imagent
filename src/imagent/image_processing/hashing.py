"""
imagent/image_processing/hashing.py — Content hashes for change and move detection.

Always SHA-256 (stdlib): the stored hashes must be identical on every machine and install.
"""

import hashlib

from PIL import Image, ImageOps

# Read files in 64 KB chunks to keep memory usage bounded on large images.
_CHUNK_SIZE = 65_536

# The pixels are hashed in bands of about this many bytes, so no second copy of a whole picture is made.
_BAND_BYTES = 1_048_576
_PALETTE_MODES = {"P", "PA"}  # their pixels are numbers into a palette: the colors are hashed, so a changed palette changes the hash


def compute_hash(path: str) -> str:
    """Return the hex-encoded SHA-256 content hash of the file at *path*."""
    hasher = hashlib.sha256()

    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_pixel_hash(path: str) -> str:
    """Return the hex-encoded SHA-256 of the picture at *path* as it is looked at: its pixels after the EXIF orientation is applied.

    Two files with the same pixel hash show the same picture to the vision model, whatever else their bytes hold (EXIF facts,
    XMP, a color profile, the way the file was saved). Turning the picture by its orientation is part of it, because the model
    sees the turned picture. Only the first frame of an animation counts, as it does in analysis. The caller makes sure the
    picture fits in memory first (``image_processing.memory.check_image_size``).

    Raises what Pillow raises for a file it cannot read."""
    hasher = hashlib.sha256()
    with Image.open(path) as img:
        ImageOps.exif_transpose(img, in_place=True)  # turned where it is: a copy of the whole image is not made
        width, height = img.size
        palette = img.mode in _PALETTE_MODES
        hasher.update(f"{'RGBA' if palette else img.mode}:{width}x{height}".encode())
        rows = max(_BAND_BYTES // (width * 4), 1)
        for top in range(0, height, rows):
            band = img.crop((0, top, width, min(top + rows, height)))
            hasher.update((band.convert("RGBA") if palette else band).tobytes())
    return hasher.hexdigest()
