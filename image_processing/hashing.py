"""
image_processing/hashing.py — Content hash computation for change and move detection.

Always SHA-256 (stdlib): the stored hashes must be identical on every machine and install.
"""

import hashlib

# Read files in 64 KB chunks to keep memory usage bounded on large images.
_CHUNK_SIZE = 65_536


def compute_hash(path: str) -> str:
    """Return the hex-encoded SHA-256 content hash of the file at *path*."""
    hasher = hashlib.sha256()

    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            hasher.update(chunk)

    return hasher.hexdigest()
