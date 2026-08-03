"""
image_processing/hashing.py — Content hash computation for duplicate and change detection.

Prefers BLAKE3 for speed and quality.  Falls back to SHA-256 (stdlib) when the
``blake3`` package is not installed so the pipeline never hard-fails on a missing
optional dependency.
"""

import hashlib

try:
    import blake3 as _blake3_mod
    _HAS_BLAKE3 = True
except ImportError:
    _HAS_BLAKE3 = False

# Read files in 64 KB chunks to keep memory usage bounded on large images.
_CHUNK_SIZE = 65_536


def compute_hash(path: str) -> str:
    """Return the hex-encoded content hash of the file at *path*.

    Uses BLAKE3 when available, SHA-256 otherwise.  The algorithm is not
    stored alongside the hash in this function — callers that need to record
    the algorithm name should use ``hash_algorithm()`` to retrieve it.
    """
    hasher = _blake3_mod.blake3() if _HAS_BLAKE3 else hashlib.sha256()

    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            hasher.update(chunk)

    return hasher.hexdigest()


def hash_algorithm() -> str:
    """Return the name of the algorithm currently in use: 'blake3' or 'sha256'."""
    return "blake3" if _HAS_BLAKE3 else "sha256"
