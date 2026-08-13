"""
pipeline/serialisation.py — Pack/unpack embedding vectors for BLOB storage.

Keeps embeddings at 4 bytes per dimension (IEEE 754 single precision) using
the stdlib ``array`` module, so this layer stays dependency-light regardless
of which numeric library downstream similarity computations use.
"""

from __future__ import annotations

import array


def serialize_embedding(embedding: list[float]) -> bytes:
    """Pack a float list into raw bytes (IEEE 754 single precision)."""
    return array.array("f", embedding).tobytes()


def deserialize_embedding(blob: bytes) -> list[float]:
    """Unpack raw bytes back into a float list (inverse of serialize_embedding)."""
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)
