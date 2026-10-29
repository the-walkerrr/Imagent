"""
pipeline/serialisation.py — Pack embedding vectors for BLOB storage.

Vectors are stored as little-endian float32 (4 bytes per dimension) on every platform,
so a database copied between computers reads back the same. Search decodes them with
``numpy.frombuffer(blob, dtype="<f4")``.
"""

from __future__ import annotations

import array
import sys


def serialize_embedding(embedding: list[float]) -> bytes:
    """Pack a float list into little-endian float32 bytes."""
    packed = array.array("f", embedding)
    if sys.byteorder == "big":  # explicit branch: the array module writes the machine's native byte order
        packed.byteswap()
    return packed.tobytes()
