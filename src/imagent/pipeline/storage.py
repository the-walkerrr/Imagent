"""
pipeline/storage.py — Tell whether a folder is on this computer's own disk.

Imagent works with pictures stored on the computer it runs on. A network drive can disappear in the middle of a run and
an external drive can be unplugged, so folders on them are refused with a clear message instead of failing image by image.

Each system says it differently, and none says it perfectly, so this only refuses what it is sure of: when the answer
cannot be found the folder is allowed, and the rest of the program copes with a drive that goes away (see config.MAX_CONSECUTIVE_SERVICE_FAILURES).
Not checked: a USB hard drive on Windows, which Windows reports as a fixed disk.
"""

from __future__ import annotations

import os
from pathlib import Path

from os_specific.storage import EXTERNAL, NETWORK, classify  # EXTERNAL is part of this module's interface: callers and tests name storage.EXTERNAL


def storage_problem(path: str | Path) -> str | None:
    """A sentence saying why the folder cannot be used, or None if it is on this computer's own disk or that cannot be told."""
    try:
        kind = classify(os.path.realpath(path))
    except Exception:
        return None
    if kind is None:
        return None
    where = "a network drive" if kind == NETWORK else "an external or removable drive"
    return (f"{path} is on {where}. Imagent only works with pictures stored on this computer's own disk, "
            "because a drive like that can disappear in the middle of a run. Copy the pictures to this computer, then scan that folder.")
