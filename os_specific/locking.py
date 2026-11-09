"""
os_specific/locking.py — Lock an open file so that no other process can lock it too, the way each operating system does it.
"""

from __future__ import annotations

import os


def lock(handle) -> None:
    """Lock the open file ``handle`` without waiting. Raises ``OSError`` when another process holds the lock.

    The operating system releases the lock when the file is closed or the program ends, however it ends."""
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock(handle) -> None:
    """Release the lock taken with ``lock``."""
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_UN)
