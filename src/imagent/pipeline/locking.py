"""
pipeline/locking.py — Keep two copies of the same Imagent job from running at once.

The lock is held by the operating system on an open file, so it is released the moment the program
ends, however it ends (a crash, a power cut, a killed terminal). There is no stale lock to clean up.
"""

from __future__ import annotations

from contextlib import contextmanager

import config
from os_specific.locking import lock, unlock


class AlreadyRunning(RuntimeError):
    """The job is already running in another window; the message says so in plain words."""


def is_running(name: str) -> bool:
    """True if another process holds the lock called ``name`` (this one must not hold it itself)."""
    try:
        with single_instance(name, ""):
            return False
    except AlreadyRunning:
        return True


@contextmanager
def single_instance(name: str, what: str):
    """Hold the lock called ``name`` while the block runs; raise ``AlreadyRunning`` if someone else holds it.

    ``what`` names the job in the error message, e.g. "Indexing".
    """
    config.IMAGENT_HOME.mkdir(parents=True, exist_ok=True)
    with open(config.IMAGENT_HOME / f"{name}.lock", "a+b") as handle:
        try:
            lock(handle)
        except OSError as exc:
            raise AlreadyRunning(f"{what} is already running in another window. Wait for it to finish, then try again.") from exc
        try:
            yield
        finally:
            unlock(handle)
