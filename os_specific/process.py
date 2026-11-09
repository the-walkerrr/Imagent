"""
os_specific/process.py — Start a background process that outlives the terminal's own signals, the way each operating system does it.
"""

from __future__ import annotations

import platform

_DETACHED_PROCESS = 0x00000008  # Windows: the process has no console of its own and does not get this one's Ctrl+C


def background_process_options() -> dict:
    """Keyword arguments for ``subprocess.Popen`` that start the process in the background, apart from this program's terminal."""
    if platform.system() == "Windows":
        return {"creationflags": _DETACHED_PROCESS}
    return {"start_new_session": True}
