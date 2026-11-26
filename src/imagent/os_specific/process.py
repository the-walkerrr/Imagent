"""
imagent/os_specific/process.py — Start a background process that outlives the terminal's own signals, the way each operating system does it.
"""

from __future__ import annotations

import platform

_CREATE_NEW_PROCESS_GROUP = 0x00000200  # Windows: the process does not get this console's Ctrl+C
_CREATE_NO_WINDOW = 0x08000000  # Windows: the process has a console of its own that is never shown, and the programs it starts share it
# Not DETACHED_PROCESS (no console at all): every console program a detached process starts, such as the model runners of
# ``ollama serve``, would get a new visible console window of its own, opening and closing as models load.


def background_process_options() -> dict:
    """Keyword arguments for ``subprocess.Popen`` that start the process in the background, apart from this program's terminal."""
    if platform.system() == "Windows":
        return {"creationflags": _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW}
    return {"start_new_session": True}
