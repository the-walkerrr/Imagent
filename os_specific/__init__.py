"""
os_specific — Everything that has to be done differently on macOS, Windows and Linux, in one place.

The rest of Imagent calls the functions here and never asks which operating system it runs on. Each module answers one
question (how much memory is free, is this folder on the computer's own disk, how is a picture converted, how is a file
locked, how is a background process started, ...), and each says what it does where an operating system is not covered.
Standard library only: the uninstaller imports it with a Python that has none of Imagent's packages.
"""

import platform


def system_name() -> str:
    """The name of the operating system, for a message to the user."""
    return platform.system()
