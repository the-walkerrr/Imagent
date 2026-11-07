"""
image_processing/problems.py — Tell what is wrong with an image that cannot be opened, from the error Pillow raised.

A damaged file, a format without a decoder, a file that is still being copied and an image too big for the memory
at hand each need a different answer, and none of them gets better by trying the same thing again.
"""

from __future__ import annotations

import os
import time

from PIL import Image, UnidentifiedImageError

import config
from pipeline.failure import IndexingFailure, attempt_record

UNREADABLE = "unreadable"  # damaged or incomplete: it is queued again when the file changes
UNSUPPORTED = "unsupported"  # no decoder for its format on this computer
TOO_LARGE = "too_large"  # more pixels than the memory free right now can hold
OCR_FAILED = "ocr_failed"  # the text in the image could not be read, by any of the ways tried
BUSY = "busy"  # still being written: tried again later, nothing is held against the file


class ImageProblem(IndexingFailure):
    """An image that cannot be indexed because of the image itself."""


def problem(kind: str, message: str, step: str = "opened the image", detail: str = "", **flags) -> ImageProblem:
    """``message`` is what the user is told (the reason and what can be done); ``detail`` is the technical side, kept in the record."""
    return ImageProblem(kind, message, [attempt_record(step, kind, detail or message)], **flags)


def file_is_changing(path: str) -> bool:
    """True if the file's size or time changes within ``config.FILE_SETTLE_SECONDS``: it is being written."""
    try:
        before = os.stat(path)
        time.sleep(config.FILE_SETTLE_SECONDS)
        after = os.stat(path)
    except OSError:
        return False
    return (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)


def explain(exc: BaseException, path: str) -> ImageProblem | None:
    """The problem with the image that ``exc`` (raised while opening or reading it) stands for, or None if it is another kind of error.

    Pillow's own errors (an unknown or broken file, a truncated one) carry no error number; a locked file, a bad sector
    or a dropped drive do, and are left to the caller, who tells them from a drive that has gone.
    """
    if isinstance(exc, MemoryError):
        return problem(TOO_LARGE, "There was not enough memory to open the picture. Close other programs, then scan again.",
                       detail=f"{type(exc).__name__}: {exc}", park=False)
    if not (isinstance(exc, (UnidentifiedImageError, SyntaxError)) or (isinstance(exc, OSError) and exc.errno is None)):
        return None
    suffix = os.path.splitext(path)[1].lower()
    if isinstance(exc, UnidentifiedImageError) and suffix not in Image.registered_extensions():
        return problem(UNSUPPORTED, f"Imagent cannot open {suffix} pictures on this computer. "
                                    "Convert the picture to JPEG or PNG and keep the copy in the same folder.",
                       detail=f"no decoder is installed for {suffix} files", retry_when="on_request")
    if file_is_changing(path):
        return problem(BUSY, "The file is still being written, so it is tried again later.", park=False, report=False)
    return problem(UNREADABLE, "The picture file is damaged or incomplete. Copy it again from the original, "
                               "or open it in a photo app and save it as a new file.",
                   detail=f"{type(exc).__name__}: {exc}", retry_when="file_changed")
