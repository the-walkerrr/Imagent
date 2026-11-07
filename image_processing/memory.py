"""
image_processing/memory.py — Decide how an image is read from the memory this computer has free right now.

No pixel count is fixed in advance: what an image may be depends on the machine, and on what else is using it. Before an
image is opened its size is read from the file's header, and it gets one of three ways of being read:

  normal     as always: the whole image is decoded and copied a few times.
  gentle     for an image that is big for the memory free: a JPEG is shrunk while it is decoded, and the text in the image
             is read in tiles from a single decoded copy.
  too_large  not even that fits; the image is set aside with the reason.
"""

from __future__ import annotations

import ctypes
import platform
import re
import subprocess
from pathlib import Path

from PIL import Image

import config
from image_processing.problems import TOO_LARGE, explain, problem

NORMAL = "normal"
GENTLE = "gentle"

# Bytes of memory one pixel needs at the peak of each way of reading, from the copies the code holds at once
# (an RGB copy is 3 bytes per pixel). Counted from the code, not measured on a machine:
NORMAL_BYTES_PER_PIXEL = 12  # the decoded image, its copy turned by the EXIF orientation, its RGB copy, and the array the OCR engine reads
GENTLE_BYTES_PER_PIXEL = 6  # the decoded image and, if it was not RGB already, one RGB copy; the OCR engine reads tiles cut from it


def _parse_meminfo(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _parse_vm_stat(text: str) -> int | None:
    size = re.search(r"page size of (\d+) bytes", text)
    pages = {name: int(count) for name, count in re.findall(r"Pages (free|inactive|speculative):\s+(\d+)", text)}
    if not size or len(pages) < 3:
        return None
    return (pages["free"] + pages["inactive"] + pages["speculative"]) * int(size.group(1))


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_memory() -> int | None:
    """Bytes of memory free for a program to use right now, or None if this computer will not say."""
    try:
        system = platform.system()
        if system == "Linux":
            return _parse_meminfo(Path("/proc/meminfo").read_text())
        if system == "Darwin":
            return _parse_vm_stat(subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout)
        if system == "Windows":
            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullAvailPhys)
    except Exception:
        pass
    return None


def choose_route(width: int, height: int, available: int | None) -> str:
    """``NORMAL``, ``GENTLE`` or ``TOO_LARGE`` for an image of this size, given the memory free (None: not known).

    When the free memory is not known the gentle way is used, which is slower but safe.
    """
    pixels = width * height
    limit = config.IMAGE_MAX_MEGAPIXELS
    if limit and pixels > limit * 1_000_000:
        return TOO_LARGE
    if available is None:
        return GENTLE
    budget = available * config.IMAGE_MEMORY_SHARE
    if pixels * NORMAL_BYTES_PER_PIXEL <= budget:
        return NORMAL
    if pixels * GENTLE_BYTES_PER_PIXEL <= budget:
        return GENTLE
    return TOO_LARGE


def plan_image(path: str) -> str:
    """How to read the image at ``path``: ``NORMAL`` or ``GENTLE``. Raises ``ImageProblem`` if it cannot be opened or is too large."""
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception as exc:
        raise (explain(exc, path) or exc) from exc
    available = available_memory()
    route = choose_route(width, height, available)
    if route == TOO_LARGE:
        megapixels = width * height / 1_000_000
        limit = config.IMAGE_MAX_MEGAPIXELS
        if limit and megapixels > limit:
            reason = f"more than the {limit:g} megapixels allowed by max_megapixels in config.json. Raise that number in config.json, or remove it"
            detail = reason
        else:
            reason = "more than the memory free right now allows. Close other programs, then scan again"
            detail = (f"more than the memory free right now allows: it needs about {width * height * GENTLE_BYTES_PER_PIXEL // 10**6} MB "
                      f"and {int(available * config.IMAGE_MEMORY_SHARE) // 10**6} MB can be used")
        raise problem(TOO_LARGE, f"The picture has {megapixels:.0f} megapixels, {reason}.", step="checked the size",
                      detail=f"{megapixels:.0f} megapixels, {detail}", park=False)
    return route
