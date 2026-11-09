"""
image_processing/memory.py — Decide whether an image fits in the memory this computer has free right now.

No pixel count is fixed in advance: what an image may be depends on the machine, and on what else is using it. Before an
image is opened its size is read from the file's header; if the memory it needs at its peak is more than the share of the
free memory it may use, the image is set aside with the reason. When this computer will not say how much memory is free,
Pillow's own limit stands in for it (see image_processing/__init__.py), so an image is never let through unchecked.
"""

from __future__ import annotations

from PIL import Image

import config
from image_processing import PIXEL_LIMIT_WHEN_MEMORY_UNKNOWN
from image_processing.problems import TOO_LARGE, explain, problem
from os_specific.memory import available_memory

# Bytes of memory one pixel needs at the peak of reading an image, by the picture's mode. Pillow keeps a decoded picture in
# 1 byte a pixel for the modes 1, L and P, 2 for the 16-bit gray ones and 4 for every other (RGB too: it is padded to 4; the
# 3 bytes are only those of tobytes() and of an array), the same on every system. Reading a picture holds, besides the decoded
# image, the larger of two copies of it that are never held together: the RGB copy of a picture that is not RGB already
# (4 bytes a pixel), or the picture turned by its EXIF orientation (ImageOps.exif_transpose makes the turned one before it
# lets go of the first, in place or not), plus a little the decoder keeps while it decodes. That makes, turned or not:
# RGB 5 (9 when turned), RGBA and every other mode of 4 bytes 9, L, P and 1 6, 16-bit gray 7. The peaks measured on pictures of
# 48 megapixels agree: RGB 4.6 (8.3 turned), RGBA 8.6. Not counted: the OCR engine's array of a whole picture, 3 more bytes, which
# is only made of a picture small enough for the engine to read in one piece.
_ONE_BYTE_MODES = {"1", "L", "P"}
_TWO_BYTE_MODES = {"I;16", "I;16L", "I;16B", "I;16N"}
_RGB_COPY = 4
_DECODER_EXTRA = 1
_EXIF_ORIENTATION = 0x0112
_TURNED = range(2, 9)  # the orientations ImageOps.exif_transpose acts on; 1, or none, leaves the picture as it is


def peak_bytes_per_pixel(mode: str, turned: bool = False) -> int:
    """Bytes of memory one pixel of a picture of this ``mode`` needs at the peak of reading it (see above).

    ``turned``: the picture is turned by its EXIF orientation. A mode that is not known is taken to be one of 4 bytes."""
    stored = 1 if mode in _ONE_BYTE_MODES else 2 if mode in _TWO_BYTE_MODES else 4
    copy = max(_RGB_COPY if mode != "RGB" else 0, stored if turned else 0)
    return stored + copy + _DECODER_EXTRA


def _is_turned(img: Image.Image) -> bool:
    """Whether reading the opened picture turns it by its EXIF orientation, told from the header alone.

    A PNG's ``getexif()`` decodes the whole picture when the header has no EXIF chunk, which is what the check is there to avoid,
    so such a PNG counts as not turned."""
    if img.format == "PNG" and "exif" not in img.info:
        return False
    return img.getexif().get(_EXIF_ORIENTATION, 1) in _TURNED


def fits_in_memory(width: int, height: int, bytes_per_pixel: int, available: int | None, share: float | None = None) -> bool:
    """Whether an image of this size fits in the memory free.

    ``available`` None means this computer does not say; the image then fits if it is within Pillow's own limit.
    ``bytes_per_pixel``: what a pixel of the image needs at the peak (``peak_bytes_per_pixel``).
    ``share`` is the part of the free memory the image may use; ``config.IMAGE_MEMORY_SHARE`` unless given.
    """
    if available is None:
        return PIXEL_LIMIT_WHEN_MEMORY_UNKNOWN is None or width * height <= PIXEL_LIMIT_WHEN_MEMORY_UNKNOWN
    budget = available * (config.IMAGE_MEMORY_SHARE if share is None else share)
    return width * height * bytes_per_pixel <= budget


def exif_can_be_read(img: Image.Image) -> bool:
    """Whether the EXIF of the opened picture can be asked for without risking the memory.

    A PNG whose header has no EXIF chunk is the exception: Pillow then reads to the end of the file for one, which decodes the
    whole picture, so it is asked only when the picture fits in the memory free. Every other format keeps its EXIF in the header."""
    if img.format != "PNG" or "exif" in img.info:
        return True
    width, height = img.size
    return fits_in_memory(width, height, peak_bytes_per_pixel(img.mode), available_memory())


def check_image_size(path: str, share: float | None = None) -> None:
    """Raises ``ImageProblem`` if the image at ``path`` cannot be opened or is too large for the memory free.

    ``share``: the part of the free memory the image may use, ``config.IMAGE_MEMORY_SHARE`` unless given (see ``fits_in_memory``)."""
    try:
        with Image.open(path) as img:
            width, height = img.size
            bytes_per_pixel = peak_bytes_per_pixel(img.mode, _is_turned(img))
    except Exception as exc:
        raise (explain(exc, path) or exc) from exc
    available = available_memory()
    share = config.IMAGE_MEMORY_SHARE if share is None else share
    if not fits_in_memory(width, height, bytes_per_pixel, available, share):
        megapixels = width * height / 1_000_000
        if available is None:
            detail = f"this computer does not say how much memory is free, and the picture is more than Pillow's own limit of {PIXEL_LIMIT_WHEN_MEMORY_UNKNOWN} pixels"
            advice = "this computer does not say how much memory is free, so only a picture that is not very large can be read. Save a smaller copy of it from a photo app, then scan again."
        else:
            detail = (f"more than the memory free right now allows: it needs about {width * height * bytes_per_pixel // 10**6} MB "
                      f"and {int(available * share) // 10**6} MB can be used")
            advice = "more than the memory free right now allows. Close other programs, then scan again."
        raise problem(TOO_LARGE, f"The picture has {megapixels:.0f} megapixels, {advice}",
                      step="checked the size", detail=f"{megapixels:.0f} megapixels, {detail}", park=False)
