"""
image_processing/ocr.py — Read the text in an image with a local OCR engine.

Runs on the original full-size image: the VLM only sees a downscaled copy, which loses small text.
Does not call Ollama.

An image larger than the engine reads without shrinking it is cut into tiles from the start (the engine would shrink it,
and finds text only above about 2% of the image side). When reading fails, what is tried next depends on where it failed,
and each step changes something: the whole image; then the image cut into tiles (no tile is larger than the engine reads without shrinking it,
so nothing is lost, and the engine works on far less at a time); then the tiles again with the engine loaded afresh.
When all of them fail the image is not stored without its text: it is set aside (``ImageProblem`` of kind ``ocr_failed``).
Finding no text is not a failure.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
from PIL import Image, ImageOps

import config
from image_processing.problems import OCR_FAILED, TOO_LARGE, ImageProblem, explain
from pipeline.failure import attempt_record

_EXIF_ORIENTATION = 0x0112
# What each EXIF orientation does to a picture, undone on each tile the same way ImageOps.exif_transpose does on a whole image.
_UNTURN = {
    2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90,
}


@functools.lru_cache(maxsize=1)
def _engine():
    # Imported here because loading the OCR models takes a moment and is only needed once.
    # The first import of requests (by rapidocr) warns when Python is built on LibreSSL, as the one macOS ships is;
    # nothing here depends on it, since Ollama is reached over plain http.
    warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
    from rapidocr import RapidOCR
    # By default it logs every model it loads; "error" also hides its warning for every image without text.
    return RapidOCR(params={"Global.log_level": "error"})


def check_available() -> None:
    """Load the OCR engine now, so a problem shows before indexing starts and not as images set aside for their text.

    Raises whatever the engine raises when it cannot load.
    """
    _engine()


def _read(pixels_bgr: np.ndarray) -> list[str]:
    """The pieces of text the engine finds in an image (an array of BGR pixels), above the confidence limit."""
    output = _engine()(pixels_bgr)
    texts = output.txts if output.txts is not None else ()  # None when no text was found
    confidences = output.scores if output.scores is not None else ()
    return [
        text.strip()
        for text, confidence in zip(texts, confidences)
        if text.strip() and float(confidence) >= config.OCR_MIN_CONFIDENCE
    ]


def _bgr(img: Image.Image) -> np.ndarray:
    return np.array(img if img.mode == "RGB" else img.convert("RGB"))[:, :, ::-1]  # the engine expects BGR


def _read_whole(image_path: str) -> list[str]:
    """Read the image in one piece; one that is larger than the engine reads without shrinking it is read in tiles at once,
    because the engine would shrink it and miss the smaller text."""
    with Image.open(image_path) as img:
        longest = max(img.size)
    if longest > int(getattr(_engine(), "max_side_len", 0) or longest):
        return _read_tiled(image_path)
    with Image.open(image_path) as img:
        ImageOps.exif_transpose(img, in_place=True)  # turned where it is: a copy of the whole image is not made
        pixels = _bgr(img)
    return _read(pixels)


def _starts(length: int, side: int, step: int) -> list[int]:
    """Where tiles of ``side`` begin along an edge of ``length``, ``step`` apart, the last one ending at the edge."""
    if length <= side:
        return [0]
    return sorted({*range(0, length - side, step), length - side})


def _read_tiled(image_path: str) -> list[str]:
    """Read the image tile by tile, from one decoded copy; only a tile is turned into an array at a time."""
    with Image.open(image_path) as img:
        orientation = img.getexif().get(_EXIF_ORIENTATION, 1)
        rgb = img if img.mode == "RGB" else img.convert("RGB")
        width, height = rgb.size
        # The engine shrinks anything above its own maximum side and, tested on images of text, finds text only when it is above about 2% of the image side;
        # so a tile is a share of that maximum, which makes smaller text readable.
        side = max(int(getattr(_engine(), "max_side_len", 0) * config.OCR_TILE_SHARE) or max(width, height), 1)
        step = max(int(side * (1 - config.OCR_TILE_OVERLAP)), 1)
        texts: list[str] = []
        for top in _starts(height, side, step):
            for left in _starts(width, side, step):
                tile = rgb.crop((left, top, min(left + side, width), min(top + side, height)))
                if orientation in _UNTURN:
                    tile = tile.transpose(_UNTURN[orientation])
                texts.extend(_read(_bgr(tile)))
    return texts


def extract_text_blocks(image_path: str) -> list[str]:
    """Return each separate piece of text found in the image (empty if it has none).

    Raises ``ImageProblem`` if the image cannot be opened, or if reading its text failed in every way tried.
    """
    steps = [("whole image", _read_whole), ("tiles", _read_tiled), ("tiles, engine loaded afresh", _read_tiled)]
    attempts: list[dict] = []
    for name, read in steps:
        if name.endswith("afresh"):
            _engine.cache_clear()
        try:
            return read(image_path)
        except Exception as exc:
            found = explain(exc, image_path)
            if found is not None and not (found.kind == TOO_LARGE and name == "whole image"):  # too little memory for the whole image: try tiles
                raise found from exc  # the image itself cannot be read; no other way of reading it helps
            attempts.append(attempt_record(name, OCR_FAILED, f"{type(exc).__name__}: {exc}"))
    message = "The text in the image could not be read, in any of the ways tried: " + "; ".join(
        f"{a['step']}: {a['detail']}" for a in attempts)
    raise ImageProblem(OCR_FAILED, message, attempts, service=True)
