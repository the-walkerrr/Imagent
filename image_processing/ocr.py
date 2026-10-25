"""
image_processing/ocr.py — Read the text in an image with a local OCR engine.

Runs on the original full-size image: the VLM only sees a downscaled copy,
which loses small text. Does not call Ollama.
"""

from __future__ import annotations

import functools

import numpy as np
from PIL import Image, ImageOps

import config


@functools.lru_cache(maxsize=1)
def _engine():
    # Imported here because loading the OCR models takes a moment and is only needed once.
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


@functools.lru_cache(maxsize=None)
def _warn_once(message: str) -> None:
    print(message)


def extract_text_blocks(image_path: str) -> list[str]:
    """Return each separate piece of text found in the image.

    Returns an empty list if the image has no text, or if OCR is unavailable
    or fails: OCR is an enhancement, not a hard dependency of indexing.
    """
    try:
        engine = _engine()
        with Image.open(image_path) as img:
            pixels = np.array(ImageOps.exif_transpose(img).convert("RGB"))[:, :, ::-1]  # the engine expects BGR
        result, _ = engine(pixels)
    except Exception as exc:
        _warn_once(f"[imagent] OCR skipped ({type(exc).__name__}: {exc}); text in images will not be indexed.")
        return []
    return [
        text.strip()
        for _, text, confidence in (result or [])
        if text.strip() and float(confidence) >= config.OCR_MIN_CONFIDENCE
    ]
