"""
image_processing/ocr.py — Read the text in an image with a local OCR engine.

Runs on the original full-size image: the VLM only sees a downscaled copy,
which loses small text. Does not call Ollama.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
from PIL import Image, ImageOps

import config


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
    """Load the OCR engine now, so a problem shows before indexing starts and not as images stored without their text.

    The first load downloads the engine's small models, which needs an internet connection once.
    Raises whatever the engine raises when it cannot load.
    """
    _engine()


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
        output = engine(pixels)
        texts = output.txts if output.txts is not None else ()  # None when no text was found
        confidences = output.scores if output.scores is not None else ()
    except Exception as exc:
        _warn_once(f"[imagent] OCR skipped ({type(exc).__name__}: {exc}); text in images will not be indexed.")
        return []
    return [
        text.strip()
        for text, confidence in zip(texts, confidences)
        if text.strip() and float(confidence) >= config.OCR_MIN_CONFIDENCE
    ]
