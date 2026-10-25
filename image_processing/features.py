"""
image_processing/features.py — Facts read straight from the pixels, not from the VLM.

Orientation and dominant colors are cheap to compute and more reliable than
asking a small model to guess them.
"""

from __future__ import annotations

import colorsys

from PIL import Image, ImageOps

_COLOR_SAMPLE_SIZE = (160, 160)
_PALETTE_SIZE = 6
_MIN_COLOR_SHARE = 0.05  # a color must cover at least this fraction of the image to be listed
_MAX_COLORS = 5


def extract_image_features(image_path: str) -> dict:
    """Return ``orientation`` ('landscape', 'portrait' or 'square') and ``dominant_colors``.

    EXIF orientation is applied first, so a phone photo stored sideways is judged as displayed.
    """
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        width, height = img.size
        if abs(width - height) / max(width, height) < 0.05:
            orientation = "square"
        else:
            orientation = "landscape" if width > height else "portrait"
        return {"orientation": orientation, "dominant_colors": _dominant_colors(img)}


def _dominant_colors(img: Image.Image) -> list[str]:
    palette_image = img.resize(_COLOR_SAMPLE_SIZE).quantize(_PALETTE_SIZE, method=Image.MEDIANCUT)
    palette = palette_image.getpalette()
    share: dict[str, int] = {}
    for count, index in palette_image.getcolors():
        name = _color_name(palette[index * 3: index * 3 + 3])
        share[name] = share.get(name, 0) + count
    total = sum(share.values())
    ranked = sorted(share.items(), key=lambda item: -item[1])
    return [name for name, count in ranked if count / total >= _MIN_COLOR_SHARE][:_MAX_COLORS]


def _color_name(rgb: list[int]) -> str:
    hue, saturation, value = colorsys.rgb_to_hsv(*(channel / 255 for channel in rgb))
    if value < 0.18:
        return "black"
    if saturation < 0.12:
        if value > 0.85:
            return "white"
        return "light gray" if value > 0.6 else ("gray" if value > 0.35 else "dark gray")
    degrees = hue * 360
    if degrees < 15 or degrees >= 345:
        base = "red" if saturation > 0.45 or value > 0.6 else "brown"
        if saturation < 0.5 and value > 0.7:
            base = "pink"
    elif degrees < 40:
        base = "orange" if value > 0.6 and saturation > 0.6 else "brown"
        if saturation < 0.35 and value > 0.7:
            base = "beige"
    elif degrees < 70:
        base = "yellow" if saturation > 0.4 else "beige"
    elif degrees < 160:
        base = "green"
    elif degrees < 200:
        base = "teal"
    elif degrees < 255:
        base = "blue"
    elif degrees < 300:
        base = "purple"
    else:
        base = "pink"
    if base in ("blue", "green", "teal") and value > 0.8 and saturation < 0.55:
        return f"light {base}"
    if base in ("blue", "green") and value < 0.4:
        return f"dark {base}"
    return base
