import io
from PIL import Image, ImageOps
import config


def shrink_on_load(img: Image.Image, size: tuple[int, int]) -> None:
    """Ask the JPEG decoder to produce a smaller picture directly, still at least ``size``, so the full-size pixels never exist in memory.

    Does nothing for other formats, which have no such shortcut.
    """
    if img.format == "JPEG":
        img.draft("RGB", size)


def preprocess_image(image_path: str, gentle: bool = False) -> bytes:
    """The image as the JPEG sent to the vision model, at most ``config.IMAGE_MAX_LONG_EDGE`` on its long side.

    ``gentle``: for an image that is big for the memory free, a JPEG is shrunk while it is decoded (see image_processing/memory.py).
    """
    with Image.open(image_path) as img:
        w, h = img.size
        long_edge = max(w, h)
        if gentle and long_edge > config.IMAGE_MAX_LONG_EDGE:
            scale = config.IMAGE_MAX_LONG_EDGE / long_edge
            shrink_on_load(img, (max(round(w * scale), 1), max(round(h * scale), 1)))
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > config.IMAGE_MAX_LONG_EDGE:
            scale = config.IMAGE_MAX_LONG_EDGE / long_edge
            new_w = max(round(w * scale), 1)
            new_h = max(round(h * scale), 1)
            img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=config.IMAGE_JPEG_QUALITY, optimize=True)
    return buf.getvalue()
