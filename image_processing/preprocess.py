import io
from PIL import Image, ImageOps
import config


def preprocess_image(image_path: str) -> bytes:
    """The image as the JPEG sent to the vision model, at most ``config.IMAGE_MAX_LONG_EDGE`` on its long side."""
    with Image.open(image_path) as img:
        ImageOps.exif_transpose(img, in_place=True)  # turned where it is: a copy of the whole image is not made
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > config.IMAGE_MAX_LONG_EDGE:
            scale = config.IMAGE_MAX_LONG_EDGE / long_edge
            new_w = max(round(w * scale), 1)
            new_h = max(round(h * scale), 1)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()  # inside the with: when the image is already RGB and small enough, img is still the opened file
        img.save(buf, format="JPEG", quality=config.IMAGE_JPEG_QUALITY, optimize=True)
    return buf.getvalue()
