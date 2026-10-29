import io
from PIL import Image, ImageOps
import config

def preprocess_image(image_path: str) -> bytes:
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > config.IMAGE_MAX_LONG_EDGE:
            scale = config.IMAGE_MAX_LONG_EDGE / long_edge
            step = config.IMAGE_SIZE_MULTIPLE
            new_w = max((round(w * scale) // step) * step, step)
            new_h = max((round(h * scale) // step) * step, step)
            img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=config.IMAGE_JPEG_QUALITY, optimize=True)
    return buf.getvalue()
