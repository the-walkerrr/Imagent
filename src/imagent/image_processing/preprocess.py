import io
from PIL import Image, ImageOps
from imagent import config


def to_rgb(img: Image.Image) -> Image.Image:
    """The picture as RGB. A picture with transparent parts is shown on white, as image viewers show it: dropping the alpha channel
    instead leaves whatever color the transparent pixels hold, usually black, which hides a dark logo or text from the model."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "PA", "La", "RGBa") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return img.convert("RGB")


def preprocess_image(image_path: str) -> bytes:
    """The image as the JPEG sent to the vision model, at most ``config.IMAGE_MAX_LONG_EDGE`` on its long side."""
    with Image.open(image_path) as img:
        ImageOps.exif_transpose(img, in_place=True)  # turned where it is: a copy of the whole image is not made
        img = to_rgb(img)
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
