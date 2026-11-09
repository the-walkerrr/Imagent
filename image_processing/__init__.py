# image_processing package
# Pillow cannot open HEIC/HEIF (iPhone photos) on its own. Registering the opener here makes it
# apply to every module of this package that calls Image.open.
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

# Pillow refuses images above a fixed number of pixels, whatever the memory of the computer. How big an image may be
# is decided from the memory free at the time instead (image_processing/memory.py), so Pillow's own limit is switched off.
# Only when this computer will not say how much memory is free is that limit used, as Pillow itself applies it: an image of
# more than twice its default number of pixels is refused. It is read here, before it is switched off, not typed in.
PIXEL_LIMIT_WHEN_MEMORY_UNKNOWN = None if Image.MAX_IMAGE_PIXELS is None else 2 * Image.MAX_IMAGE_PIXELS
Image.MAX_IMAGE_PIXELS = None
