# image_processing package
# Pillow cannot open HEIC/HEIF (iPhone photos) on its own. Registering the opener here makes it
# apply to every module of this package that calls Image.open.
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

# Pillow refuses images above a fixed number of pixels, whatever the memory of the computer. How big an image may be
# is decided from the memory free at the time instead (image_processing/memory.py), so Pillow's own limit is switched off.
Image.MAX_IMAGE_PIXELS = None
