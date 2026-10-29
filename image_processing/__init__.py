# image_processing package
# Pillow cannot open HEIC/HEIF (iPhone photos) on its own. Registering the opener here makes it
# apply to every module of this package that calls Image.open.
from pillow_heif import register_heif_opener

register_heif_opener()
