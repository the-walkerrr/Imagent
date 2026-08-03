"""
pipeline/file_mover.py — Move indexed images into the ``indexed/`` output directory.

Preserves the relative directory structure under the root folder so the original
organisation remains intact inside ``indexed/``.  Handles filename collisions by
appending a short random suffix rather than silently overwriting.
"""

import shutil
import uuid
from pathlib import Path

from pipeline.discovery import INDEXED_DIR_NAME


def move_to_indexed(image_path: Path, root: Path) -> Path:
    """Move *image_path* into ``root/indexed/`` preserving its relative structure.

    Example::

        root  = /photos
        image = /photos/vacation/beach.jpg
        dest  = /photos/indexed/vacation/beach.jpg

    If the destination file already exists a short UUID hex suffix is appended
    to the stem (e.g. ``beach_a3f9c12d.jpg``) so no file is ever overwritten.

    Args:
        image_path: Absolute path to the image being indexed.
        root:       The root directory supplied by the user.

    Returns:
        The absolute ``Path`` of the image at its new location.
    """
    relative = image_path.relative_to(root)
    dest_dir = root / INDEXED_DIR_NAME / relative.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / image_path.name

    # Resolve filename collision without clobbering existing files.
    if dest.exists():
        short_id = uuid.uuid4().hex[:8]
        dest = dest_dir / f"{image_path.stem}_{short_id}{image_path.suffix}"

    shutil.move(str(image_path), str(dest))
    return dest
