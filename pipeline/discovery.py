"""
pipeline/discovery.py — Recursively discover supported image files under a root directory.

Skips the ``indexed/`` sub-directory that Imagent creates when moving processed
images, so re-running the pipeline on the same root does not re-index files that
have already been moved.
"""

from pathlib import Path

# Extensions that Imagent can process.  All comparisons are done case-insensitively.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg",
    ".png",
    ".webp",
    ".heic", ".heif",
    ".bmp",
    ".tiff", ".tif",
    ".gif",
    ".avif",
})

# Name of the output directory created by the pipeline; excluded from discovery.
INDEXED_DIR_NAME = "indexed"


def discover_images(root: Path) -> list[Path]:
    """Return a sorted list of image paths found recursively under *root*.

    Files inside ``root/indexed/`` are excluded to avoid re-processing images
    that have already been indexed and moved by a previous run.

    Args:
        root: The validated root directory to search.

    Returns:
        A sorted list of ``Path`` objects for every discovered image file.
    """
    indexed_root = root / INDEXED_DIR_NAME

    return sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not _is_inside(path, indexed_root)
        )
    )


def _is_inside(path: Path, directory: Path) -> bool:
    """Return True if *path* is located inside *directory*."""
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False
