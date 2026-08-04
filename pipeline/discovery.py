"""pipeline/discovery.py — Recursively discover supported image files under a root directory."""

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


def discover_images(root: Path) -> list[Path]:
    """Return a sorted list of image paths found recursively under *root*.

    Args:
        root: The validated root directory to search.

    Returns:
        A sorted list of ``Path`` objects for every discovered image file.
    """
    return sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )
