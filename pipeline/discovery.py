"""pipeline/discovery.py — Recursively discover supported image files under a root directory."""

import os
from pathlib import Path
from typing import NamedTuple

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

# Folders that hold no pictures of the user's own, only copies, caches or system files. Names are compared
# case-insensitively; a folder whose name starts with "." (.git, .Trash, .thumbnails, ...) is skipped too,
# and so is one that ends in a suffix of SKIPPED_FOLDER_SUFFIXES (the photo libraries of macOS).
SKIPPED_FOLDERS: frozenset[str] = frozenset({
    "node_modules", "__pycache__",
    "@eadir",  # Synology thumbnails
    "$recycle.bin", "system volume information",  # Windows
    "__macosx",  # left behind by unzipping on a Mac
})
SKIPPED_FOLDER_SUFFIXES: tuple[str, ...] = (".photoslibrary", ".aplibrary", ".migratedphotolibrary")


class Discovery(NamedTuple):
    """What a folder walk found."""
    images: list[Path]  # sorted
    unreadable_dirs: list[Path]  # folders that could not be listed; what is in them is unknown, not gone
    bad_names: list[str]  # files whose name is not valid text (stored paths must be); shown with unprintable characters replaced


def _skip_folder(name: str) -> bool:
    lowered = name.lower()
    return name.startswith(".") or lowered in SKIPPED_FOLDERS or lowered.endswith(SKIPPED_FOLDER_SUFFIXES)


def _skip_file(name: str) -> bool:
    # "._photo.jpg" is the metadata companion that macOS writes next to "photo.jpg" on drives it cannot store it on; it is not an image.
    return name.startswith("._") or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS


def discover(root: Path) -> Discovery:
    """Walk *root* recursively and report its images, the folders it could not read and the files with unusable names."""
    images: list[Path] = []
    unreadable: list[Path] = []
    bad_names: list[str] = []

    def on_error(exc: OSError) -> None:
        if exc.filename:
            unreadable.append(Path(os.fsdecode(exc.filename)))

    for folder, subfolders, files in os.walk(root, onerror=on_error):
        subfolders[:] = sorted(name for name in subfolders if not _skip_folder(name))  # pruning in place steers the walk
        for name in files:
            if _skip_file(name):
                continue
            path = Path(folder) / name
            try:
                str(path).encode("utf-8")
            except UnicodeEncodeError:
                bad_names.append(os.fsdecode(path).encode("utf-8", "replace").decode("utf-8"))
                continue
            if path.is_file():  # not a broken link or a folder with an image-like name
                images.append(path)
    return Discovery(sorted(images), sorted(unreadable), sorted(bad_names))


def discover_images(root: Path) -> list[Path]:
    """Return a sorted list of image paths found recursively under *root* (see ``discover`` for what else a walk learns)."""
    return discover(root).images
