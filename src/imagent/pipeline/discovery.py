"""imagent/pipeline/discovery.py — Recursively discover supported image files under a root directory."""

import os
from pathlib import Path
from typing import NamedTuple

from imagent.os_specific.files import is_hidden_folder

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
# case-insensitively; a folder whose name starts with "." (.git, .Trash, .thumbnails, ...) is skipped too, and so is
# one the operating system itself marks as hidden (AppData on Windows, ~/Library on macOS: see os_specific/files.py),
# and one that ends in a suffix of SKIPPED_FOLDER_SUFFIXES (program bundles and the photo libraries and previews of photo programs).
# The names are listed even where the hidden mark would catch them, because a disk written by Windows and read on a Mac or Linux
# computer shows no such mark. A name that a person could also give to a folder of their own pictures ("windows", "recovery",
# "build", "venv") is left out on purpose.
SKIPPED_FOLDERS: frozenset[str] = frozenset({
    "node_modules", "__pycache__", "site-packages", "bower_components",  # what programs install: Python and JavaScript packages
    "@eadir", "#recycle", "#snapshot", "@recycle",  # Synology thumbnails, recycle bin and snapshots; QNAP recycle bin
    "$recycle.bin", "recycler", "system volume information",  # Windows: recycle bins and restore points
    "$windows.~bt", "$windows.~ws", "$getcurrent", "$sysreset", "$winreagent", "windows.old",  # Windows: upgrade, reset and recovery leftovers
    "program files", "program files (x86)", "programdata",  # Windows: installed programs
    "__macosx",  # left behind by unzipping on a Mac
})
# ".app": a macOS program (its icons and pictures are the program's); ".lrdata": Lightroom's previews;
# the others are libraries that hold their pictures in a package of their own.
SKIPPED_FOLDER_SUFFIXES: tuple[str, ...] = (".photoslibrary", ".aplibrary", ".migratedphotolibrary", ".photolibrary", ".lrdata", ".app")


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


def list_folders(root: str) -> list[str]:
    """Every folder below *root* that a scan would look inside, in no particular order; unreadable folders are left out.

    It prunes with the same rules as ``discover``, so a folder is listed exactly when a scan of its parent would reach it. A link to a
    folder is listed but not followed, as ``discover`` does not follow it. The folders are read from the disk itself, so none is
    missing or out of date."""
    found: list[str] = []
    pending = [root]
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    try:
                        if not entry.is_dir() or _skip_folder(entry.name) or is_hidden_folder(entry.path):
                            continue
                        found.append(entry.path)
                        if not entry.is_symlink():
                            pending.append(entry.path)
                    except OSError:
                        continue  # gone while being read
        except OSError:
            continue  # not readable: what is in it is unknown
    return found


def discover(root: Path) -> Discovery:
    """Walk *root* recursively and report its images, the folders it could not read and the files with unusable names."""
    images: list[Path] = []
    unreadable: list[Path] = []
    bad_names: list[str] = []

    def on_error(exc: OSError) -> None:
        if exc.filename:
            unreadable.append(Path(os.fsdecode(exc.filename)))

    for folder, subfolders, files in os.walk(root, onerror=on_error):
        subfolders[:] = sorted(name for name in subfolders  # pruning in place steers the walk
                               if not _skip_folder(name) and not is_hidden_folder(os.path.join(folder, name)))
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
