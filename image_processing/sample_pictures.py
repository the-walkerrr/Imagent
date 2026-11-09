"""
image_processing/sample_pictures.py — The pictures setup checks this computer with.

Real photos of many kinds and sizes ship as one zip (tests/fixtures/images.zip, with a manifest of where each came from and
its licence). What no zip should hold is made here instead: pictures far larger than anyone would commit, and files that are
damaged or odd in their way of storing colour. How many are made, and how big, follows the memory and disk this computer has.

``check_pictures`` then runs the steps of indexing that need no model over all of them and says how each one went.
"""

from __future__ import annotations

import io
import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

import config
from image_processing.memory import available_memory, check_image_size, peak_bytes_per_pixel
from image_processing.preprocess import preprocess_image
from image_processing.problems import BUSY, TOO_LARGE, UNREADABLE, UNSUPPORTED, ImageProblem, explain

ZIP_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "images.zip"
TARGET_BYTES = 120_000_000  # the size of the largest picture made: the range the pictures cover runs from small up to this
PROBE_PIXELS = 4_000_000  # a picture this size is made first, to learn how many bytes a pixel of such a picture takes


def unpack(folder: Path) -> int:
    """Extract the zip into ``folder``, leaving files already there as they are. Returns how many pictures it holds."""
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = [name for name in archive.namelist() if name == os.path.basename(name)]  # a flat folder: nothing can land elsewhere
        for name in names:
            target = folder / name
            if not target.exists() or target.stat().st_size != archive.getinfo(name).file_size:
                with archive.open(name) as source, open(target, "wb") as handle:
                    shutil.copyfileobj(source, handle)
    return sum(1 for name in names if name != "manifest.json")


def _noise_jpeg(pixels: int, quality: int = 92) -> bytes:
    """A JPEG of about ``pixels`` pixels that is hard to compress (smooth noise), so its size follows its pixel count."""
    side = max(int(pixels ** 0.5), 8)
    rng = np.random.default_rng(pixels)
    small = rng.integers(0, 256, (side // 2, side // 2, 3), dtype=np.uint8)
    picture = Image.fromarray(small).resize((side, side), Image.BICUBIC)
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _fits(pixels: int, folder: Path) -> str | None:
    """None if a picture of this many pixels can be made here, else why not (memory to build it, disk to keep it)."""
    available = available_memory()
    if available is not None and pixels * 8 > available * config.IMAGE_MEMORY_SHARE:
        return "not enough free memory to make it"
    if shutil.disk_usage(folder).free < pixels * 4 + TARGET_BYTES:
        return "not enough free disk space to keep it"
    return None


def make_extremes(folder: Path, target_bytes: int = TARGET_BYTES, say: Callable[[str], None] = lambda m: None) -> dict:
    """Make the pictures no zip should hold. Returns {file name: what reading it should come to} for the ones made.

    The huge photographs are an eighth, half and all of ``target_bytes``; any this computer has no room for is left out, and said so."""
    folder.mkdir(parents=True, exist_ok=True)
    made: dict[str, str] = {}

    probe = len(_noise_jpeg(PROBE_PIXELS))
    for share, label in ((8, "large"), (2, "very-large"), (1, "huge")):
        pixels = int(target_bytes / share / probe * PROBE_PIXELS)
        reason = _fits(pixels, folder)
        if reason:
            say(f"Skipped the {label} photograph ({pixels / 1e6:.0f} megapixels): {reason}.")
            continue
        name = f"extreme-{label}.jpg"
        (folder / name).write_bytes(_noise_jpeg(pixels))
        made[name] = "read"  # read or too large for the free memory, but never an unexpected error

    # A small file that expands to twice the pixels the memory check allows, kept as one bit a pixel so that making it is cheap.
    available = available_memory()
    if available:
        side = int((2 * available * config.IMAGE_MEMORY_SHARE / peak_bytes_per_pixel("1")) ** 0.5)
        if _fits(side * side // 8, folder) is None:
            Image.new("1", (side, side)).save(folder / "extreme-expands-to-many-pixels.png")
            made["extreme-expands-to-many-pixels.png"] = "too large"

    good = io.BytesIO()
    Image.fromarray(np.random.default_rng(1).integers(0, 256, (600, 800, 3), dtype=np.uint8)).save(good, format="JPEG")
    jpeg = good.getvalue()
    (folder / "damaged-cut-off.jpg").write_bytes(jpeg[: len(jpeg) * 6 // 10])
    (folder / "damaged-empty.jpg").write_bytes(b"")
    (folder / "damaged-not-a-picture.png").write_bytes(b"This is text, not a picture.\n" * 40)
    made.update({"damaged-cut-off.jpg": "damaged", "damaged-empty.jpg": "damaged", "damaged-not-a-picture.png": "damaged"})

    odd = {
        "odd-transparent.png": Image.new("RGBA", (400, 300), (0, 0, 0, 0)),
        "odd-sixteen-bit.png": Image.fromarray((np.arange(300 * 400, dtype=np.uint32).reshape(300, 400) % 65536).astype(np.uint16)),
        "odd-cmyk.jpg": Image.new("CMYK", (400, 300), (10, 80, 200, 20)),
        "odd-panorama.jpg": Image.new("RGB", (12000, 300), (90, 140, 200)),
        "odd-one-pixel.png": Image.new("RGB", (1, 1), (255, 0, 0)),
    }
    for name, picture in odd.items():
        picture.save(folder / name)
        made[name] = "read"
    return made


def check_pictures(folder: Path, expected: dict | None = None, say: Callable[[str], None] = lambda m: None) -> dict:
    """Run the steps of indexing that need no model over every picture in ``folder``. Returns the number of pictures per outcome.

    The outcomes: ``read``, ``too large`` for the
    memory this computer has free, ``damaged``, ``unsupported``, ``busy``, and ``unexpected`` for an error that should have been
    one of those. Where ``expected`` says what a picture should come to, any other outcome is said."""
    outcomes: dict[str, int] = {}
    for path in sorted(p for p in folder.iterdir() if p.is_file() and p.name != "manifest.json"):
        try:
            check_image_size(str(path))
            try:
                preprocess_image(str(path))
            except Exception as exc:  # as vlm.analyzer.analyse_image does: an error of the picture becomes its explanation
                raise (explain(exc, str(path)) or exc) from exc
            outcome = "read"
        except ImageProblem as problem:
            outcome = {TOO_LARGE: "too large", UNREADABLE: "damaged", UNSUPPORTED: "unsupported", BUSY: "busy"}.get(problem.kind, "unexpected")
        except Exception as exc:  # an error that no step of indexing turns into a plain explanation
            outcome = "unexpected"
            say(f"{path.name}: {type(exc).__name__}: {exc}")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        wanted = (expected or {}).get(path.name)
        if wanted and outcome not in _accepted(wanted):
            say(f"{path.name} came to '{outcome}', not '{wanted}'.")
    return outcomes


def _accepted(wanted: str) -> tuple:
    """What a picture meant to be read may come to: read, or too large for the memory this computer has free."""
    return ("read", "too large") if wanted == "read" else (wanted,)
