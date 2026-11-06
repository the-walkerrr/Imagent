"""Shared helpers: every test runs against its own temporary Imagent folder, never the real one."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import config
from db.connection import init_db
from image_processing.hashing import compute_hash
from image_processing.metadata import iso_utc


# What setup measured about a computer (db/operations.set_device_settings).
DEVICE = {"num_ctx": 8192, "use_mmap": False, "image_long_edge": 1024, "timeout_seconds": 300, "measurements": {}}


class IsolatedHome(unittest.TestCase):
    """A test case whose database, config.json and lock files live in a temporary folder."""

    def setUp(self):
        # A run puts the measured settings in place of config's starting values (config.apply_device_settings); undo that afterwards.
        for name in ("OLLAMA_TIMEOUT", "OLLAMA_NUM_CTX", "OLLAMA_USE_MMAP", "IMAGE_MAX_LONG_EDGE"):
            patcher = mock.patch.object(config, name, getattr(config, name))
            patcher.start()
            self.addCleanup(patcher.stop)
        for options in (config.LLM_OPTIONS, config.EMBED_OPTIONS):
            patcher = mock.patch.dict(options)
            patcher.start()
            self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name) / "imagent-home"
        self.work = Path(tmp.name) / "work"
        self.work.mkdir()
        for name, value in (("IMAGENT_HOME", self.home), ("DB_PATH", self.home / "imagent.db"),
                            ("CONFIG_PATH", self.home / "config.json")):
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def connect(self):
        conn = init_db(str(config.DB_PATH))
        self.addCleanup(conn.close)
        return conn

    def make_image(self, path: Path, shade: int = 0) -> Path:
        """Write a small valid image; a different ``shade`` gives different content (and so a different hash)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (shade % 256, (shade // 256) % 256, 7)).save(path)
        return path


def add_indexed(conn, root_id: int, path: Path) -> None:
    """Record ``path`` as indexed, the way a finished index run leaves it (no model involved)."""
    with conn:
        conn.execute(
            "INSERT INTO image_data(root_id, file_path, hash, vlm_metadata_json, modified_at) VALUES (?, ?, ?, ?, ?)",
            (root_id, str(path), compute_hash(str(path)), "{}", iso_utc(os.stat(path).st_mtime)),
        )
