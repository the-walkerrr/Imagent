#!/usr/bin/env python3
"""
show_failed/show_failed.py — Imagent: list the images that could not be indexed, with the reason for each and what can be done about it.

An image is listed only after every try for its kind of failure is over. It leaves the list by itself once it is indexed.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from imagent import config
    from imagent.db.connection import init_db
    from imagent.pipeline.console import make_output_safe, run_cli
    from imagent.scan_and_index.scan_and_index import print_failed
except ModuleNotFoundError as exc:
    sys.exit(f"Imagent is not set up yet (the Python package '{exc.name}' is missing). "
             f"Run: {'install.bat' if sys.platform == 'win32' else 'sh install.sh'}")


def main() -> None:
    make_output_safe()
    conn = init_db(str(config.DB_PATH))
    try:
        if not print_failed(conn, limit=None):
            print("No image has failed to index.")
    finally:
        conn.close()


if __name__ == "__main__":
    run_cli(main)
