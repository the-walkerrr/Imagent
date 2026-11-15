#!/usr/bin/env python3
"""
imagent/commands/failed.py — Imagent: list the images that could not be indexed, with the reason for each and what can be done about it.

An image is listed only after every try for its kind of failure is over. It leaves the list by itself once it is indexed.
"""

from __future__ import annotations

from imagent import config
from imagent.db.connection import init_db
from imagent.pipeline.console import make_output_safe
from imagent.commands.scan import print_failed


def main() -> None:
    make_output_safe()
    conn = init_db(str(config.DB_PATH))
    try:
        if not print_failed(conn, limit=None):
            print("No image has failed to index.")
    finally:
        conn.close()

