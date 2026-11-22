#!/usr/bin/env python3
"""
imagent/commands/forget.py — Imagent: delete what it remembers about pictures that were deleted, after two confirmations.

Run by ``imagent forget``, and offered by the menu when something is remembered. It deletes the kept analyses of deleted pictures
(``deleted_analyses``). The indexed pictures, the folders and the user's choice of how many deleted pictures to remember are not touched,
so pictures deleted from now on are remembered again, up to that number.
"""

from __future__ import annotations

import sys

from imagent import config
from imagent.db.connection import init_db
from imagent.db.operations import clear_deleted_analyses, count_deleted_analyses
from imagent.pipeline.console import make_output_safe
from imagent.pipeline.locking import AlreadyRunning, single_instance

CANCELLED = "Cancelled. Nothing was deleted."


def confirm_twice(remembered: int) -> bool:
    """Ask two different questions; any wrong answer cancels everything."""
    if input("\n  Confirmation 1 of 2. Do you want to forget everything above? Type yes: ").strip().lower() != "yes":
        return False
    if input("  Confirmation 2 of 2. How many deleted pictures will be forgotten? (see above): ").strip() != str(remembered):
        print("  That is not the number above.")
        return False
    return True


def main() -> None:
    make_output_safe()
    if not sys.stdin.isatty():
        sys.exit("Forgetting needs an interactive terminal. Run: imagent forget")

    conn = init_db(str(config.DB_PATH))
    try:
        remembered = count_deleted_analyses(conn)
        if not remembered:
            print("Imagent does not remember any deleted pictures.")
            return
        print("\n" + "=" * 64)
        print("  THIS DELETES WHAT IMAGENT REMEMBERS ABOUT DELETED PICTURES")
        print("=" * 64)
        print(f"  Deleted pictures remembered   {remembered}   (their descriptions and search data)")
        print("=" * 64)
        print("  If one of them comes back later, it will be analysed again. Your indexed pictures and your choice")
        print("  of how many deleted pictures to remember are not touched.")
        if not confirm_twice(remembered):
            sys.exit(f"\n{CANCELLED}")
        try:
            with single_instance("indexing", "Indexing"):  # a scan running now could remember more while this deletes
                with conn:
                    forgotten = clear_deleted_analyses(conn)
        except AlreadyRunning as exc:
            sys.exit(f"\n{exc} Nothing was deleted.")
        print(f"\n  ✓ Forgot {forgotten} deleted picture(s).")
    finally:
        conn.close()
