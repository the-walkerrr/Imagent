"""imagent/cli.py — The ``imagent`` command: read the command line and run the command it names.

``imagent`` with no command opens the menu. The menu always appears, set up or not; every command except ``setup`` and ``uninstall`` runs setup first if it has not been done (so a command picked in the menu does). Each command is a module in ``imagent/commands``; it is imported only when it is
run, so ``imagent --help`` and a mistyped command answer at once, without loading the image and model libraries.
"""

from __future__ import annotations

import argparse

from imagent import config
from imagent.pipeline.console import run_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imagent",
        description="Index your pictures and find them again by describing them. Everything runs on this computer. "
                    "Run it with no command to open a menu.")
    parser.add_argument("--version", action="version", version=f"imagent {config.APP_VERSION}")
    commands = parser.add_subparsers(dest="command", metavar="command", title="commands")

    search = commands.add_parser("search", help="find pictures by describing them", description="Show the pictures that best match a description.")
    search.add_argument("query", nargs="*", help="what to look for, for example: red car at night. "
                                                 "Leave it out to keep searching until you type exit")
    search.add_argument("--full", action="store_true", help="show each picture's full description instead of its one-line summary")

    scan = commands.add_parser("scan", help="scan the folders you added and index their pictures",
                               description="Scan every folder you added for new or changed pictures, then index them. "
                                           "Add a folder with: scan add \"folder path\".")
    scan_commands = scan.add_subparsers(dest="scan_command", metavar="command", title="commands")
    scan_add = scan_commands.add_parser("add", help="add a folder, then scan and index it",
                                        description="Add a folder to the folders that are scanned, scan it, then index its pictures. "
                                                    "A folder added before is scanned again.")
    scan_add.add_argument("folder", help="the folder to add")

    commands.add_parser("failed", help="list the pictures that could not be indexed",
                        description="List the pictures that could not be indexed, with the reason for each.")
    commands.add_parser("forget", help="forget the deleted pictures Imagent remembers",
                        description="Delete what Imagent remembers about pictures that were deleted (their descriptions and search data), after two confirmations. "
                                    "Your indexed pictures are not touched.")
    commands.add_parser("setup", help="choose the models and prepare this computer (once)",
                        description="Check Ollama, download and lock the three models, and measure this computer. "
                                    "Run it once after installing. Run again, it shows the models and restores a missing one.")
    commands.add_parser("uninstall", help="remove Imagent's data and models",
                        description="Delete the index, Imagent's data folder and the models it downloaded, after three confirmations.")
    return parser


def _set_up_first() -> None:
    """Run setup before the command if it has not been done, so the user is told at the start, not after the command has done its work.

    Setup ends the program if it fails or is cancelled, so the command only runs once Imagent is set up."""
    from imagent.commands import menu
    if menu.needs_setup():
        from imagent.commands import setup
        setup.say("Imagent is not set up yet, so setup starts now.")
        run_cli(setup.main, cancelled=setup.CANCELLED)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command not in (None, "setup", "uninstall"):  # the menu always appears; setup is the fix itself, and uninstall must work on a broken install
        _set_up_first()
    if args.command == "search":
        from imagent.commands import search
        run_cli(lambda: search.main(" ".join(args.query), full=args.full), cancelled="Search cancelled.")
    elif args.command == "scan":
        from imagent.commands import scan
        if args.scan_command == "add":
            run_cli(lambda: scan.add(args.folder))
        else:
            run_cli(scan.main)
    elif args.command == "failed":
        from imagent.commands import failed
        run_cli(failed.main)
    elif args.command == "forget":
        from imagent.commands import forget
        run_cli(forget.main, cancelled=forget.CANCELLED)
    elif args.command == "setup":
        from imagent.commands import setup
        run_cli(setup.main, cancelled=setup.CANCELLED)
    elif args.command == "uninstall":
        from imagent.commands import uninstall
        run_cli(uninstall.main, cancelled=uninstall.CANCELLED)
    else:
        from imagent.commands import menu
        run_cli(menu.main)
