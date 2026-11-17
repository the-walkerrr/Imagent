"""imagent/cli.py — The ``imagent`` command: read the command line and run the command it names.

``imagent`` with no command opens the menu, after running setup first if it has not been done. Each command is a module in ``imagent/commands``; it is imported only when it is
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

    scan = commands.add_parser("scan", help="scan a folder and index its pictures",
                               description="Scan a folder for new or changed pictures, then index them.")
    scan.add_argument("folder", nargs="?", help="the folder to scan. Leave it out to choose one")

    commands.add_parser("failed", help="list the pictures that could not be indexed",
                        description="List the pictures that could not be indexed, with the reason for each.")
    commands.add_parser("setup", help="choose the models and prepare this computer (once)",
                        description="Check Ollama, download and lock the three models, and measure this computer. "
                                    "Run it once after installing. Run again, it shows the models and restores a missing one.")
    commands.add_parser("uninstall", help="remove Imagent's data and models",
                        description="Delete the index, Imagent's data folder and the models it downloaded, after three confirmations.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "search":
        from imagent.commands import search
        run_cli(lambda: search.main(" ".join(args.query)), cancelled="Search cancelled.")
    elif args.command == "scan":
        from imagent.commands import scan
        run_cli(lambda: scan.main(args.folder))
    elif args.command == "failed":
        from imagent.commands import failed
        run_cli(failed.main)
    elif args.command == "setup":
        from imagent.commands import setup
        run_cli(setup.main, cancelled=setup.CANCELLED)
    elif args.command == "uninstall":
        from imagent.commands import uninstall
        run_cli(uninstall.main, cancelled=uninstall.CANCELLED)
    else:
        from imagent.commands import menu
        if menu.needs_setup():
            from imagent.commands import setup
            setup.say("Imagent is not set up yet, so setup starts now.")
            run_cli(setup.main, cancelled=setup.CANCELLED)  # ends the program if setup fails or is cancelled
        run_cli(menu.main)
