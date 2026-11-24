"""imagent/cli.py — The ``imagent`` command: read the command line and run the command it names.

``imagent`` with no command opens the menu. The menu, like every command except ``setup``, ``uninstall`` and ``help``, runs setup first if it has not been done, so it only opens once Imagent is set up. Each command is a module in ``imagent/commands``; it is imported only when it is
run, so ``imagent --help`` and a mistyped command answer at once, without loading the image and model libraries.
"""

from __future__ import annotations

import argparse
import sys

from imagent import config
from imagent.pipeline.console import leave_a_gap, run_cli


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
    detail = search.add_mutually_exclusive_group()
    detail.add_argument("--full", dest="full", action="store_const", const=True, default=None,
                        help="show each picture's full description instead of its one-line summary (for this search; the usual one is set in settings)")
    detail.add_argument("--short", dest="full", action="store_const", const=False, default=None,
                        help="show each picture's one-line summary instead of its full description (for this search; the usual one is set in settings)")

    scan = commands.add_parser("scan", help="scan the folders you added and index their pictures",
                               description="Scan every folder you added for new or changed pictures, then index them. "
                                           "Add a folder with: add \"folder path\".")
    add = commands.add_parser("add", help="add a folder, then scan and index it",
                              description="Add a folder to the folders that are scanned, scan it, then index its pictures. "
                                          "A folder added before is scanned again.")
    add.add_argument("folder", nargs="?", help="the folder to add (without it you pick one from a list of suggestions)")

    commands.add_parser("failed", help="list the pictures that could not be indexed",
                        description="List the pictures that could not be indexed, with the reason for each.")
    commands.add_parser("forget", help="forget the deleted pictures Imagent remembers",
                        description="Delete what Imagent remembers about pictures that were deleted (their descriptions and search data), after two confirmations. "
                                    "Your indexed pictures are not touched.")
    delete = commands.add_parser("delete", help="remove folders from the scan list, and what was indexed from them",
                                 description="Remove folders from the scan list and delete what was indexed from them, after showing what will be lost and two questions. "
                                             "Your pictures are never touched. You can choose to keep the analysis, so adding a folder again does not analyse its pictures again.")
    delete.add_argument("folder", nargs="*", help="the folders to remove from the scan list; put a path with spaces in quotes (without any, you choose from the folders you added)")
    commands.add_parser("settings", help="change how Imagent behaves, or uninstall it",
                        description="Open a menu to change how many deleted pictures Imagent remembers, whether a search shows the summary or the full description, "
                                    "and how long the scan waits before it runs again. It also holds Uninstall.")
    commands.add_parser("setup", help="choose the models and prepare this computer (once)",
                        description="Check Ollama, download and lock the three models, and measure this computer. "
                                    "Run it once after installing. Run again, it shows the models and restores a missing one.")
    commands.add_parser("help", help="explain each command briefly",
                        description="Explain each command briefly, with an example of how to run it.")
    commands.add_parser("uninstall", help="remove Imagent's data and models",
                        description="Delete the index, Imagent's data folder and the models it downloaded, after three confirmations.")
    return parser


def _set_up_first() -> None:
    """Run setup before the command (or the menu) if it has not been done, so the user is told at the start, not after the command has done its work.

    Setup ends the program if it fails or is cancelled, so the command or menu only runs once Imagent is set up."""
    from imagent.commands import menu
    if menu.needs_setup():
        from imagent.commands import setup
        setup.say("Imagent is not set up yet, so setup starts now.")
        run_cli(setup.main, cancelled=setup.CANCELLED)


def main(argv: list[str] | None = None) -> None:
    """Run the command, and however it ends (done, a message, Ctrl+C, ``--help``) leave a blank line before the shell's prompt."""
    try:
        _run(argv)
    except SystemExit as stop:
        if sys.stdout.isatty() and isinstance(stop.code, str):  # Python would print this message after the gap, so it is printed here, before it
            print(stop.code, file=sys.stderr)
            stop = SystemExit(1)
        leave_a_gap()
        raise stop
    leave_a_gap()


def _run(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "add" and not args.folder and sys.stdin.isatty():
        from imagent.pipeline import folder_picker
        folder_picker.start_listing()  # the folders are listed while setup (if it runs) and the user's typing go on
    if args.command not in ("setup", "uninstall", "help"):  # setup is the fix itself, uninstall must work on a broken install, and help needs nothing
        _set_up_first()
    if args.command == "search":
        from imagent.commands import search
        run_cli(lambda: search.main(" ".join(args.query), full=args.full), cancelled="Search cancelled.")
    elif args.command == "add":
        from imagent.commands import scan
        run_cli(lambda: scan.add(args.folder))
    elif args.command == "scan":
        from imagent.commands import scan
        run_cli(scan.main)
    elif args.command == "delete":
        from imagent.commands import delete
        run_cli(lambda: delete.main(args.folder), cancelled=delete.CANCELLED)
    elif args.command == "failed":
        from imagent.commands import failed
        run_cli(failed.main)
    elif args.command == "forget":
        from imagent.commands import forget
        run_cli(forget.main, cancelled=forget.CANCELLED)
    elif args.command == "settings":
        from imagent.commands import settings
        run_cli(settings.main, cancelled=settings.CANCELLED)
    elif args.command == "setup":
        from imagent.commands import setup
        run_cli(setup.main, cancelled=setup.CANCELLED)
        from imagent.commands import menu  # setup ends the program if it fails or is cancelled, so the menu opens only once it is done
        run_cli(menu.main)
    elif args.command == "help":
        from imagent.commands import help as show_help
        run_cli(show_help.main)
    elif args.command == "uninstall":
        from imagent.commands import uninstall
        run_cli(uninstall.main, cancelled=uninstall.CANCELLED)
    else:
        from imagent.commands import menu
        run_cli(menu.main)
