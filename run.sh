#!/bin/sh
# Imagent launcher for macOS and Linux: opens a menu (arrow keys and Enter) to search, scan and index, see the images that
# could not be indexed, re-embed images (only offered when the embedding model changed), or uninstall.
# The menu and the commands run inside .venv. The uninstaller does not: it deletes .venv, so it picks its own Python.
# If Imagent is not installed yet, it offers to run the installer first.
# Usage: sh run.sh

DIR=$(cd "$(dirname "$0")" && pwd)
UNINSTALL_EXIT_CODE=75  # the same number as in menu/menu.py

if [ ! -f "$DIR/.venv/bin/activate" ]; then
    echo "Imagent is not installed yet (the .venv folder was not found)."
    while true; do
        printf "Install it now? (Y/N): "
        read -r answer || exit 1
        case "$answer" in
            [Yy] | [Yy][Ee][Ss]) break ;;
            [Nn] | [Nn][Oo])
                echo "Nothing was installed. Run the installer any time: sh install.sh"
                echo "To remove Imagent without it: sh uninstall/uninstall.sh"
                exit 1 ;;
            *) echo "Please type yes or no." ;;
        esac
    done
    sh "$DIR/install.sh" || exit $?
    if [ ! -f "$DIR/.venv/bin/activate" ]; then
        echo "The installer finished but the .venv folder is still missing." >&2
        exit 1
    fi
fi

( . "$DIR/.venv/bin/activate" && python "$DIR/menu/menu.py" )  # a subshell, so .venv is not active when the uninstaller starts
code=$?
if [ "$code" -eq "$UNINSTALL_EXIT_CODE" ]; then
    exec sh "$DIR/uninstall/uninstall.sh"
fi
exit "$code"
