#!/bin/sh
# Imagent launcher for macOS and Linux: asks whether to search, scan and index, or uninstall, then runs it.
# Search and scan run inside .venv. The uninstaller does not: it deletes .venv, so it picks its own Python.
# Usage: sh run.sh

DIR=$(cd "$(dirname "$0")" && pwd)

echo "What would you like to do?"
echo "  1) Search your images"
echo "  2) Scan folders and index images"
printf '\033[31m  3) Uninstall Imagent\033[0m\n'
while true; do
    printf "Choose 1, 2 or 3: "
    read -r choice || exit 1
    case "$choice" in
        1) script="search/search.py"; break ;;
        2) script="scan_and_index/scan_and_index.py"; break ;;
        3) exec sh "$DIR/uninstall/uninstall.sh" ;;
        *) echo "Please type 1, 2 or 3." ;;
    esac
done

if [ ! -f "$DIR/.venv/bin/activate" ]; then
    echo "The .venv folder was not found. Run the installer first: sh install.sh" >&2
    exit 1
fi

. "$DIR/.venv/bin/activate"
exec python "$DIR/$script"
