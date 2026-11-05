#!/bin/sh
# Imagent launcher for macOS and Linux: asks whether to search, scan and index, retry images that could not be indexed,
# re-embed images, or uninstall, then runs it.
# Search and scan run inside .venv. The uninstaller does not: it deletes .venv, so it picks its own Python.
# Usage: sh run.sh

DIR=$(cd "$(dirname "$0")" && pwd)

echo "What would you like to do?"
echo "  1) Search your images"
echo "  2) Scan folders and index images"
echo "  3) Try again the images that could not be indexed"
echo "  4) Re-embed images (after the embedding model changed)"
printf '\033[31m  5) Uninstall Imagent\033[0m\n'
while true; do
    printf "Choose 1, 2, 3, 4 or 5: "
    read -r choice || exit 1
    case "$choice" in
        1) script="search/search.py"; break ;;
        2) script="scan_and_index/scan_and_index.py"; break ;;
        3) script="retry_failed/retry_failed.py"; break ;;
        4) script="reembed/reembed.py"; break ;;
        5) exec sh "$DIR/uninstall/uninstall.sh" ;;
        *) echo "Please type 1, 2, 3, 4 or 5." ;;
    esac
done

if [ ! -f "$DIR/.venv/bin/activate" ]; then
    echo "The .venv folder was not found. Run the installer first: sh install.sh" >&2
    exit 1
fi

. "$DIR/.venv/bin/activate"
exec python "$DIR/$script"
