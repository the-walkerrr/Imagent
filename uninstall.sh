#!/bin/sh
# Imagent uninstaller launcher for macOS and Linux: finds Python 3.9 or newer and runs the uninstaller with it.
# The uninstaller shows what will be lost and asks three times before it deletes anything.
# Usage: sh uninstall.sh

DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR" || exit 1

for candidate in python3 python python3.13 python3.12 python3.11 python3.10 python3.9; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c "import sys; sys.exit(sys.version_info < (3, 9))" >/dev/null 2>&1; then
        exec "$candidate" -B -m pipeline.uninstall "$@"
    fi
done

echo "The uninstaller needs Python 3.9 or newer, and none was found on this computer." >&2
echo "Install one, or remove Imagent by hand:" >&2
echo "  - delete the .venv folder in this project and the Imagent data folder (see the README)" >&2
echo "  - remove the Ollama models Imagent downloaded: ollama rm <name>" >&2
exit 1
