#!/bin/sh
# Imagent scan and index launcher for macOS and Linux: activates .venv and runs scan_and_index.py.
# Usage: sh scan_and_index.sh [folder]

DIR=$(cd "$(dirname "$0")" && pwd)

if [ ! -f "$DIR/.venv/bin/activate" ]; then
    echo "The .venv folder was not found. Run the installer first: sh install.sh" >&2
    exit 1
fi

. "$DIR/.venv/bin/activate"
exec python "$DIR/scan_and_index.py" "$@"
