#!/bin/sh
# Imagent search launcher for macOS and Linux: activates .venv and runs search.py.
# Usage: sh search.sh [query]

DIR=$(cd "$(dirname "$0")" && pwd)

if [ ! -f "$DIR/.venv/bin/activate" ]; then
    echo "The .venv folder was not found. Run the installer first: sh install.sh" >&2
    exit 1
fi

. "$DIR/.venv/bin/activate"
exec python "$DIR/search.py" "$@"
