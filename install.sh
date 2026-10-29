#!/bin/sh
# Imagent launcher for macOS and Linux: finds Python 3.9 or newer and runs install.py with it.
# It installs nothing; if Python is missing it says how to get it.
# Usage: sh install.sh

DIR=$(cd "$(dirname "$0")" && pwd)

for candidate in python3 python python3.13 python3.12 python3.11 python3.10 python3.9; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c "import sys; sys.exit(sys.version_info < (3, 9))" >/dev/null 2>&1; then
        exec "$candidate" "$DIR/install.py" "$@"
    fi
done

echo "Imagent needs Python 3.9 or newer, and none was found on this computer." >&2
echo "Install one, then run this again: sh install.sh" >&2
echo >&2
if [ "$(uname -s)" = "Darwin" ]; then
    echo "  macOS:  brew install python" >&2
    echo "          or download it from https://www.python.org/downloads/" >&2
    echo "  (If a window offered to install the 'command line developer tools', that is not enough" >&2
    echo "   on its own to get a recent Python; use one of the options above.)" >&2
else
    echo "  Debian / Ubuntu:  sudo apt install python3 python3-venv" >&2
    echo "  Fedora:           sudo dnf install python3" >&2
    echo "  Arch:             sudo pacman -S python" >&2
    echo "  or download it from https://www.python.org/downloads/" >&2
fi
exit 1
