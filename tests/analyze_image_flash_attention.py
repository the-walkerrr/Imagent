#!/usr/bin/env python3
"""
analyze_image_flash_attention.py — single-image analysis with flash attention ON.

Flash attention is an Ollama *server* setting (OLLAMA_FLASH_ATTENTION=1), not a
per-request option. This script therefore starts a temporary local ``ollama serve``
on its own port with the flag set, runs the same analysis as analyze_image.py
against it, and shuts it down again. Your regular Ollama instance is untouched.

Requires the ``ollama`` CLI and the configured VLM model pulled locally.

Usage:  python tests/analyze_image_flash_attention.py
"""

import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

import ollama

from _common import ask_image_path, run_analysis  # sets up sys.path for ``config``
import config

FLASH_HOST = "127.0.0.1:11435"
STARTUP_TIMEOUT = 30


@contextmanager
def flash_attention_server():
    """Run ``ollama serve`` with flash attention enabled; yield its log file path."""
    env = {**os.environ, "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_HOST": FLASH_HOST}
    with tempfile.NamedTemporaryFile("w+", suffix=".log", prefix="ollama-flash-") as log:
        server = subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            client = ollama.Client(host=f"http://{FLASH_HOST}")
            deadline = time.monotonic() + STARTUP_TIMEOUT
            while True:
                try:
                    client.list()
                    break
                except Exception:
                    if server.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        sys.exit(f"Ollama server failed to start on {FLASH_HOST}:\n{log.read()}")
                    time.sleep(0.5)
            yield log.name
        finally:
            server.terminate()
            server.wait(timeout=10)


def print_server_log(log_path: str, keywords: tuple[str, ...], title: str) -> None:
    """Show the server log lines containing any of ``keywords`` (case-insensitive)."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip() for line in f if any(k in line.lower() for k in keywords)]
    print(f"\n{title}:")
    print("\n".join(f"  {line}" for line in lines) or "  (no matching lines found)")


def main() -> None:
    image_path = ask_image_path()
    with flash_attention_server() as log_path:
        print(f"Flash attention server running on {FLASH_HOST}")
        config.OLLAMA_HOST = f"http://{FLASH_HOST}"
        try:
            run_analysis(image_path)
        except Exception:
            print_server_log(
                log_path, ("error", "insufficient", "n_tokens", "flash", "-ub", "memory"), "Server log (failure details)"
            )
            raise
        print_server_log(log_path, ("flash",), "Server log (flash attention)")


if __name__ == "__main__":
    main()
