#!/usr/bin/env python3
"""
analyze_image.py — run the production image analysis on a single image.

Asks for a local image path, then calls ``vlm.analyzer.analyse_image``
with the real ``analysis_prompt.txt`` against the Ollama host/model from
``~/.imagent/config.json`` and prints the structured response.

Usage:  python tests/analyze_image.py
"""

from _common import ask_image_path, run_analysis


def main() -> None:
    run_analysis(ask_image_path())


if __name__ == "__main__":
    main()
