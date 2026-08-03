#!/usr/bin/env python3

import time
from vlm.analyzer import analyse_image, embed_search_text
from text_processing.formatters import build_search_text
from storage.saver import save_json, save_embedding_artifact

IMAGE_SOURCE: str = "/Users/mdivakarnaidu/Downloads/WhatsApp Image 2026-07-05 at 11.52.39 AM.jpeg"

def main() -> None:
    start = time.perf_counter()
    payload = analyse_image(IMAGE_SOURCE)
    search_text = build_search_text(payload)
    embedding = embed_search_text(search_text)
    elapsed = time.perf_counter() - start
    print(f"[imagent] Analysis completed in {elapsed:.2f}s")
    save_json(payload)
    save_embedding_artifact(payload, search_text, embedding)


if __name__ == "__main__":
    main()

