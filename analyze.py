#!/usr/bin/env python3

import json
import os
import time
import ollama

import config
from VLMResponseFormat import ImagePayload

TEMP_JSON_PATH: str = os.path.join(os.path.dirname(__file__), "temp.json")

IMAGE_SOURCE: str = "/Users/mdivakarnaidu/Desktop/Test Images/IMG_20260621_081122.jpg.jpeg"

def analyse_image(image_path: str) -> ImagePayload:
    client = ollama.Client(host=config.OLLAMA_HOST)
    print(f"[imagent] Sending image to Ollama model: {config.OLLAMA_MODEL} @ {config.OLLAMA_HOST} …")
    response = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": config.ANALYSIS_PROMPT,
            "images": [image_path],
        }],
        format=ImagePayload.model_json_schema(),
        options={"num_ctx": config.OLLAMA_NUM_CTX},
    )
    return ImagePayload.model_validate_json(response.message.content)



def save_json(payload: ImagePayload) -> None:
    with open(TEMP_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(payload.model_dump_json(indent=2))
    print(f"[imagent] Saved analysis to {TEMP_JSON_PATH}")

def main() -> None:
    start = time.perf_counter()
    payload = analyse_image(IMAGE_SOURCE)
    elapsed = time.perf_counter() - start
    print(f"[imagent] Analysis completed in {elapsed:.2f}s")
    save_json(payload)


if __name__ == "__main__":
    main()
