#!/usr/bin/env python3

import io
import json
import os
import re
import time
import ollama
from PIL import Image

import config
from VLMResponseFormat import ImagePayload

RESPONSES_DIR: str = os.path.join(os.path.dirname(__file__), "Response")

IMAGE_SOURCE: str = "/Users/mdivakarnaidu/Downloads/WhatsApp Image 2026-07-05 at 11.52.39 AM.jpeg"

def preprocess_image(image_path: str) -> bytes:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > config.IMAGE_MAX_LONG_EDGE:
            scale = config.IMAGE_MAX_LONG_EDGE / long_edge
            new_w = (round(w * scale) // 28) * 28
            new_h = (round(h * scale) // 28) * 28
            new_w = max(new_w, 28)
            new_h = max(new_h, 28)
            img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=config.IMAGE_JPEG_QUALITY, optimize=True)
    return buf.getvalue()

def analyse_image(image_path: str) -> ImagePayload:
    client = ollama.Client(host=config.OLLAMA_HOST)
    image_bytes = preprocess_image(image_path)
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



def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:80]

def save_json(payload: ImagePayload) -> None:
    model_dir = os.path.join(RESPONSES_DIR, _slugify(config.OLLAMA_MODEL))
    os.makedirs(model_dir, exist_ok=True)
    filename = _slugify(payload.short_description) + ".json"
    out_path = os.path.join(model_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(payload.model_dump_json(indent=2))
    print(f"[imagent] Saved analysis to {out_path}")

def main() -> None:
    start = time.perf_counter()
    payload = analyse_image(IMAGE_SOURCE)
    elapsed = time.perf_counter() - start
    print(f"[imagent] Analysis completed in {elapsed:.2f}s")
    save_json(payload)


if __name__ == "__main__":
    main()
