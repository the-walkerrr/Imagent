import os
import json
import config
from VLMResponseFormat import ImagePayload
from text_processing.cleaners import slugify

ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESPONSES_DIR: str = os.path.join(ROOT_DIR, "Response")

def save_embedding_artifact(payload: ImagePayload, search_text: str, embedding: list[float]) -> None:
    model_dir = os.path.join(RESPONSES_DIR, slugify(config.OLLAMA_MODEL))
    os.makedirs(model_dir, exist_ok=True)
    filename = slugify(payload.short_description) + ".embedding.json"
    out_path = os.path.join(model_dir, filename)
    record = {
        "embedding_model": config.OLLAMA_EMBEDDING_MODEL,
        "search_text": search_text,
        "embedding": embedding,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=2))
    print(f"[imagent] Saved embedding artifact to {out_path}")

def save_json(payload: ImagePayload) -> None:
    model_dir = os.path.join(RESPONSES_DIR, slugify(config.OLLAMA_MODEL))
    os.makedirs(model_dir, exist_ok=True)
    filename = slugify(payload.short_description) + ".json"
    out_path = os.path.join(model_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(payload.model_dump_json(indent=2))
    print(f"[imagent] Saved analysis to {out_path}")
