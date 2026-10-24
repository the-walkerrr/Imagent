from __future__ import annotations

import ollama
import config
from VLMResponseFormat import ImagePayload, QueryUnderstanding, RelevanceVerdict
from image_processing.preprocess import preprocess_image

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
        options={"num_ctx": config.OLLAMA_NUM_CTX, "use_mmap": config.OLLAMA_USE_MMAP},
    )
    return ImagePayload.model_validate_json(response.message.content)

def embed_search_text(search_text: str) -> list[float]:
    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.embed(
        model=config.OLLAMA_EMBEDDING_MODEL,
        input=[search_text],
        options={"use_mmap": config.OLLAMA_USE_MMAP},
    )
    return list(response.embeddings[0])

def understand_query(query: str) -> QueryUnderstanding:
    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": f"{config.QUERY_PROMPT}\n\nUser query: {query}",
        }],
        format=QueryUnderstanding.model_json_schema(),
        options={"num_ctx": config.OLLAMA_NUM_CTX, "use_mmap": config.OLLAMA_USE_MMAP, "temperature": 0},
    )
    return QueryUnderstanding.model_validate_json(response.message.content)

def judge_candidate(query: str, document: str, required_details: list[str] | None = None) -> RelevanceVerdict:
    checklist = "\n".join(f"- {detail}" for detail in required_details) if required_details else "(none extracted)"
    content = (
        f"{config.JUDGE_PROMPT}\n\n"
        f"User query: {query}\n\n"
        f"Checklist:\n{checklist}\n\n"
        f"Candidate record:\n{document}"
    )
    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": content}],
        format=RelevanceVerdict.model_json_schema(),
        options={"num_ctx": config.OLLAMA_NUM_CTX, "use_mmap": config.OLLAMA_USE_MMAP, "temperature": 0},
    )
    return RelevanceVerdict.model_validate_json(response.message.content)
