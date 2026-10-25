from __future__ import annotations

import ollama
import config
from VLMResponseFormat import ImageAnalysis, ImagePayload, QueryUnderstanding, RelevanceVerdict
from image_processing.features import extract_image_features
from image_processing.ocr import extract_text_blocks
from image_processing.preprocess import preprocess_image
from text_processing.cleaners import normalize_terms

# Words in the description that back up a time_of_day value from the model.
_TIME_OF_DAY_WORDS = {
    "morning": ("morning", "sunrise", "dawn"),
    "afternoon": ("afternoon", "midday", "noon"),
    "evening": ("evening", "sunset", "dusk"),
    "night": ("night",),
    "golden_hour": ("golden hour",),
}

def analyse_image(image_path: str) -> ImagePayload:
    client = ollama.Client(host=config.OLLAMA_HOST)
    image_bytes = preprocess_image(image_path)
    print(f"[imagent] Sending image to Ollama model: {config.OLLAMA_MODEL} @ {config.OLLAMA_HOST} …")
    analysis = _request_analysis(client, image_bytes)
    return _build_payload(analysis, extract_text_blocks(image_path), extract_image_features(image_path))

def _request_analysis(client: ollama.Client, image_bytes: bytes) -> ImageAnalysis:
    """Ask the VLM to analyse the image, with every layer on the GPU first.

    If that attempt fails (an empty or invalid reply, or Ollama refusing the load, e.g. when the
    model does not fit in GPU memory), retry once with Ollama's own GPU/CPU split.
    """
    try:
        return _chat_for_analysis(client, image_bytes, {"num_gpu": config.OLLAMA_ALL_GPU_LAYERS})
    except (ValueError, ollama.ResponseError) as exc:
        print(f"[imagent] Analysis failed with all layers on the GPU ({type(exc).__name__}); retrying with Ollama's default split …")
        return _chat_for_analysis(client, image_bytes, {})

def _chat_for_analysis(client: ollama.Client, image_bytes: bytes, extra_options: dict) -> ImageAnalysis:
    response = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": config.ANALYSIS_PROMPT,
            "images": [image_bytes],
        }],
        format=ImageAnalysis.model_json_schema(),
        options={
            "num_ctx": config.OLLAMA_NUM_CTX,
            "use_mmap": config.OLLAMA_USE_MMAP,
            "num_predict": config.OLLAMA_MAX_OUTPUT_TOKENS,
            **extra_options,
        },
    )
    return ImageAnalysis.model_validate_json(response.message.content)

def _build_payload(analysis: ImageAnalysis, text_blocks: list[str], features: dict) -> ImagePayload:
    """Combine the VLM's answer, the OCR text and the pixel-derived facts into the stored record."""
    sections = [analysis.scene, analysis.objects_and_layout, analysis.people_details, analysis.colors_light_mood]
    description = "\n\n".join(section.strip() for section in sections if section.strip())

    # Drop values the description never states: the model sometimes guesses them.
    stated = description.lower()
    time_of_day = analysis.time_of_day
    if time_of_day and not any(word in stated for word in _TIME_OF_DAY_WORDS[time_of_day]):
        time_of_day = None
    weather = analysis.weather if analysis.weather and analysis.weather in stated else None

    if text_blocks:
        description += "\n\nText visible in the image: " + ", ".join(text_blocks)

    return ImagePayload(
        short_description=analysis.short_description.strip(),
        description=description,
        primary_category=analysis.primary_category,
        objects=normalize_terms(analysis.objects),
        people_count=analysis.people_count,
        landmarks=normalize_terms(analysis.landmarks),
        weather=weather,
        time_of_day=time_of_day,
        lighting=analysis.lighting,
        tags=normalize_terms(analysis.tags),
        visible_text=text_blocks,
        orientation=features["orientation"],
        dominant_colors=features["dominant_colors"],
    )

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
