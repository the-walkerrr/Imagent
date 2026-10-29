from __future__ import annotations

import re
from typing import Callable

import ollama
import config
from pydantic import ValidationError
from vlm.response_format import ImageAnalysis, ImagePayload, QueryUnderstanding
from image_processing.features import extract_image_features
from image_processing.ocr import extract_text_blocks
from image_processing.preprocess import preprocess_image
from text_processing.cleaners import normalize_terms

# Words in the description that back up a time_of_day / weather value from the model.
_TIME_OF_DAY_WORDS = {
    "morning": ("morning", "sunrise", "dawn"),
    "afternoon": ("afternoon", "midday", "noon"),
    "evening": ("evening", "sunset", "dusk"),
    "night": ("night", "nighttime", "midnight"),
    "golden_hour": ("golden hour", "golden-hour"),
}
_WEATHER_WORDS = {
    "sunny": ("sunny", "sun", "sunlight", "sunshine", "clear sky", "clear skies"),
    "cloudy": ("cloudy", "cloud", "clouds", "overcast"),
    "rainy": ("rainy", "rain", "raining", "drizzle", "showers"),
    "snowy": ("snowy", "snow", "snowing", "snowfall"),
    "foggy": ("foggy", "fog", "mist", "misty", "haze", "hazy"),
}


def _is_stated(text: str, words: tuple[str, ...]) -> bool:
    """True if one of ``words`` appears in ``text`` as a whole word (so "night" does not match "nightstand")."""
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def analyse_image(client: ollama.Client, model: str, image_path: str,
                  on_stage: Callable[[str], None] = lambda stage: None) -> ImagePayload:
    """Analyse one image; ``on_stage`` is told what is being done, before each step."""
    on_stage("preparing image")
    image_bytes = preprocess_image(image_path)
    on_stage("analysing image")
    analysis = _request_analysis(client, model, image_bytes)
    on_stage("reading text in image")
    return _build_payload(analysis, extract_text_blocks(image_path), extract_image_features(image_path))

def _request_analysis(client: ollama.Client, model: str, image_bytes: bytes) -> ImageAnalysis:
    """Ask the model to describe the image; if its reply is not valid for the schema, tell it what was wrong and ask again.

    A reply can be cut off (it hit the output cap) or have the wrong shape. The asks after the first use a
    higher temperature, because at temperature 0 the same input would give the same bad reply.
    """
    messages = [
        {"role": "system", "content": config.ANALYSIS_PROMPT},
        {"role": "user", "content": "Analyze this image.", "images": [image_bytes]},
    ]
    for attempt in range(config.ANALYSIS_RETRIES + 1):
        options = {**config.LLM_OPTIONS, "num_predict": config.OLLAMA_MAX_OUTPUT_TOKENS}
        if attempt:
            options.update(temperature=config.ANALYSIS_RETRY_TEMPERATURE, seed=attempt)
        response = client.chat(model=model, messages=messages, format=ImageAnalysis.model_json_schema(), options=options)
        try:
            return ImageAnalysis.model_validate_json(response.message.content)
        except ValidationError as exc:
            if attempt == config.ANALYSIS_RETRIES:
                raise
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'reply'}: {e['msg']}" for e in exc.errors()[:3])
            messages = messages + [
                {"role": "assistant", "content": response.message.content},
                {"role": "user", "content": f"That reply was not valid ({problems[:300]}). Reply again with one complete, "
                                            "valid JSON object that follows the schema, and keep every text field short."},
            ]

def _cap_text_blocks(text_blocks: list[str]) -> list[str]:
    """Keep OCR text up to ``config.OCR_MAX_CHARS`` in total, so a text-heavy image cannot overflow the embedding model's context."""
    kept: list[str] = []
    remaining = config.OCR_MAX_CHARS
    for block in text_blocks:
        if remaining <= 0:
            break
        kept.append(block[:remaining])
        remaining -= len(block)
    return kept

def _build_payload(analysis: ImageAnalysis, text_blocks: list[str], features: dict) -> ImagePayload:
    """Combine the VLM's answer, the OCR text and the pixel-derived facts into the stored record."""
    sections = [analysis.scene, analysis.objects_and_layout, analysis.people_details, analysis.colors_light_mood]
    description = "\n\n".join(section.strip() for section in sections if section.strip())

    # Drop values the description never states: the model sometimes guesses them.
    stated = description.lower()
    time_of_day = analysis.time_of_day
    if time_of_day and not _is_stated(stated, _TIME_OF_DAY_WORDS[time_of_day]):
        time_of_day = None
    weather = analysis.weather
    if weather and not _is_stated(stated, _WEATHER_WORDS[weather]):
        weather = None

    # The OCR text is stored in visible_text; the search text lists it once, in its own section.
    text_blocks = _cap_text_blocks(text_blocks)

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

def embed_search_text(client: ollama.Client, model: str, search_text: str) -> list[float]:
    response = client.embed(
        model=model,
        input=[search_text],
        options=config.EMBED_OPTIONS,
    )
    return list(response.embeddings[0])

def understand_query(client: ollama.Client, model: str, query: str) -> QueryUnderstanding:
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": config.QUERY_PROMPT},
            {"role": "user", "content": f"User query: {query}"},
        ],
        format=QueryUnderstanding.model_json_schema(),
        options=config.LLM_OPTIONS,
    )
    return QueryUnderstanding.model_validate_json(response.message.content)
