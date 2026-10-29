from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional

Category = Literal["photo", "screenshot", "document", "artwork", "meme", "infographic"]
Weather = Optional[Literal["sunny", "cloudy", "rainy", "snowy", "foggy"]]
TimeOfDay = Optional[Literal["morning", "afternoon", "evening", "night", "golden_hour"]]
Lighting = Optional[Literal["natural", "artificial", "mixed", "low_light"]]

# Sizes the model is asked to keep to (they are in the JSON schema sent to Ollama). A reply that goes over is
# trimmed, not rejected: a valid image must not fail because the model wrote one sentence too many.
_TEXT_LIMITS = {"scene": 700, "objects_and_layout": 700, "people_details": 600, "colors_light_mood": 500}
_LIST_LIMITS = {"objects": 15, "landmarks": 3, "tags": 10}


def _shorten(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters, at the end of a sentence (or else a word) when there is one."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sentence_end = max(cut.rfind(mark) for mark in ".!?")
    if sentence_end >= limit // 2:
        return cut[: sentence_end + 1]
    word_end = cut.rfind(" ")
    return cut[:word_end] if word_end >= limit // 2 else cut


class ImageAnalysis(BaseModel):
    """What the VLM returns for one image; its JSON schema is sent to Ollama.

    The description is asked for in sections, which the model fills in order;
    ``analyse_image`` joins them into the single stored ``description``.
    Texts and lists longer than the limits are trimmed; short ones are accepted
    ("No people are visible." is a valid answer), but not all four sections empty.
    """
    short_description: str
    scene: str = Field(max_length=_TEXT_LIMITS["scene"])
    objects_and_layout: str = Field(max_length=_TEXT_LIMITS["objects_and_layout"])
    people_details: str = Field(max_length=_TEXT_LIMITS["people_details"])
    colors_light_mood: str = Field(max_length=_TEXT_LIMITS["colors_light_mood"])
    primary_category: Category
    objects: list[str] = Field(max_length=_LIST_LIMITS["objects"])
    people_count: int
    landmarks: list[str] = Field(max_length=_LIST_LIMITS["landmarks"])
    weather: Weather
    time_of_day: TimeOfDay
    lighting: Lighting
    tags: list[str] = Field(max_length=_LIST_LIMITS["tags"])

    @field_validator("scene", "objects_and_layout", "people_details", "colors_light_mood", mode="before")
    @classmethod
    def _trim_text(cls, value, info):
        return _shorten(value, _TEXT_LIMITS[info.field_name]) if isinstance(value, str) else value

    @field_validator("objects", "landmarks", "tags", mode="before")
    @classmethod
    def _trim_list(cls, value, info):
        return value[: _LIST_LIMITS[info.field_name]] if isinstance(value, list) else value

    @model_validator(mode="after")
    def _has_description(self):
        if not any(section.strip() for section in (self.scene, self.objects_and_layout, self.people_details, self.colors_light_mood)):
            raise ValueError("all four description sections are empty")
        return self

class ImagePayload(BaseModel):
    """Everything stored and indexed for one image, after post-processing."""
    short_description: str
    description: str
    primary_category: Category
    objects: list[str]
    people_count: int
    landmarks: list[str]
    weather: Weather
    time_of_day: TimeOfDay
    lighting: Lighting
    tags: list[str]
    visible_text: list[str]
    orientation: str
    dominant_colors: list[str]

class QueryUnderstanding(BaseModel):
    normalized_query: str
    required_details: list[str]
    hypothetical_description: str
