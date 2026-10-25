from pydantic import BaseModel, Field
from typing import Literal, Optional

Category = Literal["photo", "screenshot", "document", "artwork", "meme", "infographic"]
Weather = Optional[Literal["sunny", "cloudy", "rainy", "snowy", "foggy"]]
TimeOfDay = Optional[Literal["morning", "afternoon", "evening", "night", "golden_hour"]]
Lighting = Optional[Literal["natural", "artificial", "mixed", "low_light"]]

class ImageAnalysis(BaseModel):
    """What the VLM returns for one image; its JSON schema is sent to Ollama.

    The description is asked for in sections, which the model fills in order;
    ``analyse_image`` joins them into the single stored ``description``.
    """
    short_description: str
    scene: str = Field(min_length=200, max_length=700)
    objects_and_layout: str = Field(min_length=200, max_length=700)
    people_details: str = Field(min_length=30, max_length=600)
    colors_light_mood: str = Field(min_length=100, max_length=500)
    primary_category: Category
    objects: list[str] = Field(max_length=15)
    people_count: int
    landmarks: list[str] = Field(max_length=3)
    weather: Weather
    time_of_day: TimeOfDay
    lighting: Lighting
    tags: list[str] = Field(max_length=10)

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

class RelevanceVerdict(BaseModel):
    matched_details: list[str]
    missing_requirements: list[str]
    is_match: bool
    confidence: float
    reasoning: str
