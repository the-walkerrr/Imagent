from pydantic import BaseModel
from typing import Optional

class TextBlock(BaseModel):
    type: str
    content: str

class Person(BaseModel):
    age_group: str
    gender: str
    face_visible: bool
    actions: list[str]
    emotion: Optional[str]

class ObjectItem(BaseModel):
    name: str
    category: str
    count: int
    confidence: float

class ContentSafety(BaseModel):
    is_safe_for_work: bool
    adult: float
    violence: float
    medical: float

class StatItem(BaseModel):
    metric_or_subject: str
    value: str

class InfographicData(BaseModel):
    is_infographic: bool
    data_visualizations_present: list[str]
    statistics: list[StatItem]
    geographic_data: list[str]

class ImagePayload(BaseModel):
    short_description: str
    description: str
    primary_category: str
    content_type: str
    infographic_data: InfographicData
    content_safety: ContentSafety
    entities_brands: list[str]
    entities_landmarks: list[str]
    tags: list[str]
    search_terms: list[str]
    sparse_keywords: list[str]
    objects: list[ObjectItem]
    people: list[Person]
    text_blocks: list[TextBlock]
    has_text: bool
    weather: Optional[str]
    time_of_day: Optional[str]
    lighting: Optional[str]
    dominant_colors: list[str]
    orientation: str
    shot_type: str
    is_selfie: bool