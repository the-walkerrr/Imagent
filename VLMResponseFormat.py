from pydantic import BaseModel
from typing import Optional


class ObjectItem(BaseModel):
    name: str
    count: int
    location: Optional[str] = None  # e.g. center, top-left, foreground


class Person(BaseModel):
    actions: list[str]


class ImagePayload(BaseModel):
    # Descriptions
    short_description: str
    description: str

    # Classification
    primary_category: str  # photo, screenshot, document, artwork, meme, infographic
    content_type: str      # portrait, landscape, food, product, nature, etc.

    # Search metadata
    tags: list[str]
    entities: list[str]

    # Visual content
    objects: list[ObjectItem]
    people: list[Person]

    # OCR
    has_text: bool
    ocr_text: Optional[str] = None

    # Appearance
    dominant_colors: list[str]