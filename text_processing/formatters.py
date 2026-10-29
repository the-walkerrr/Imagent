from vlm.response_format import ImagePayload
from text_processing.cleaners import (
    normalize_whitespace,
    normalize_terms,
    lead_sentence,
    join_bullets,
    split_sections,
)

def format_objects(payload: ImagePayload, seen: set[str]) -> list[str]:
    lines: list[str] = []
    for item in payload.objects:
        name = normalize_whitespace(item)
        if not name:
            continue
        canonical = name.casefold()
        if canonical in seen:
            continue
        seen.add(canonical)
        lines.append(name)
    return lines

def format_people(payload: ImagePayload) -> list[str]:
    if payload.people_count <= 0:
        return []
    noun = "person" if payload.people_count == 1 else "people"
    return [f"{payload.people_count} {noun} visible"]

def format_scene(payload: ImagePayload) -> list[str]:
    items: list[str] = []
    if payload.weather:
        items.append(payload.weather.title())
    if payload.time_of_day:
        items.append(payload.time_of_day.replace("_", " ").title())
    if payload.lighting:
        lighting = payload.lighting.replace("_", " ").lower()
        items.append(f"{lighting} light")
    return items

def format_entities(payload: ImagePayload, seen: set[str]) -> list[str]:
    entities = normalize_terms(payload.landmarks)
    lines: list[str] = []
    for entity in entities:
        canonical = entity.casefold()
        if canonical in seen:
            continue
        seen.add(canonical)
        lines.append(entity)
    return lines

def format_ocr(payload: ImagePayload, seen: set[str]) -> list[str]:
    text_lines: list[str] = []
    for block in payload.visible_text:
        content = normalize_whitespace(block)
        if not content:
            continue
        canonical = content.casefold()
        if canonical in seen:
            continue
        seen.add(canonical)
        text_lines.append(content)
    return text_lines

def format_tags(payload: ImagePayload, seen: set[str]) -> list[str]:
    lines: list[str] = []
    for tag in normalize_terms(payload.tags):
        canonical = tag.casefold()
        if canonical in seen:
            continue
        seen.add(canonical)
        lines.append(tag)
    return lines

def build_search_text(payload: ImagePayload) -> str:
    seen: set[str] = set()
    sections: list[str] = []

    description = normalize_whitespace(payload.description)
    if description:
        sections.append(description)
        seen.add(description.casefold())

    short_description = lead_sentence(payload.short_description)
    if short_description and short_description.casefold() != description.casefold():
        sections.append(short_description)
        seen.add(short_description.casefold())

    objects = format_objects(payload, seen)
    if objects:
        sections.append("")
        sections.append("Objects:")
        sections.extend(join_bullets(objects))

    people = format_people(payload)
    if people:
        sections.append("")
        sections.append("People:")
        sections.extend(join_bullets(people))

    ocr = format_ocr(payload, seen)
    if ocr:
        sections.append("")
        sections.append("Visible text:")
        sections.extend(ocr)

    tags = format_tags(payload, seen)
    if tags:
        sections.append("")
        sections.append("Tags:")
        sections.extend(join_bullets(tags))

    scene = format_scene(payload)
    if scene:
        sections.append("")
        sections.append("Scene:")
        sections.extend(join_bullets(scene))

    entities = format_entities(payload, seen)
    if entities:
        sections.append("")
        sections.append("Landmarks:")
        sections.extend(join_bullets(entities))

    return split_sections(sections)
