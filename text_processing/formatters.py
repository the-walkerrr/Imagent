from VLMResponseFormat import ImagePayload
from text_processing.cleaners import (
    normalize_whitespace,
    normalize_terms,
    lead_sentence,
    join_bullets,
    split_sections,
)

def format_objects(payload: ImagePayload, seen: set[str]) -> list[str]:
    if not payload.objects:
        return []
    lines: list[str] = []
    for item in payload.objects:
        name = normalize_whitespace(item.name)
        if not name:
            continue
        canonical = name.casefold()
        if canonical in seen:
            continue
        seen.add(canonical)
        count_prefix = f"{item.count} " if item.count > 1 else ""
        lines.append(f"{count_prefix}{name}")
    return lines

def format_people(payload: ImagePayload, seen: set[str]) -> list[str]:
    if not payload.people:
        return []
    lines: list[str] = []
    for person in payload.people:
        sentence_parts: list[str] = []
        if person.age_group:
            sentence_parts.append(person.age_group)
        if person.gender and person.gender.lower() != "unknown":
            sentence_parts.append(person.gender)
        actions = normalize_terms(person.actions)
        if actions:
            sentence_parts.append("while " + ", ".join(actions))
        if person.emotion:
            sentence_parts.append(f"appearing {person.emotion}")
        sentence = normalize_whitespace(" ".join(sentence_parts)).strip(" .")
        canonical = sentence.casefold()
        if sentence and canonical not in seen:
            seen.add(canonical)
            lines.append(f"{sentence}.")
    return lines

def format_scene(payload: ImagePayload) -> list[str]:
    items: list[str] = []
    if payload.weather:
        items.append(payload.weather.title())
    if payload.time_of_day:
        items.append(payload.time_of_day.replace("_", " ").title())
    if payload.lighting:
        lighting = payload.lighting.replace("_", " ").lower()
        items.append(f"{lighting} light")
    if not items:
        return []
    return items

def format_entities(payload: ImagePayload, seen: set[str]) -> list[str]:
    entities = normalize_terms(payload.entities_brands + payload.entities_landmarks)
    lines: list[str] = []
    for entity in entities:
        canonical = entity.casefold()
        if canonical in seen:
            continue
        seen.add(canonical)
        lines.append(entity)
    return lines

def format_infographic(payload: ImagePayload, seen: set[str]) -> list[str]:
    info = payload.infographic_data
    if not info.is_infographic and not info.data_visualizations_present and not info.statistics and not info.geographic_data:
        return []
    lines: list[str] = []
    if info.data_visualizations_present:
        visualizations = normalize_terms(info.data_visualizations_present)
        if visualizations:
            lines.append("Infographic showing " + ", ".join(visualizations) + ".")
    for stat in info.statistics:
        item = f"{normalize_whitespace(stat.metric_or_subject)}: {normalize_whitespace(stat.value)}"
        canonical = item.casefold()
        if item and canonical not in seen:
            seen.add(canonical)
            lines.append(item)
    for place in normalize_terms(info.geographic_data):
        canonical = place.casefold()
        if canonical not in seen:
            seen.add(canonical)
            lines.append(place)
    return lines

def format_ocr(payload: ImagePayload, seen: set[str]) -> list[str]:
    if not payload.has_text:
        return []
    text_lines: list[str] = []
    for block in payload.text_blocks:
        content = normalize_whitespace(block.content)
        if not content:
            continue
        canonical = content.casefold()
        if canonical in seen:
            continue
        seen.add(canonical)
        text_lines.append(content)
    return text_lines

def format_search_terms(payload: ImagePayload, seen: set[str]) -> list[str]:
    lines: list[str] = []
    for term in normalize_terms(payload.search_terms):
        canonical = term.casefold()
        if canonical in seen:
            continue
        seen.add(canonical)
        lines.append(term)
    return lines

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

    description = lead_sentence(payload.description)
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

    people = format_people(payload, seen)
    if people:
        sections.append("")
        sections.append("People:")
        sections.extend(join_bullets(people))

    ocr = format_ocr(payload, seen)
    if ocr:
        sections.append("")
        sections.append("Visible text:")
        sections.extend(ocr)

    search_terms = format_search_terms(payload, seen)
    if search_terms:
        sections.append("")
        sections.append("Search terms:")
        sections.extend(join_bullets(search_terms))

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
        sections.append("Brands and landmarks:")
        sections.extend(join_bullets(entities))

    infographic = format_infographic(payload, seen)
    if infographic:
        sections.append("")
        sections.append("Infographic:")
        sections.extend(join_bullets(infographic))

    return split_sections(sections)
