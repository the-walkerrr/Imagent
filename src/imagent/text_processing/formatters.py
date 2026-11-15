import json

from text_processing.cleaners import (
    normalize_whitespace,
    normalize_terms,
    lead_sentence,
    join_bullets,
    split_sections,
)


SEARCH_TEXT_VERSION = 1  # raise it whenever build_search_text changes what it makes: stored with each record, it shows which are out of date


def _heading(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def _text(value) -> str:
    """A scalar or nested value as one line of text; empty if there is nothing to say."""
    if value is None or value == "" or value == [] or value == {}:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return normalize_whitespace(str(value))


def build_description(reply: dict, summary_field: str) -> str:
    """The prose of the model's reply: its text fields in the order the model wrote them, without the one-line summary."""
    texts = [value.strip() for key, value in reply.items() if key != summary_field and isinstance(value, str)]
    return "\n\n".join(text for text in texts if text)


def _unique(items: list, seen: set[str]) -> list[str]:
    """The items as text, each once; what is in ``seen`` is left out and what is kept is added to it."""
    kept: list[str] = []
    for item in normalize_terms([_text(item) for item in items]):
        canonical = item.casefold()
        if canonical not in seen:
            seen.add(canonical)
            kept.append(item)
    return kept


def build_search_text(reply: dict, ocr_text: list[str], summary_field: str) -> str:
    """The text that is embedded for one image, built from whatever fields the model's reply has.

    Text fields make up the description; the summary follows if it adds anything; each list field gets its own
    section named after it, any other field is one line of its own; the text read in the image comes last.
    """
    seen: set[str] = set()
    sections: list[str] = []

    description = normalize_whitespace(build_description(reply, summary_field))
    if description:
        sections.append(description)
        seen.add(description.casefold())

    summary = reply.get(summary_field)
    lead = lead_sentence(summary) if isinstance(summary, str) else ""
    if lead and lead.casefold() != description.casefold():
        sections.append(lead)
        seen.add(lead.casefold())

    for key, value in reply.items():
        if key == summary_field or isinstance(value, str):
            continue
        if isinstance(value, list):
            items = _unique(value, seen)
            if items:
                sections.extend(["", f"{_heading(key)}:", *join_bullets(items)])
        elif _text(value):
            sections.extend(["", f"{_heading(key)}: {_text(value)}"])

    ocr = _unique(ocr_text, seen)
    if ocr:
        sections.extend(["", "Visible text:", *ocr])

    return split_sections(sections)
