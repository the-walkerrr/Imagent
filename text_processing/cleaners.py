import re

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.casefold().strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item.strip())
    return result

def lead_sentence(text: str) -> str:
    normalized = normalize_whitespace(text)
    match = re.match(r"^(.+?[.!?])(?:\s|$)", normalized)
    if match:
        return match.group(1).strip()
    return normalized

def format_optional(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return normalize_whitespace(str(value)) or "None"

def normalize_terms(values: list[str]) -> list[str]:
    return dedupe_preserve_order([normalize_whitespace(value) for value in values if normalize_whitespace(value)])

def split_sections(lines: list[str]) -> str:
    cleaned = [line for line in lines if line and line.strip()]
    return "\n".join(cleaned).strip()

def join_bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items if item.strip()]

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:80]
