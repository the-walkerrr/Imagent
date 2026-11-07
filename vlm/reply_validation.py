"""
vlm/reply_validation.py — Check the vision model's reply against the JSON schema, without knowing any field.

The schema (``config.ANALYSIS_SCHEMA``) is sent to Ollama, which already holds the reply to it while it is being
written, so what is checked here is what can still go wrong: a reply that was cut off, is not an object, lacks a
required field, has a field of the wrong type, or says nothing at all. A reply that passes is used exactly as the model
wrote it; nothing is trimmed, dropped or filled in.
"""

from __future__ import annotations

import json

_TYPES = {"string": str, "boolean": bool, "array": list, "object": dict, "null": type(None)}


def _type_matches(value, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _TYPES[name])


def _check(value, schema: dict, path: str, found: list[str]) -> None:
    expected = schema.get("type")
    if expected is not None:
        names = expected if isinstance(expected, list) else [expected]
        if not any(_type_matches(value, name) for name in names):
            found.append(f"{path}: expected {' or '.join(names)}")
            return
    if "enum" in schema and value not in schema["enum"]:
        found.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                found.append(f"{path}.{key}: missing")
        for key, sub_schema in schema.get("properties", {}).items():
            if key in value:
                _check(value[key], sub_schema, f"{path}.{key}", found)
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _check(item, schema["items"], f"{path}[{index}]", found)


def _says_nothing(value) -> bool:
    """True if no text is found anywhere in ``value``."""
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return all(_says_nothing(item) for item in value.values())
    if isinstance(value, list):
        return all(_says_nothing(item) for item in value)
    return True


def check_reply(text: str | None, schema: dict) -> tuple[dict | None, list[str]]:
    """Return ``(reply, [])`` if ``text`` is a JSON object that fits ``schema``, else ``(None, problems)``.

    ``problems`` say what is wrong, in words that can be sent back to the model.
    """
    try:
        value = json.loads(text or "")
    except ValueError as exc:
        return None, [f"reply: not valid JSON ({exc})"]
    found: list[str] = []
    if not isinstance(value, dict):
        return None, ["reply: expected a JSON object"]
    _check(value, schema, "reply", found)
    if not found and _says_nothing(value):
        found.append("reply: every field is empty")
    return (None if found else value), found
