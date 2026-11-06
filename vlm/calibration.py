"""
vlm/calibration.py — Measure this computer with the vision model, so the settings that depend on it are measured, not guessed.

Run once by setup (and again when the vision model changed). It sends the model the real prompt and schema with a picture
made here (a scene drawn with Pillow, so nothing is downloaded or shipped), at smaller and smaller sizes, until one works:

  * image_long_edge: the largest size, up to ``config.IMAGE_EDGE_CEILING``, that the computer handles (a trial that runs out
    of memory, crashes or never answers goes on to the next smaller size);
  * num_ctx: the smallest context window (a power of two) that holds the prompt, that image, and a reply with room to raise
    the output cap once; a larger window only costs memory;
  * use_mmap: whether the model is paged in from disk. Loading it fully is tried first when it fits in the free memory with
    room to spare, paging otherwise; when a way fails (out of memory, or an Ollama that answers nothing in it) the other is tried;
  * timeout_seconds: the time the largest first reply needs at the speed that was measured, times a margin.

``calibrate`` raises ``NotPossible`` when not even the smallest size works: the model does not run on this computer.
"""

from __future__ import annotations

import io
import math
import random
from typing import Callable

import ollama
from PIL import Image, ImageDraw

import config
from image_processing.memory import available_memory
from vlm.analyzer import OUT_OF_MEMORY, ServiceTrouble, send_request
from vlm.client import unload_model


class NotPossible(Exception):
    """The vision model cannot run on this computer; the message says why, in plain words."""


class NoAnswer(Exception):
    """Ollama accepted the request and returned nothing: no text and no timings, and no error."""


def calibration_image(edge: int) -> bytes:
    """A JPEG of a made-up scene whose longer side is ``edge``: a gradient, shapes of many sizes and some text.

    What a picture shows does not change what it costs the model to read it; its size does."""
    width, height = edge, max(edge * 3 // 4, 1)
    rng = random.Random(edge)  # the same picture for the same size
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        shade = 60 + 150 * y // height
        draw.line([(0, y), (width, y)], fill=(shade, 120 + shade // 3, 255 - shade))
    for _ in range(40):
        x, y = rng.randrange(width), rng.randrange(height)
        size = rng.randrange(max(edge // 40, 2), max(edge // 5, 4))
        colour = tuple(rng.randrange(256) for _ in range(3))
        box = [x, y, x + size, y + size]
        draw.ellipse(box, fill=colour) if rng.random() < 0.5 else draw.rectangle(box, fill=colour)
    for line in range(8):
        draw.text((edge // 20, edge // 20 + line * max(edge // 16, 12)), f"Imagent test picture, line {line + 1}", fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=config.IMAGE_JPEG_QUALITY)
    return buffer.getvalue()


def _power_of_two_at_least(n: int) -> int:
    return 1 << max(math.ceil(math.log2(max(n, 1))), 0)


def _model_context_limit(client: ollama.Client, model: str) -> int | None:
    """The longest context the model was trained for, or None when Ollama does not say."""
    try:
        info = client.show(model).modelinfo or {}
    except Exception:
        return None
    return next((int(value) for key, value in info.items() if key.endswith(".context_length") and isinstance(value, int)), None)


def _model_bytes(client: ollama.Client, model: str) -> int | None:
    try:
        return next((int(m.size) for m in client.list().models if m.model == model and m.size), None)
    except Exception:
        return None


def _seconds(nanoseconds) -> float:
    return (nanoseconds or 0) / 1e9


def _trial(client: ollama.Client, model: str, edge: int, num_ctx: int, use_mmap: bool) -> tuple[dict, ollama.Client]:
    """One real request at this size and context. Returns what Ollama measured, and the client to go on with.

    Raises ``ServiceTrouble`` when the model could not be reached even after the steps for its kind of trouble, and
    ``NoAnswer`` when it was reached and said nothing."""
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": config.ANALYSIS_PROMPT},
            {"role": "user", "content": "Analyze this image.", "images": [calibration_image(edge)]},
        ],
        "format": config.ANALYSIS_SCHEMA,
        "options": {**config.LLM_OPTIONS, "num_ctx": num_ctx, "use_mmap": use_mmap, "num_predict": config.CALIBRATION_TRIAL_TOKENS},
    }
    response, client = send_request(client, model, request, lambda kind, detail: None, lambda stage: None)
    if not (response.message.content or "").strip() or not getattr(response, "eval_count", 0):
        raise NoAnswer("Ollama answered with nothing.")
    eval_seconds = _seconds(getattr(response, "eval_duration", 0))
    return {
        "prompt_tokens": int(getattr(response, "prompt_eval_count", 0) or 0),
        "prefill_seconds": _seconds(getattr(response, "prompt_eval_duration", 0)),
        "load_seconds": _seconds(getattr(response, "load_duration", 0)),
        "tokens_per_second": (getattr(response, "eval_count", 0) or 0) / eval_seconds if eval_seconds else 0.0,
    }, client


def _fit_context(client: ollama.Client, model: str, edge: int, use_mmap: bool, limit: int | None,
                 start: int) -> tuple[int, dict, ollama.Client]:  # raises ServiceTrouble, NoAnswer or NotPossible
    """The smallest context that holds the prompt at this size, found by trying: Ollama cuts a prompt that does not fit.

    Returns it with the trial's measurements. Raises ``ServiceTrouble`` if the trial fails, ``NotPossible`` if the
    model's own limit is too small for this size."""
    context = start
    while True:
        measured, client = _trial(client, model, edge, context, use_mmap)
        needed = _power_of_two_at_least(measured["prompt_tokens"] + math.ceil(config.OLLAMA_MAX_OUTPUT_TOKENS * config.ANALYSIS_CAP_STEP))
        if needed <= context:
            return context, measured, client
        if limit is not None and needed > limit:
            raise NotPossible(f"At {edge} pixels the model needs a context of {needed}, more than it supports ({limit}).")
        context = needed


def calibrate(client: ollama.Client, model: str, say: Callable[[str], None] = print) -> dict:
    """Measure this computer with the vision model ``model``. Returns the settings (see ``db.operations.set_device_settings``).

    The model is unloaded afterwards. Raises ``NotPossible`` when it does not run here at any size worth sending."""
    available = available_memory()
    size = _model_bytes(client, model)
    limit = _model_context_limit(client, model)
    # Paged in from disk only when the model would not fit in the free memory with headroom to spare; if that way fails, the other is tried.
    prefer_mmap = bool(size and available and size > available * (1 - config.CALIBRATION_MEMORY_HEADROOM))
    start = _power_of_two_at_least(config.OLLAMA_MAX_OUTPUT_TOKENS * 2)
    trouble = ""
    edge = config.IMAGE_EDGE_CEILING
    try:
        while edge >= config.CALIBRATION_MIN_EDGE:
            say(f"Trying pictures up to {edge} pixels…")
            for use_mmap in (prefer_mmap, not prefer_mmap):
                try:
                    context, first, client = _fit_context(client, model, edge, use_mmap, limit, start)
                except NotPossible as exc:  # the model's own limit: another memory mode cannot help
                    trouble = str(exc)
                    break
                except NoAnswer as exc:
                    trouble = str(exc)
                    continue
                except ServiceTrouble as exc:
                    trouble = str(exc)
                    if exc.kind == OUT_OF_MEMORY:
                        continue
                    break  # no reply in time, or Ollama stopped: the other way would only wait as long again
                # Measure again with the model loaded: how fast it writes once it is running. Ollama keeps what it worked
                # out for an identical request, so a second reading of the prompt takes almost no time; the first one counts.
                try:
                    warm, client = _trial(client, model, edge, context, use_mmap)
                except (NoAnswer, ServiceTrouble) as exc:
                    trouble = str(exc)
                    break
                speed = warm["tokens_per_second"] or first["tokens_per_second"]
                if speed <= 0:
                    trouble = "The model did not report how fast it works."
                    break
                prefill = max(first["prefill_seconds"], warm["prefill_seconds"])
                first_reply = first["load_seconds"] + prefill + config.OLLAMA_MAX_OUTPUT_TOKENS / speed
                return {
                    "num_ctx": context, "use_mmap": use_mmap, "image_long_edge": edge,
                    "timeout_seconds": math.ceil(first_reply * config.CALIBRATION_TIMEOUT_MARGIN),
                    "measurements": {
                        "prompt_tokens": warm["prompt_tokens"], "prefill_seconds": round(prefill, 2),
                        "load_seconds": round(first["load_seconds"], 2), "tokens_per_second": round(speed, 2),
                        "model_bytes": size, "free_memory_bytes": available, "model_context_limit": limit,
                    },
                }
            edge = int(edge * config.CALIBRATION_EDGE_STEP)
        raise NotPossible(f"did not work on this computer at any picture size worth using"
                          f"{f' (last problem: {trouble})' if trouble else ''}. A smaller model may.")
    finally:
        unload_model(client, model)
