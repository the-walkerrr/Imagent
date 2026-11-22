"""
imagent/pipeline/error_messages.py — What the user is told when an image cannot be indexed, one entry per error id.

The database keeps only the id of the error and the values that fill its gaps (``backlog.error_id`` and
``backlog.error_params``), never the wording: the wording is built from this table when it is shown, so
it can be improved without touching what is stored. A message may have gaps written as ``{name}``.
"""

from __future__ import annotations

MESSAGES: dict[str, str] = {
    # The picture itself
    "damaged_picture": "The picture file is damaged or incomplete. Copy it again from the original.",
    "unsupported_format": "Imagent cannot open {suffix} pictures on this computer. "
                          "Convert the picture to JPEG or PNG and keep the copy in the same folder.",
    "no_memory_to_open": "There was not enough memory to open the picture. Close other programs, then scan again.",
    "picture_too_large": "The picture has {megapixels} megapixels, more than the memory free right now allows. "
                         "Close other programs, then scan again.",
    "picture_too_large_memory_unknown": "The picture has {megapixels} megapixels, this computer does not say how much memory is free, "
                                        "so only a picture that is not very large can be read. "
                                        "Save a smaller copy of it from a photo app, then scan again.",
    "not_downloaded": "The picture is kept in cloud storage and is not on this computer, and Imagent does not use the internet. "
                      "Make it available offline (download it), then scan again.",
    "changed_since_scan": "The file changed after it was scanned, so it may still be being written. It is tried again later.",
    "text_unreadable": "The text in the image could not be read, in any of the ways tried: {steps}",
    # The model and Ollama
    "reply_cut_off": "The model could not finish describing this picture, even after trying again with different settings "
                     "and with more room. Saving the picture again from a photo app, as a new file, gives it a fresh try.",
    "reply_invalid": "The model's answer was not in the expected form, even after it was told what was wrong. "
                     "Saving the picture again from a photo app, as a new file, gives it a fresh try.",
    "model_timeout": "The model did not reply in time, even after waiting longer.",
    "model_out_of_memory": "Ollama ran out of memory, even after other models were unloaded.",
    "model_stopped": "Ollama stopped responding, and starting it again did not help.",
    # Anything not foreseen: the kind of error and what it said
    "unexpected": "{type}: {text}",
}

# Added after the message of trouble with the machine that was tried ``tries`` times in a row (see commands/scan.py).
TRIED_SEVERAL_TIMES = "Imagent tried {tries} times without success. Close other apps that use a lot of memory, then scan again."


def all_messages() -> dict[str, str]:
    """Every message as stored in the ``error_messages`` table: ``MESSAGES`` and the sentence added after a message that was tried several times."""
    return {**MESSAGES, "tried_several_times": TRIED_SEVERAL_TIMES}


def describe(error_id: str | None, params: dict | None = None) -> str:
    """The message for ``error_id``, its gaps filled from ``params``; empty when there is no id. Raises ``KeyError`` for an id not in ``MESSAGES``."""
    if error_id is None:
        return ""
    params = params or {}
    message = MESSAGES[error_id].format(**params)
    if "tries" in params:
        message += " " + TRIED_SEVERAL_TIMES.format(**params)
    return message
