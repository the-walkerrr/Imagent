"""
pipeline/failure.py — What the indexing loop is told when one image cannot be indexed, or the whole run cannot go on.

Each kind of trouble does its own retrying where it happens (reading the image, reading its text, asking the model).
When that is not enough, it raises one of these, and the loop only has to know what to do with the file:
set it aside (``park``), or leave it queued without using up one of its attempts.
"""

from __future__ import annotations


class IndexingFailure(Exception):
    """One image could not be indexed, even after the steps for its kind of trouble.

    ``kind``       what went wrong, in a word (``cut_off``, ``unreadable``, ``timeout`` ...).
    ``attempts``   the record of what was tried, stored in ``failure_history``.
    ``park``       True: set the file aside (it is queued again when it changes); False: leave it queued, no attempt used.
    ``retry_when`` what would make another try worthwhile: ``file_changed``, ``model_changed`` or ``on_request``.
    ``service``    True when the trouble is not the file's but the machine's or Ollama's: several in a row stop the run.
    ``model_trouble`` True when the model's reply was the problem: many of these stop the run too (see config.RETRY_RATE_*).
    """

    def __init__(self, kind: str, message: str, attempts: list[dict] | None = None, *, park: bool = True,
                 retry_when: str = "on_request", service: bool = False, model_trouble: bool = False):
        super().__init__(message)
        self.kind = kind
        self.attempts = attempts or []
        self.park = park
        self.retry_when = retry_when
        self.service = service
        self.model_trouble = model_trouble


class RunStopped(Exception):
    """Something is wrong for every image, not for this one: the run stops and the file stays queued, untouched."""


def attempt_record(step: str, kind: str | None, detail: str = "", options: dict | None = None, raw_reply: str | None = None,
                   done_reason: str | None = None, eval_count: int | None = None, prompt_eval_count: int | None = None) -> dict:
    """One entry of an ``attempts`` list, in the form ``db.operations.add_failure_history`` stores."""
    return {"step": step, "kind": kind, "options": options or {}, "detail": detail, "raw_reply": raw_reply,
            "done_reason": done_reason, "eval_count": eval_count, "prompt_eval_count": prompt_eval_count}
