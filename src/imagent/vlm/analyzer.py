from __future__ import annotations

import time
from typing import Callable, NamedTuple

import httpx
import ollama
import config
from vlm.client import make_client, other_models_running, restart_server, unload_model
from vlm.reply_validation import check_reply
from vlm.response_format import QueryUnderstanding
from image_processing.convert import ConversionFailed, converted_copy
from image_processing.features import extract_image_features
from image_processing.memory import check_image_size
from image_processing.ocr import extract_text_blocks
from image_processing.preprocess import preprocess_image
from image_processing.problems import TOO_LARGE, UNSUPPORTED, ImageProblem, explain
from pipeline.failure import IndexingFailure, attempt_record

# What went wrong; each has its own way of being put right.
CUT_OFF = "cut_off"  # the reply ended at the output cap
INVALID_REPLY = "invalid_reply"  # the reply is complete but does not fit the schema
TIMEOUT = "timeout"  # no reply within the time limit
SERVER_CRASH = "server_crash"  # the connection to Ollama broke, or its model runner stopped
OUT_OF_MEMORY = "out_of_memory"  # Ollama has not enough memory for the model

# What the indexing loop does with a file for each kind (see pipeline/failure.py).
_HANDLING = {
    CUT_OFF: {"retry_when": "model_changed", "model_trouble": True},
    INVALID_REPLY: {"retry_when": "model_changed", "model_trouble": True},
    # The machine's trouble, not the file's: the file stays queued and is tried again by the next run (the loop in
    # scan_and_index tries it config.SERVICE_TRIES times first, and tells the user when that was not enough).
    TIMEOUT: {"park": False, "service": True},
    SERVER_CRASH: {"park": False, "service": True},
    OUT_OF_MEMORY: {"park": False, "service": True},
}


class AnalysisFailed(IndexingFailure):
    """The model could not be made to give a usable reply by any of the steps for its kind of trouble.

    ``kind`` is the last kind of trouble and ``attempts`` records every request made (see ``_request_analysis``).
    """

    def __init__(self, kind: str, message: str, attempts: list[dict]):
        super().__init__(kind, message, attempts, **_HANDLING[kind])


class ServiceTrouble(Exception):
    """Reaching the model failed in a way the steps in ``send_request`` could not put right."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class Analysis(NamedTuple):
    """Everything learned about one image."""
    reply: dict  # the vision model's reply, exactly as it wrote it
    ocr_text: list[str]  # the pieces of text read in the image
    features: dict  # facts read from the pixels
    attempts: list[dict]  # one entry per request to the model; ``kind`` says what was wrong with the reply, if anything


def analyse_image(client: ollama.Client, model: str, image_path: str,
                  on_stage: Callable[[str], None] = lambda stage: None,
                  on_retry: Callable[[str], None] = lambda kind: None) -> Analysis:
    """Analyse one image; ``on_stage`` is told what is being done, before each step.

    ``on_retry`` is told the kind of trouble each time the model's reply has to be asked for again.
    """
    on_stage("checking image")
    try:
        return _analyse_file(client, model, image_path, on_stage, on_retry)
    except ImageProblem as unsupported:
        if unsupported.kind != UNSUPPORTED:
            raise
        on_stage("converting image")
        try:
            with converted_copy(image_path) as converted:  # a PNG copy that is deleted when the analysis is done
                return _analyse_file(client, model, converted, on_stage, on_retry)
        except ConversionFailed as failed:
            unsupported.attempts.append(attempt_record("converted the image", UNSUPPORTED, str(failed)))
            raise unsupported from failed


def _check_image_size(client: ollama.Client, image_path: str, on_stage: Callable[[str], None]) -> None:
    """``check_image_size``, but an image too large for the memory free is given more chances, one step after the other:
    every model Ollama has loaded is unloaded (whatever setup found about the models sharing the memory) and the size is checked
    again with the memory this frees; if it is still too large, it is checked once more with the larger share of the free memory,
    ``config.IMAGE_MEMORY_SHARE_LAST_TRY``.

    The models load again by themselves when the next request needs them. Raises ``ImageProblem`` as ``check_image_size`` does."""
    try:
        check_image_size(image_path)
        return
    except ImageProblem as large:
        if large.kind != TOO_LARGE:
            raise
    loaded = other_models_running(client, [])
    if loaded:  # when no model is using memory there is nothing to free, and no reason to wait
        on_stage("freeing memory for a large image")
        for name in loaded:
            unload_model(client, name)
        time.sleep(config.OLLAMA_RECOVERY_WAIT)  # the memory is not free the moment Ollama says the model is gone
        try:
            check_image_size(image_path)
            return
        except ImageProblem as still_large:
            if still_large.kind != TOO_LARGE:
                raise
    check_image_size(image_path, config.IMAGE_MEMORY_SHARE_LAST_TRY)


def _analyse_file(client: ollama.Client, model: str, image_path: str,
                  on_stage: Callable[[str], None], on_retry: Callable[[str], None]) -> Analysis:
    _check_image_size(client, image_path, on_stage)  # raises ImageProblem if the image cannot be opened or is too large
    # The steps that are quick and need no model come first: when one of them fails, no reply was asked for in vain.
    try:
        on_stage("preparing image")
        image_bytes = preprocess_image(image_path)
        features = extract_image_features(image_path)
        on_stage("reading text in image")
        ocr_text = extract_text_blocks(image_path)
    except ImageProblem:
        raise
    except Exception as exc:
        raise (explain(exc, image_path) or exc) from exc
    on_stage("analysing image")
    reply, attempts = _request_analysis(client, model, image_bytes, on_stage, on_retry)
    return Analysis(reply, ocr_text, features, attempts)


def _ran_out_of_memory(exc: Exception) -> bool:
    """True if a failed request was the computer running out of memory.

    Ollama does not always say so: when the graphics memory runs out in the middle of a request, all it reports is
    "Compute error" (status 500), so that counts too."""
    if not isinstance(exc, ollama.ResponseError):
        return False
    message = str(exc).lower()
    return "memory" in message or "compute error" in message


def send_request(client: ollama.Client, model: str, request: dict, note: Callable[[str, str], None],
          on_stage: Callable[[str], None]) -> tuple:
    """Send a chat request, putting right trouble in reaching the model, each kind in its own way.

    No reply in time: ask again and wait longer. The connection broke or the model runner stopped: make sure Ollama is
    running (starting it if it is on this computer) and ask again. Not enough memory: unload the other models, wait, ask again.
    Each of these is done once for the connection troubles and up to ``config.OLLAMA_TIMEOUT_RAISES`` times for the wait.
    Returns the reply and the client that got it (a new one if the wait was lengthened); raises ``ServiceTrouble`` when it runs out.
    """
    timeout, raises, recovered = config.OLLAMA_TIMEOUT, 0, False
    while True:
        try:
            return client.chat(**request), client
        except httpx.ReadTimeout as exc:
            note(TIMEOUT, f"no reply within {timeout:g} seconds")
            if raises >= config.OLLAMA_TIMEOUT_RAISES:
                raise ServiceTrouble(TIMEOUT, "The model did not reply in time, even after waiting longer.") from exc
            raises += 1
            timeout *= config.OLLAMA_TIMEOUT_STEP
            client = make_client(timeout=timeout)
            on_stage(f"retrying: no reply in time; waiting up to {timeout:g} s")
        except (httpx.RemoteProtocolError, httpx.ReadError, ConnectionError, ollama.ResponseError) as exc:
            if isinstance(exc, ollama.ResponseError) and exc.status_code != 500:
                raise  # not a stopped runner (a missing model, a bad request): another kind of problem
            out_of_memory = _ran_out_of_memory(exc)
            kind = OUT_OF_MEMORY if out_of_memory else SERVER_CRASH
            note(kind, f"{type(exc).__name__}: {exc}")
            if recovered:
                raise ServiceTrouble(kind, "Ollama ran out of memory, even after other models were unloaded." if out_of_memory
                                     else "Ollama stopped responding, and starting it again did not help.") from exc
            recovered = True
            if out_of_memory:
                for name in other_models_running(client, [model]):
                    unload_model(client, name)
                time.sleep(config.OLLAMA_RECOVERY_WAIT)
                on_stage("retrying: memory was freed")
            elif restart_server(client):
                on_stage("retrying: Ollama was restarted")
            else:
                raise


def _was_cut_off(response, num_predict: int) -> bool:
    """True if the model stopped because it reached the output cap, not because it was finished."""
    if getattr(response, "done_reason", None) == "length":
        return True
    used = getattr(response, "eval_count", None)
    return isinstance(used, int) and used >= num_predict


def _request_analysis(client: ollama.Client, model: str, image_bytes: bytes,
                      on_stage: Callable[[str], None] = lambda stage: None,
                      on_retry: Callable[[str], None] = lambda kind: None) -> tuple[dict, list[dict]]:
    """Ask the model to describe the image and put right what is wrong with its reply, each kind of trouble in its own way.

    A reply that ended at the output cap is either a loop or an answer that is simply long, and these are told apart by
    what works. First one retry with different sampling at the same cap: it ends a loop, and costs a long answer only
    one more try. If it is cut off again, the cap is raised step by step, up to what the context window has room for after
    the prompt, with the normal sampling (a long answer is then the same answer, only finished). Nothing is asked to be shorter.

    A complete reply that does not fit the schema is sent back with the problems found, and the model answers again.

    Returns the reply and the record of every request; raises ``AnalysisFailed`` when the steps for its trouble run out.
    """
    first_messages = [
        {"role": "system", "content": config.ANALYSIS_PROMPT},
        {"role": "user", "content": "Analyze this image.", "images": [image_bytes]},
    ]
    messages = first_messages
    cap = config.OLLAMA_MAX_OUTPUT_TOKENS
    changed_sampling = False
    step = "first try"
    sampling_tried = False
    raises = 0
    feedback_tries = 0
    retries = 0  # also the seed of a retry with changed sampling, so that no two retries repeat each other
    attempts: list[dict] = []

    def note(kind: str, detail: str) -> None:
        """Record trouble that was put right on the way (a wait that ran out, a restart) as an attempt of its own."""
        attempts.append(attempt_record(step, kind, detail, options=options))
        on_retry(kind)

    while True:
        options = {**config.LLM_OPTIONS, "num_predict": cap}
        if changed_sampling:
            options.update(temperature=config.ANALYSIS_RETRY_TEMPERATURE, seed=retries,
                           repeat_penalty=config.ANALYSIS_RETRY_REPEAT_PENALTY)
        try:
            response, client = send_request(client, model, {"model": model, "messages": messages, "format": config.ANALYSIS_SCHEMA,
                                                     "options": options}, note, on_stage)
        except ServiceTrouble as trouble:
            raise AnalysisFailed(trouble.kind, str(trouble), attempts) from trouble
        content = response.message.content or ""
        reply, problems = check_reply(content, config.ANALYSIS_SCHEMA)
        prompt_tokens = getattr(response, "prompt_eval_count", None)
        attempt = {
            "step": step, "kind": None, "options": options, "detail": "", "raw_reply": None,
            "done_reason": getattr(response, "done_reason", None),
            "eval_count": getattr(response, "eval_count", None), "prompt_eval_count": prompt_tokens,
        }
        attempts.append(attempt)
        if reply is not None:
            return reply, attempts

        kind = CUT_OFF if _was_cut_off(response, cap) else INVALID_REPLY
        attempt.update(kind=kind, detail="; ".join(problems)[:500], raw_reply=content)

        if kind == CUT_OFF:
            if not sampling_tried:
                sampling_tried, changed_sampling, step = True, True, "changed sampling"
                retries += 1
                messages = first_messages
                reason = "the reply was cut off; asking again with different sampling"
            else:
                room = config.OLLAMA_NUM_CTX - (prompt_tokens or 0)  # the context window holds the prompt and the reply
                larger = min(int(cap * config.ANALYSIS_CAP_STEP), room)
                if raises >= config.ANALYSIS_CAP_RAISES or larger <= cap:
                    raise AnalysisFailed(kind, "The model could not finish describing this picture, even after trying again with "
                                               "different settings and with more room. Saving the picture again from a photo app, "
                                               "as a new file, gives it a fresh try.", attempts)
                cap, changed_sampling, step = larger, False, "raised cap"
                raises += 1
                messages = first_messages
                reason = f"the reply was cut off; asking again with room for {cap} tokens"
        else:
            if feedback_tries >= config.ANALYSIS_RETRIES:
                raise AnalysisFailed(kind, "The model's answer was not in the expected form, even after it was told what was wrong. "
                                           "Saving the picture again from a photo app, as a new file, gives it a fresh try.", attempts)
            feedback_tries += 1
            retries += 1
            changed_sampling, step = True, "problems sent back"
            messages = first_messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"That reply was not valid ({'; '.join(problems[:3])[:300]}). "
                                            "Reply again with one complete, valid JSON object that follows the schema."},
            ]
            reason = "the reply did not fit the schema; asking again"
        on_retry(kind)
        on_stage(f"retrying: {reason}")


def _too_long(exc: ollama.ResponseError) -> bool:
    """True if Ollama refused an input because it does not fit the embedding model's context."""
    message = str(exc).lower()
    return exc.status_code == 400 and any(word in message for word in ("context", "too long", "exceed"))


def _split(text: str) -> tuple[str, str]:
    """Cut a text in two, at a line break near the middle, or else at a space."""
    lines = text.splitlines()
    if len(lines) > 1:
        running = 0
        for index, line in enumerate(lines):
            running += len(line) + 1
            if running >= len(text) / 2:
                break
        cut = min(max(index + 1, 1), len(lines) - 1)
        return "\n".join(lines[:cut]), "\n".join(lines[cut:])
    middle = len(text) // 2
    space = text.rfind(" ", 0, middle)
    cut = space if space > 0 else middle
    return text[:cut], text[cut:]


def _embed(client: ollama.Client, model: str, text: str) -> list[float]:
    """One embedding request. If the embedding model's runner fails (status 500: usually the memory ran out because another
    model is loaded beside it), every model is unloaded, which also resets the failed runner, and it is asked once more."""
    request = {"model": model, "input": [text], "truncate": False, "options": config.EMBED_OPTIONS}
    try:
        return list(client.embed(**request).embeddings[0])
    except ollama.ResponseError as exc:
        if exc.status_code != 500:
            raise
    for name in other_models_running(client, []):
        unload_model(client, name)
    time.sleep(config.OLLAMA_RECOVERY_WAIT)
    return list(client.embed(**request).embeddings[0])


def embed_search_text(client: ollama.Client, model: str, search_text: str) -> list[float]:
    """The embedding of a text. Nothing is cut off: a text too long for the model's context is split in two, again and
    again until the parts fit, and its vector is the average of the parts', each weighted by its length."""
    try:
        return _embed(client, model, search_text)
    except ollama.ResponseError as exc:
        if not _too_long(exc) or len(search_text) < 2:
            raise
    first, second = _split(search_text)
    vectors = embed_search_text(client, model, first), embed_search_text(client, model, second)
    weights = len(first), len(second)
    return [(a * weights[0] + b * weights[1]) / sum(weights) for a, b in zip(*vectors)]


def understand_query(client: ollama.Client, model: str, query: str) -> QueryUnderstanding:
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": config.QUERY_PROMPT},
            {"role": "user", "content": f"User query: {query}"},
        ],
        format=QueryUnderstanding.model_json_schema(),
        options=config.LLM_OPTIONS,
    )
    return QueryUnderstanding.model_validate_json(response.message.content)
