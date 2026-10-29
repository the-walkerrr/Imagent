"""
vlm/client.py — The Ollama client (with timeouts) and the readiness check run before every
indexing or search session.
"""

from __future__ import annotations

import atexit
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from typing import Callable
from urllib.parse import urlparse

import httpx
import ollama

import config
from db.operations import get_model_settings


# Structured output (the JSON schema sent with each request) arrived in Ollama 0.5.0.
MIN_OLLAMA_VERSION = (0, 5, 0)


class NotReady(Exception):
    """Setup is incomplete or Ollama cannot serve the locked models; the message tells the user what to do."""


def make_client(host: str | None = None) -> ollama.Client:
    """Return an Ollama client that can never wait forever.

    ``host`` defaults to the one saved by setup. The read timeout also applies between the
    progress updates of a download, so a stalled download ends too.
    """
    timeout = httpx.Timeout(config.OLLAMA_TIMEOUT, connect=config.OLLAMA_CONNECT_TIMEOUT)
    return ollama.Client(host=host or config.OLLAMA_HOST, timeout=timeout)


def is_local(host: str) -> bool:
    """True if ``host`` is this machine, the only kind of Ollama server Imagent can start."""
    parsed = urlparse(host if "://" in host else f"http://{host}")
    return parsed.hostname in ("localhost", "127.0.0.1", "::1")


def is_reachable(client: ollama.Client) -> bool:
    """True if the Ollama server answers."""
    try:
        client.list()
        return True
    except Exception:
        return False


def _base_url(host: str) -> str:
    """The address the Ollama client uses for ``host``: a host with no scheme gets port 11434, one with a scheme the scheme's port."""
    has_scheme = "://" in host
    parsed = urlparse(host if has_scheme else f"http://{host}")
    port = parsed.port or (11434 if not has_scheme else 443 if parsed.scheme == "https" else 80)
    name = parsed.hostname or "127.0.0.1"
    return f"{parsed.scheme}://{f'[{name}]' if ':' in name else name}:{port}"


def server_version(host: str) -> tuple[int, ...] | None:
    """The version of the Ollama server at ``host`` as numbers, e.g. (0, 6, 4); None if it cannot be told.

    A development build reports 0.0.0, which says nothing, so it is None too.
    """
    try:
        reply = httpx.get(f"{_base_url(host)}/api/version", timeout=config.OLLAMA_CONNECT_TIMEOUT)
        version = tuple(int(part) for part in reply.json()["version"].split("-")[0].split(".")[:3])
    except Exception:
        return None
    return version if any(version) else None


def _exit_on_signal(signum, frame) -> None:
    sys.exit(128 + signum)  # a normal exit, so ``atexit`` and ``finally`` blocks run and the server is stopped


def _stop_server_when_killed() -> None:
    """Make a closed terminal (SIGHUP) or a plain ``kill`` (SIGTERM) end the program normally instead of abruptly."""
    for name in ("SIGTERM", "SIGHUP"):  # SIGHUP does not exist on Windows
        number = getattr(signal, name, None)
        if number is not None:
            try:
                signal.signal(number, _exit_on_signal)
            except ValueError:  # not the main thread
                pass


def start_server(client: ollama.Client, host: str | None = None, wait_seconds: int = 30) -> subprocess.Popen | None:
    """Start ``ollama serve`` in the background, on the address of ``host`` (default: the one saved by setup), and wait until it answers.

    Returns the process, or None if the command could not be run or the server never answered
    (it is stopped again in that case). The caller must stop the process with ``stop_server``.
    A terminal closed or a ``kill`` while it runs also stops it.
    """
    kwargs = {"creationflags": 0x00000008} if platform.system() == "Windows" else {"start_new_session": True}
    host = host or config.OLLAMA_HOST
    env = {**os.environ, "OLLAMA_HOST": host} if host else None  # otherwise the server listens on its default port, not the saved one
    try:
        process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, **kwargs)
    except OSError:
        return None
    _stop_server_when_killed()
    for _ in range(wait_seconds):
        if is_reachable(client):
            return process
        time.sleep(1)
    stop_server(process)
    return None


def stop_server(process: subprocess.Popen) -> None:
    """Stop a server started with ``start_server``."""
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _stop_started_server(process: subprocess.Popen) -> None:
    stop_server(process)
    print("[imagent] Stopped the Ollama server this run started.")


def check_ready(conn, roles: tuple[str, ...], say: Callable[[str], None] = print) -> tuple[ollama.Client, dict]:
    """Confirm setup is done and Ollama is reachable with the locked models for ``roles`` installed.

    ``roles`` are the models this run uses: ("vlm", "embedding") to index, ("judge", "embedding") to search.
    Returns the client and the locked model settings (names, digests, embedding_dim).
    Raises NotReady otherwise, so a run stops at once instead of failing on every image.

    If Ollama is not running and the host is this machine, it is started here and stopped
    again when the program exits. Its notices go through ``say``, so a caller with a live
    progress line can print them without breaking it.
    """
    settings = get_model_settings(conn)
    if settings is None:
        raise NotReady(f"Imagent is not set up yet. Run: {config.INSTALL_COMMAND}")
    if not config.OLLAMA_HOST:
        raise NotReady("Imagent's settings file (config.json) is missing or damaged, so it does not know where Ollama is. "
                       f"Run: {config.INSTALL_COMMAND}. It restores the address; your models and index are kept.")

    client = make_client()
    if not is_reachable(client) and is_local(config.OLLAMA_HOST) and shutil.which("ollama"):
        process = start_server(client, config.OLLAMA_HOST)
        if process is not None:
            atexit.register(_stop_started_server, process)
    try:
        installed = {model.model: model.digest for model in client.list().models}
    except Exception as exc:
        install_hint = (" The `ollama` command was not found: install Ollama from https://ollama.com/download."
                        if is_local(config.OLLAMA_HOST) and not shutil.which("ollama") else " Start Ollama and try again.")
        raise NotReady(f"Cannot reach Ollama at {config.OLLAMA_HOST} ({type(exc).__name__}: {exc}).{install_hint}") from exc

    version = server_version(config.OLLAMA_HOST)
    if version is not None and version < MIN_OLLAMA_VERSION:
        raise NotReady(
            f"Your Ollama is version {'.'.join(map(str, version))}, which is too old: Imagent needs "
            f"{'.'.join(map(str, MIN_OLLAMA_VERSION))} or newer. Update it from https://ollama.com/download and try again."
        )

    for role in roles:
        name = settings[f"{role}_model"]
        digest = installed.get(name)
        if digest is None:
            raise NotReady(f"The {role} model '{name}' is not installed in Ollama. Run: {config.INSTALL_COMMAND}")
        stored = settings[f"{role}_digest"]
        if stored and digest != stored:
            say(f"[imagent] Warning: '{name}' is not the same version it was when you set up Imagent. "
                "Results may be less accurate" +
                (" and searches may miss images." if role == "embedding" else "."))
    return client, settings


def other_models_running(client: ollama.Client, keep: list[str]) -> list[str]:
    """The names of the models Ollama has in memory right now that are not in ``keep``.

    Best effort: if Ollama cannot say, the answer is none.
    """
    try:
        return [model.model for model in client.ps().models if model.model not in keep]
    except Exception:
        return []


def load_model(client: ollama.Client, name: str, embedding: bool = False) -> None:
    """Ask Ollama to load a model into memory now (best effort), so the first request does not wait for it.

    Uses the same options as the real requests, so Ollama keeps the model it loaded here.
    """
    try:
        if embedding:
            client.embed(model=name, input=["warm up"], options=config.EMBED_OPTIONS)
        else:
            client.generate(model=name, prompt="", options=config.LLM_OPTIONS)
    except Exception:
        pass  # the real request reports any problem with its own message


def unload_model(client: ollama.Client, name: str) -> None:
    """Ask Ollama to free a model's memory now (best effort), so the next phase has room for its own models."""
    try:
        client.generate(model=name, prompt="", keep_alive=0)
    except Exception:
        pass  # only an optimisation; Ollama unloads idle models by itself
