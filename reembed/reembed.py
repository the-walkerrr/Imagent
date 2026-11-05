#!/usr/bin/env python3
"""
reembed/reembed.py — Imagent: make every stored vector again with the embedding model that is installed now.

Needed when the embedding model is no longer the version the vectors were made with (Imagent then refuses to index or
search, because vectors of different versions cannot be compared). The vision model is not asked again: the search text
of each image is built again from the reply and the text in the image that were stored, and only that is embedded.

How long it takes is measured on a few images and shown first. All the vectors are made before any is replaced, so
an interrupted run changes nothing. Afterwards the model is saved as a copy of its own (see vlm.client.pin_model), so a
later update of the original cannot change it again.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the project root importable when this file is run directly (``python reembed/reembed.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import config
    from db.connection import init_db
    from db.operations import list_records_for_embedding, update_embedding_settings, upsert_text_embedding
    from pipeline.console import make_output_safe, run_cli
    from pipeline.locking import AlreadyRunning, single_instance
    from pipeline.serialisation import serialize_embedding
    from text_processing.formatters import SEARCH_TEXT_VERSION, build_search_text
    from vlm.analyzer import embed_search_text
    from vlm.client import NotReady, check_ready, is_pinned, pin_model
except ModuleNotFoundError as exc:
    sys.exit(f"Imagent is not set up yet (the Python package '{exc.name}' is missing). "
             f"Run: {'install.bat' if sys.platform == 'win32' else 'sh install.sh'}")

SAMPLE_SIZE = 3  # images embedded first, to measure how long one takes


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def search_texts(records: list[dict]) -> tuple[list[tuple[int, str]], int]:
    """``(file_id, search text)`` for each record that has a stored reply to build it from, and how many have none."""
    texts, without = [], 0
    for record in records:
        try:
            stored = json.loads(record["vlm_metadata_json"])
            reply = stored["vlm"]
            if not isinstance(reply, dict):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            without += 1
            continue
        texts.append((record["file_id"], build_search_text(reply, stored.get("ocr_text") or [], config.SUMMARY_FIELD)))
    return texts, without


def _model_to_use(client, settings: dict) -> tuple[str, dict]:
    """The installed model to embed with: the locked one, or the model it was copied from if the copy is gone."""
    installed = {model.model: model.digest for model in client.list().models}
    name = settings["embedding_model"]
    if name in installed:
        return name, installed
    base = (config.load_install_record().get("pinned_models") or {}).get(name)
    if base in installed:
        return base, installed
    sys.exit(f"[imagent] The embedding model '{name}' is not installed in Ollama. Run: {config.INSTALL_COMMAND} to put it back, then run this again.")


def reembed(assume_yes: bool = False) -> None:
    """Embed every stored image again; see the module's docstring."""
    conn = init_db(str(config.DB_PATH))
    try:
        try:
            client, settings = check_ready(conn, ())
        except NotReady as exc:
            sys.exit(f"[imagent] {exc}")
        model, installed = _model_to_use(client, settings)
        texts, without = search_texts(list_records_for_embedding(conn))
        if without:
            print(f"[imagent] {without} image(s) have no stored reply to build their text from; their vectors are left as they are.")
        if not texts:
            print("[imagent] Nothing to re-embed.")
            return

        sample = texts[:SAMPLE_SIZE]
        start = time.perf_counter()
        try:
            for _, text in sample:
                embed_search_text(client, model, text)
        except Exception as exc:
            sys.exit(f"[imagent] The embedding model did not answer ({type(exc).__name__}: {exc}). Nothing was changed.")
        per_image = (time.perf_counter() - start) / len(sample)
        print(f"[imagent] {len(texts)} image(s) to embed again with {model}; one took about {per_image:.2f} s, "
              f"so this takes about {_duration(per_image * len(texts))}.")
        if not assume_yes:
            try:
                if input("[imagent] Continue? [y/N]: ").strip().lower() not in ("y", "yes"):
                    print("[imagent] Nothing was changed.")
                    return
            except EOFError:
                print("[imagent] Nothing was changed.")
                return

        vectors = []
        for done, (_, text) in enumerate(texts, start=1):
            try:
                vectors.append(embed_search_text(client, model, text))
            except Exception as exc:
                sys.exit(f"[imagent] Embedding stopped after {done - 1} of {len(texts)} ({type(exc).__name__}: {exc}). "
                         "Nothing was changed; run it again when Ollama is working.")
            if done % 25 == 0 or done == len(texts):
                print(f"[imagent] {done} of {len(texts)} embedded…", flush=True)
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            sys.exit("[imagent] The model gave vectors of different sizes, so nothing was changed.")

        try:
            pinned, digest = (model, installed[model]) if is_pinned(model) else pin_model(client, model)
            if pinned != model:
                config.record_pinned_model(pinned, model)
        except Exception as exc:
            pinned, digest = model, installed[model]
            print(f"[imagent] Could not save a fixed copy of {model} ({type(exc).__name__}: {exc}); it is used as it is.")

        with conn:  # all or nothing
            for (file_id, _), vector in zip(texts, vectors):
                upsert_text_embedding(conn, file_id, serialize_embedding(vector), pinned, digest, dimension)
                conn.execute("UPDATE image_data SET search_text_version = ? WHERE file_id = ?", (SEARCH_TEXT_VERSION, file_id))
            update_embedding_settings(conn, pinned, digest, dimension)
        print(f"[imagent] Done: {len(texts)} vector(s) made again with {pinned}.")
    finally:
        conn.close()


def main() -> None:
    make_output_safe()
    try:
        with single_instance("indexing", "Indexing or re-embedding"):
            reembed(assume_yes="--yes" in sys.argv)
    except AlreadyRunning as exc:
        sys.exit(f"[imagent] {exc}")


if __name__ == "__main__":
    run_cli(main)
