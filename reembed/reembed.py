#!/usr/bin/env python3
"""
reembed/reembed.py — Imagent: make every stored vector again with the embedding model that is installed now.

Needed when the embedding model is no longer the version the vectors were made with (Imagent then refuses to index or
search, because vectors of different versions cannot be compared). The vision model is not asked again: the search text
of each image is built again from the reply and the text in the image that were stored, and only that is embedded.

How long it takes is measured on a few images and shown first. All the vectors are made before any is replaced, so
an interrupted run changes nothing: each is kept as it is made in a temporary table (not in memory, so the number of images
does not decide how much memory this needs), and all are swapped in at once. Afterwards the model is saved as a copy of its own (see vlm.client.pin_model), so a
later update of the original cannot change it again.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Iterator

# Make the project root importable when this file is run directly (``python reembed/reembed.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import config
    from db.connection import init_db
    from db.operations import (
        count_records_for_embedding,
        create_embedding_staging,
        drop_embedding_staging,
        iter_records_for_embedding,
        replace_embeddings_from_staging,
        stage_embedding,
        update_embedding_settings,
    )
    from pipeline.console import make_output_safe, run_cli
    from pipeline.locking import AlreadyRunning, single_instance
    from pipeline.serialisation import serialize_embedding
    from text_processing.formatters import SEARCH_TEXT_VERSION, build_search_text
    from vlm.analyzer import embed_search_text
    from vlm.client import NotReady, check_ready, is_pinned, pin_model
except ModuleNotFoundError as exc:
    sys.exit(f"Imagent is not set up yet (the Python package '{exc.name}' is missing). "
             f"Run: {'install.bat' if sys.platform == 'win32' else 'sh install.sh'}")

SAMPLE_SIZE = 3  # images embedded first, to measure how long one takes; the first of them only starts the model, and is not timed


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def search_text(record: dict) -> str | None:
    """The search text of one record, built again from the reply and the text in the image that were stored; None if it has no stored reply."""
    try:
        stored = json.loads(record["vlm_metadata_json"])
        reply = stored["vlm"]
        if not isinstance(reply, dict):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        return None
    return build_search_text(reply, stored.get("ocr_text") or [], config.SUMMARY_FIELD)


def search_texts(conn) -> Iterator[tuple[int, str]]:
    """``(file_id, search text)`` for each image that has a stored reply to build it from, one at a time (the images are read in batches)."""
    for batch in iter_records_for_embedding(conn, config.DB_BATCH_ROWS):
        for record in batch:
            text = search_text(record)
            if text is not None:
                yield record["file_id"], text


def _model_to_use(client, settings: dict) -> tuple[str, dict]:
    """The installed model to embed with: the locked one, or the model it was copied from if the copy is gone."""
    installed = {model.model: model.digest for model in client.list().models}
    name = settings["embedding_model"]
    if name in installed:
        return name, installed
    base = (config.load_install_record().get("pinned_models") or {}).get(name)
    if base in installed:
        return base, installed
    sys.exit(f"The embedding model '{name}' is not installed in Ollama. Run: {config.INSTALL_COMMAND} to put it back, then run this again.")


def reembed(assume_yes: bool = False) -> None:
    """Embed every stored image again; see the module's docstring."""
    conn = init_db(str(config.DB_PATH))
    try:
        try:
            client, settings = check_ready(conn, ())
        except NotReady as exc:
            sys.exit(f"{exc}")
        model, installed = _model_to_use(client, settings)
        total = sum(1 for _ in search_texts(conn))  # counted once, so that the user can be told how many and how long before anything is made
        without = count_records_for_embedding(conn) - total
        if without:
            print(f"{without} image(s) have no stored reply to build their text from; their vectors are left as they are.")
        if not total:
            print("Nothing to re-embed.")
            return

        # The first image is only there to start the model, whose loading is no measure of what an image costs; unless it is the only one.
        sample = list(itertools.islice(search_texts(conn), SAMPLE_SIZE))
        try:
            embed_search_text(client, model, sample[0][1])
            start = time.perf_counter()
            timed = sample[1:] or sample
            for _, text in timed:
                embed_search_text(client, model, text)
        except Exception as exc:
            sys.exit(f"The embedding model did not answer ({type(exc).__name__}: {exc}). Nothing was changed.")
        per_image = (time.perf_counter() - start) / len(timed)
        print(f"{total} image(s) to embed again with {model}; one took about {per_image:.2f} s, "
              f"so this takes about {_duration(per_image * total)}.")
        if not assume_yes:
            try:
                if input("Continue? [y/N]: ").strip().lower() not in ("y", "yes"):
                    print("Nothing was changed.")
                    return
            except EOFError:
                print("Nothing was changed.")
                return

        create_embedding_staging(conn)
        try:
            dimension = None
            for done, (file_id, text) in enumerate(search_texts(conn), start=1):
                try:
                    vector = embed_search_text(client, model, text)
                except Exception as exc:
                    sys.exit(f"Embedding stopped after {done - 1} of {total} ({type(exc).__name__}: {exc}). "
                             "Nothing was changed; run it again when Ollama is working.")
                if dimension is None:
                    dimension = len(vector)
                elif len(vector) != dimension:
                    sys.exit("The model gave vectors of different sizes, so nothing was changed.")
                stage_embedding(conn, file_id, serialize_embedding(vector))  # as bytes at once: a list of floats takes eight times the room
                if done % 25 == 0 or done == total:
                    conn.commit()  # only the temporary table has been written to; this ends the transaction, it does not change the database
                    print(f"{done} of {total} embedded…", flush=True)

            try:
                pinned, digest = (model, installed[model]) if is_pinned(model) else pin_model(client, model)
                if pinned != model:
                    config.record_pinned_model(pinned, model)
            except Exception as exc:
                pinned, digest = model, installed[model]
                print(f"Could not save a fixed copy of {model} ({type(exc).__name__}: {exc}); it is used as it is.")

            with conn:  # all or nothing
                replace_embeddings_from_staging(conn, pinned, digest, dimension, SEARCH_TEXT_VERSION)
                update_embedding_settings(conn, pinned, digest, dimension)
            print(f"Done: {total} vector(s) made again with {pinned}.")
        finally:
            drop_embedding_staging(conn)
    finally:
        conn.close()


def main() -> None:
    make_output_safe()
    try:
        with single_instance("indexing", "Indexing or re-embedding"):
            reembed(assume_yes="--yes" in sys.argv)
    except AlreadyRunning as exc:
        sys.exit(f"{exc}")


if __name__ == "__main__":
    run_cli(main)
