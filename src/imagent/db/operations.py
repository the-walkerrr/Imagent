"""
imagent/db/operations.py — Low-level SQL helpers for model_settings, scan_roots, image_data, backlog,
text_embeddings, error_logs and failure_history.

All functions accept an open ``sqlite3.Connection`` and operate on a single row
(or, where named so, on a list of rows).
Transaction management (BEGIN / COMMIT / ROLLBACK) is the caller's responsibility.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterator, NamedTuple

from imagent import config

# The columns of image_data that say what produced a record (see db/schema.sql); a record without them has them empty.
VERSION_COLUMNS = ("vlm_digest", "prompt_hash", "schema_hash", "search_text_version", "ocr_version", "ollama_version", "app_version")

# Rows deleted per statement. SQLite limits the number of ``?`` in one statement
# (999 before version 3.32), so long lists are processed in chunks below that.
_CHUNK = 500

_BEFORE_EVERY_ID = -2 ** 63  # SQLite's smallest integer: batches are read by file_id, from the first one


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# model_settings
# ---------------------------------------------------------------------------

def get_model_settings(conn: sqlite3.Connection) -> dict | None:
    """Return the locked model pair, or None when setup has not been completed."""
    row = conn.execute(
        """
        SELECT vlm_model, embedding_model, judge_model,
               vlm_digest, embedding_digest, judge_digest, embedding_dim
        FROM model_settings WHERE id = 1
        """
    ).fetchone()
    return dict(row) if row else None


def set_model_settings(conn: sqlite3.Connection, settings: dict) -> None:
    """Lock in the models. A plain INSERT: raises ``sqlite3.IntegrityError`` if they are already set."""
    conn.execute(
        """
        INSERT INTO model_settings
            (id, vlm_model, embedding_model, judge_model,
             vlm_digest, embedding_digest, judge_digest, embedding_dim, locked_at)
        VALUES (1, :vlm_model, :embedding_model, :judge_model,
                :vlm_digest, :embedding_digest, :judge_digest, :embedding_dim, :locked_at)
        """,
        {**settings, "locked_at": _now()},
    )


def get_device_settings(conn: sqlite3.Connection) -> dict | None:
    """Return what setup measured about this computer (see ``vlm.calibration``), or None when it has not been measured."""
    row = conn.execute(
        "SELECT num_ctx, use_mmap, image_long_edge, timeout_seconds, one_model_at_a_time, vlm_digest, measurements "
        "FROM device_settings WHERE id = 1"
    ).fetchone()
    if row is None:
        return None
    apart = row["one_model_at_a_time"]
    return {**dict(row), "use_mmap": bool(row["use_mmap"]), "one_model_at_a_time": None if apart is None else bool(apart),
            "measurements": json.loads(row["measurements"])}


def set_device_settings(conn: sqlite3.Connection, device: dict, vlm_digest: str | None) -> None:
    """Store what was measured, replacing the earlier measurement (the computer or the vision model may have changed)."""
    conn.execute(
        """
        INSERT OR REPLACE INTO device_settings
            (id, num_ctx, use_mmap, image_long_edge, timeout_seconds, one_model_at_a_time, vlm_digest, measurements, app_version, calibrated_at)
        VALUES (1, :num_ctx, :use_mmap, :image_long_edge, :timeout_seconds, :one_model_at_a_time, :vlm_digest, :measurements,
                :app_version, :calibrated_at)
        """,
        {**device, "use_mmap": int(device["use_mmap"]), "vlm_digest": vlm_digest,
         "one_model_at_a_time": None if device.get("one_model_at_a_time") is None else int(device["one_model_at_a_time"]),
         "measurements": json.dumps(device["measurements"], sort_keys=True), "app_version": config.APP_VERSION,
         "calibrated_at": _now()},
    )


# ---------------------------------------------------------------------------
# scan_roots
# ---------------------------------------------------------------------------

def get_or_add_root(conn: sqlite3.Connection, path: str) -> int:
    """Return the root_id of a scanned folder, adding the folder if it is new."""
    conn.execute("INSERT OR IGNORE INTO scan_roots (path, added_at) VALUES (?, ?)", (path, _now()))
    return conn.execute("SELECT root_id FROM scan_roots WHERE path = ?", (path,)).fetchone()["root_id"]


def list_roots(conn: sqlite3.Connection) -> list[dict]:
    """Return every scanned folder, the most recently scanned first."""
    rows = conn.execute(
        "SELECT root_id, path FROM scan_roots ORDER BY COALESCE(last_scanned_at, added_at) DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def mark_root_scanned(conn: sqlite3.Connection, root_id: int) -> None:
    conn.execute("UPDATE scan_roots SET last_scanned_at = ? WHERE root_id = ?", (_now(), root_id))


# ---------------------------------------------------------------------------
# image_data
# ---------------------------------------------------------------------------

def lookup_by_path(conn: sqlite3.Connection, file_path: str) -> dict | None:
    """Return the stored record for a path, or None if the path is not indexed.

    Used at the start of each indexing cycle to decide whether to skip,
    update, or freshly index an image.
    """
    row = conn.execute(
        "SELECT file_id, file_path, hash, modified_at FROM image_data WHERE file_path = ?",
        (file_path,),
    ).fetchone()
    return dict(row) if row else None


class StoredImage(NamedTuple):
    """What a scan needs to know about an image that is indexed already."""
    file_id: int
    hash: str | None
    modified_at: str | None
    pixel_hash: str | None = None
    width: int | None = None
    height: int | None = None
    file_size: int | None = None


_STORED_COLUMNS = "file_id, hash, modified_at, pixel_hash, width, height, file_size"


def images_by_path(conn: sqlite3.Connection, root_id: int) -> dict[str, StoredImage]:
    """Every image stored under a root, by file path, for a scan to tell new, changed, moved and gone files apart.

    Built row by row from the cursor: no list of all the rows is held beside the result, and each image is one small tuple."""
    return {row[0]: StoredImage(*row[1:]) for row in conn.execute(
        f"SELECT file_path, {_STORED_COLUMNS} FROM image_data WHERE root_id = ?", (root_id,))}


def images_outside_root(conn: sqlite3.Connection, root_id: int) -> Iterator[tuple[int, str, StoredImage]]:
    """Every image stored under the other folders, as ``(root_id, file_path, StoredImage)``, one row at a time: a scan looks
    among them for the pictures that were moved into its folder."""
    for row in conn.execute(f"SELECT root_id, file_path, {_STORED_COLUMNS} FROM image_data WHERE root_id != ?", (root_id,)):
        yield row[0], row[1], StoredImage(*row[2:])


_EXIF_FIELDS = ("taken_at", "camera_make", "camera_model", "gps_lat", "gps_lon")


def get_exif_fields(conn: sqlite3.Connection, file_id: int) -> dict:
    """The EXIF facts stored for an image (``taken_at``, ``camera_make``, ``camera_model``, ``gps_lat``, ``gps_lon``)."""
    row = conn.execute(f"SELECT {', '.join(_EXIF_FIELDS)} FROM image_data WHERE file_id = ?", (file_id,)).fetchone()
    return dict(row)


def upsert_image_data(conn: sqlite3.Connection, record: dict) -> int:
    """Insert a new image_data row or update every column of the existing one for that path.

    ``record`` must contain all column keys expected by the table except ``file_id``.
    Returns the row's file_id (unchanged when the row already existed).
    """
    conn.execute(
        """
        INSERT INTO image_data (
            root_id, file_path, hash, pixel_hash,
            short_description, description, vlm_metadata_json,
            width, height, file_size, mime_type,
            created_at, modified_at, indexed_at,
            processing_time_ms, metadata_version, vlm_model,
            taken_at, camera_make, camera_model, gps_lat, gps_lon,
            vlm_digest, prompt_hash, schema_hash, search_text_version, ocr_version, ollama_version, app_version
        ) VALUES (
            :root_id, :file_path, :hash, :pixel_hash,
            :short_description, :description, :vlm_metadata_json,
            :width, :height, :file_size, :mime_type,
            :created_at, :modified_at, :indexed_at,
            :processing_time_ms, :metadata_version, :vlm_model,
            :taken_at, :camera_make, :camera_model, :gps_lat, :gps_lon,
            :vlm_digest, :prompt_hash, :schema_hash, :search_text_version, :ocr_version, :ollama_version, :app_version
        )
        ON CONFLICT(file_path) DO UPDATE SET
            root_id             = excluded.root_id,
            hash                = excluded.hash,
            pixel_hash          = excluded.pixel_hash,
            short_description   = excluded.short_description,
            description         = excluded.description,
            vlm_metadata_json   = excluded.vlm_metadata_json,
            width               = excluded.width,
            height              = excluded.height,
            file_size           = excluded.file_size,
            mime_type           = excluded.mime_type,
            created_at          = excluded.created_at,
            modified_at         = excluded.modified_at,
            indexed_at          = excluded.indexed_at,
            processing_time_ms  = excluded.processing_time_ms,
            metadata_version    = excluded.metadata_version,
            vlm_model           = excluded.vlm_model,
            taken_at            = excluded.taken_at,
            camera_make         = excluded.camera_make,
            camera_model        = excluded.camera_model,
            gps_lat             = excluded.gps_lat,
            gps_lon             = excluded.gps_lon,
            vlm_digest          = excluded.vlm_digest,
            prompt_hash         = excluded.prompt_hash,
            schema_hash         = excluded.schema_hash,
            search_text_version = excluded.search_text_version,
            ocr_version         = excluded.ocr_version,
            ollama_version      = excluded.ollama_version,
            app_version         = excluded.app_version
        """,
        {**dict.fromkeys(VERSION_COLUMNS), **record},
    )
    return conn.execute(
        "SELECT file_id FROM image_data WHERE file_path = ?", (record["file_path"],)
    ).fetchone()["file_id"]


REUSED_COLUMNS = ("short_description", "description", "vlm_metadata_json", "metadata_version", "vlm_model", *VERSION_COLUMNS)


def find_analysis_by_pixels(conn: sqlite3.Connection, pixel_hash: str, file_path: str, wanted: dict) -> dict | None:
    """The analysis of another picture with the same pixels (same ``pixel_hash``), if it was made exactly the way a new one
    would be now, so that it can be used again instead of analysing the picture a second time; None when there is none.

    The indexed images are looked at first, then the analyses kept of deleted ones (``deleted_analyses``).

    ``wanted`` is what an analysis made now would be made with: ``vlm_model``, ``vlm_digest``, ``prompt_hash``, ``schema_hash``,
    ``search_text_version``, ``ocr_version``, ``metadata_version``, ``embedding_model``, ``embedding_digest`` and ``embedding_dim``.
    A value that is None (not known) matches nothing. The image at ``file_path`` itself is not a candidate; of the others the one
    indexed most recently is used. Returns the ``REUSED_COLUMNS`` of its row, with its ``embedding``, ``embedding_digest`` and
    ``embedding_dim``, and ``from_deleted``: whether it is a kept analysis of a deleted picture (``use_deleted_analysis`` then
    takes it out of the table)."""
    params = {**wanted, "pixel_hash": pixel_hash, "file_path": file_path}
    row = conn.execute(
        f"""
        SELECT {', '.join('i.' + column for column in REUSED_COLUMNS)}, e.embedding, e.embedding_digest, e.embedding_dim, 0 AS from_deleted
        FROM image_data i JOIN text_embeddings e ON e.file_id = i.file_id
        WHERE i.pixel_hash = :pixel_hash AND i.file_path != :file_path
          AND i.vlm_model = :vlm_model AND i.vlm_digest = :vlm_digest AND i.prompt_hash = :prompt_hash
          AND i.schema_hash = :schema_hash AND i.search_text_version = :search_text_version AND i.ocr_version = :ocr_version
          AND i.metadata_version = :metadata_version
          AND e.embedding_model = :embedding_model AND e.embedding_digest = :embedding_digest AND e.embedding_dim = :embedding_dim
        ORDER BY i.indexed_at DESC LIMIT 1
        """,
        params,
    ).fetchone()
    if row is None:
        row = conn.execute(
            f"""
            SELECT {', '.join(REUSED_COLUMNS)}, embedding, embedding_digest, embedding_dim, 1 AS from_deleted
            FROM deleted_analyses
            WHERE pixel_hash = :pixel_hash
              AND vlm_model = :vlm_model AND vlm_digest = :vlm_digest AND prompt_hash = :prompt_hash
              AND schema_hash = :schema_hash AND search_text_version = :search_text_version AND ocr_version = :ocr_version
              AND metadata_version = :metadata_version
              AND embedding_model = :embedding_model AND embedding_digest = :embedding_digest AND embedding_dim = :embedding_dim
            """,
            params,
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# deleted_analyses, index_preferences (the user's settings)
# ---------------------------------------------------------------------------

def _get_preference(conn: sqlite3.Connection, column: str):
    """One column of the preferences row (``column`` is always one of the names below, never typed by the user); None when it was not chosen."""
    row = conn.execute(f"SELECT {column} FROM index_preferences WHERE id = 1").fetchone()
    return None if row is None else row[column]


def _set_preference(conn: sqlite3.Connection, column: str, value) -> None:
    conn.execute(f"INSERT INTO index_preferences (id, {column}) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET {column} = excluded.{column}", (value,))


def get_keep_deleted(conn: sqlite3.Connection) -> int | None:
    """How many deleted pictures' analyses the user chose to keep (0: none), or None when the question was not answered yet."""
    return _get_preference(conn, "keep_deleted")


def set_keep_deleted(conn: sqlite3.Connection, keep: int) -> None:
    """Store how many deleted pictures' analyses to keep, and forget the oldest ones beyond it."""
    _set_preference(conn, "keep_deleted", keep)
    _prune_deleted_analyses(conn, keep)


def get_search_detail(conn: sqlite3.Connection) -> str | None:
    """What a search shows for each picture by default, ``"short"`` or ``"full"``, or None when the user never chose."""
    return _get_preference(conn, "search_detail")


def set_search_detail(conn: sqlite3.Connection, detail: str) -> None:
    _set_preference(conn, "search_detail", detail)


def _prune_deleted_analyses(conn: sqlite3.Connection, keep: int) -> None:
    conn.execute(
        "DELETE FROM deleted_analyses WHERE pixel_hash NOT IN "
        "(SELECT pixel_hash FROM deleted_analyses ORDER BY deleted_at DESC, rowid DESC LIMIT ?)",
        (keep,),
    )


def keep_analyses_of(conn: sqlite3.Connection, file_ids: list[int], keep: int | None) -> None:
    """Keep the analysis of the images ``file_ids`` (about to be deleted) in ``deleted_analyses``, the newest ``keep`` deletions
    in all. Nothing is kept when ``keep`` is None (not chosen yet) or 0, and an image without a pixel hash is not kept: it could
    not be found again. Call it before ``delete_images``, in the same transaction."""
    if not keep:
        return
    now = _now()
    columns = ", ".join(REUSED_COLUMNS)
    for start in range(0, len(file_ids), _CHUNK):
        chunk = file_ids[start:start + _CHUNK]
        conn.execute(
            f"""
            INSERT OR REPLACE INTO deleted_analyses
                (pixel_hash, {columns}, embedding, embedding_model, embedding_digest, embedding_dim, deleted_at)
            SELECT i.pixel_hash, {', '.join('i.' + column for column in REUSED_COLUMNS)},
                   e.embedding, e.embedding_model, e.embedding_digest, e.embedding_dim, ?
            FROM image_data i JOIN text_embeddings e ON e.file_id = i.file_id
            WHERE i.pixel_hash IS NOT NULL AND i.file_id IN ({", ".join("?" for _ in chunk)})
            """,
            [now, *chunk],
        )
    _prune_deleted_analyses(conn, keep)


def count_deleted_analyses(conn: sqlite3.Connection) -> int:
    """How many deleted pictures' analyses are kept."""
    return conn.execute("SELECT COUNT(*) FROM deleted_analyses").fetchone()[0]


def clear_deleted_analyses(conn: sqlite3.Connection) -> int:
    """Forget every kept analysis of a deleted picture (the user's choice of how many to keep is left as it is). Returns how many were forgotten."""
    return conn.execute("DELETE FROM deleted_analyses").rowcount


def use_deleted_analysis(conn: sqlite3.Connection, pixel_hash: str) -> None:
    """Take a kept analysis out of ``deleted_analyses``: its picture is indexed again, so it is no longer a deleted one."""
    conn.execute("DELETE FROM deleted_analyses WHERE pixel_hash = ?", (pixel_hash,))


def update_image_location(conn: sqlite3.Connection, file_id: int, root_id: int, file_path: str, modified_at: str) -> None:
    """Point an already-indexed image at its new path, under ``root_id`` (a move or rename of the same picture)."""
    conn.execute(
        "UPDATE image_data SET root_id = ?, file_path = ?, modified_at = ? WHERE file_id = ?",
        (root_id, file_path, modified_at, file_id),
    )


def refresh_image_file(conn: sqlite3.Connection, file_id: int, file_hash: str, details: dict) -> None:
    """Bring what is stored about the file in line with the file, for a picture that did not change: its bytes hash, size,
    timestamps and EXIF facts. ``details`` is what ``extract_file_metadata`` returns. Nothing about the analysis is touched."""
    conn.execute(
        """
        UPDATE image_data SET hash = :hash, width = :width, height = :height, file_size = :file_size, mime_type = :mime_type,
            created_at = :created_at, modified_at = :modified_at,
            taken_at = :taken_at, camera_make = :camera_make, camera_model = :camera_model, gps_lat = :gps_lat, gps_lon = :gps_lon
        WHERE file_id = :file_id
        """,
        {**details, "hash": file_hash, "file_id": file_id},
    )


def update_image_modified_at(conn: sqlite3.Connection, file_id: int, modified_at: str) -> None:
    """Update the stored modification timestamp of an image whose content did not change."""
    conn.execute(
        "UPDATE image_data SET modified_at = ? WHERE file_id = ?",
        (modified_at, file_id),
    )


def count_images(conn: sqlite3.Connection) -> int:
    """How many images are indexed, in every scanned folder."""
    return conn.execute("SELECT COUNT(*) AS n FROM image_data").fetchone()["n"]


def delete_images(conn: sqlite3.Connection, file_ids: list[int]) -> int:
    """Delete image_data rows by file_id (embeddings and error logs follow); returns the count."""
    deleted = 0
    for start in range(0, len(file_ids), _CHUNK):
        chunk = file_ids[start:start + _CHUNK]
        placeholders = ", ".join("?" for _ in chunk)
        deleted += conn.execute(f"DELETE FROM image_data WHERE file_id IN ({placeholders})", chunk).rowcount
    return deleted


def root_stats(conn: sqlite3.Connection, root_id: int) -> dict:
    """What the index holds for one scanned folder: ``images`` indexed, their ``bytes`` on disk, the ``analysis_ms`` spent
    analysing them, and the files ``queued`` to be indexed."""
    row = conn.execute(
        "SELECT COUNT(*) AS images, COALESCE(SUM(file_size), 0) AS bytes, COALESCE(SUM(processing_time_ms), 0) AS analysis_ms "
        "FROM image_data WHERE root_id = ?", (root_id,)).fetchone()
    queued = conn.execute("SELECT COUNT(*) FROM backlog WHERE root_id = ?", (root_id,)).fetchone()[0]
    return {**dict(row), "queued": queued}


def count_keepable_analyses(conn: sqlite3.Connection, root_ids: list[int]) -> int:
    """How many different pictures of the folders ``root_ids`` ``keep_analyses_of`` could keep (those with a pixel hash and an embedding)."""
    marks = ", ".join("?" for _ in root_ids)
    return conn.execute(
        f"SELECT COUNT(DISTINCT i.pixel_hash) FROM image_data i JOIN text_embeddings e ON e.file_id = i.file_id "
        f"WHERE i.pixel_hash IS NOT NULL AND i.root_id IN ({marks})", root_ids).fetchone()[0]


def image_ids_in_roots(conn: sqlite3.Connection, root_ids: list[int]) -> list[int]:
    """The file_id of every indexed image of the folders ``root_ids``."""
    marks = ", ".join("?" for _ in root_ids)
    return [row[0] for row in conn.execute(f"SELECT file_id FROM image_data WHERE root_id IN ({marks})", root_ids)]


def delete_roots(conn: sqlite3.Connection, root_ids: list[int]) -> int:
    """Remove the folders ``root_ids`` from the scanned folders; their images (with embeddings and error logs) and queued files go with them.
    Returns how many folders were removed. To keep the analyses of the images, call ``keep_analyses_of`` first, in the same transaction."""
    marks = ", ".join("?" for _ in root_ids)
    return conn.execute(f"DELETE FROM scan_roots WHERE root_id IN ({marks})", root_ids).rowcount


# ---------------------------------------------------------------------------
# backlog
# ---------------------------------------------------------------------------

def _dump_params(params: dict | None) -> str | None:
    return json.dumps(params, sort_keys=True) if params else None


def _with_error_params(row: sqlite3.Row) -> dict:
    """``row`` as a dict whose ``error_params`` is the dict that was stored (empty when there were none)."""
    item = dict(row)
    item["error_params"] = json.loads(item["error_params"]) if item["error_params"] else {}
    return item


def upsert_backlog(conn: sqlite3.Connection, file_path: str, root_id: int, modified_at: str) -> None:
    """Queue a file for indexing. If the file changed since it was last queued, its failures are forgotten (a parked file is queued again)."""
    conn.execute(
        """
        INSERT INTO backlog (file_path, root_id, modified_at, queued_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            root_id     = excluded.root_id,
            attempts    = CASE WHEN backlog.modified_at != excluded.modified_at THEN 0 ELSE backlog.attempts END,
            status      = CASE WHEN backlog.modified_at != excluded.modified_at THEN 'queued' ELSE backlog.status END,
            park_kind   = CASE WHEN backlog.modified_at != excluded.modified_at THEN NULL ELSE backlog.park_kind END,
            retry_when  = CASE WHEN backlog.modified_at != excluded.modified_at THEN NULL ELSE backlog.retry_when END,
            error_id    = CASE WHEN backlog.modified_at != excluded.modified_at THEN NULL ELSE backlog.error_id END,
            error_params = CASE WHEN backlog.modified_at != excluded.modified_at THEN NULL ELSE backlog.error_params END,
            modified_at = excluded.modified_at
        """,
        (file_path, root_id, modified_at, _now()),
    )


def get_backlog(conn: sqlite3.Connection, max_attempts: int) -> list[dict]:
    """Return the queued backlog rows (file_path, root_id and the modified_at the scan saw) that have failed fewer than ``max_attempts`` times (parked files are not queued)."""
    rows = conn.execute(
        "SELECT file_path, root_id, modified_at FROM backlog WHERE status = 'queued' AND attempts < ? ORDER BY file_path",
        (max_attempts,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_exhausted_backlog(conn: sqlite3.Connection, max_attempts: int) -> list[dict]:
    """Return the backlog rows that could not be indexed, with their last error (``error_id`` and its ``error_params`` as a dict): parked, failed ``max_attempts`` times,
    or left queued for the next run after every try for their kind of failure (``park_kind`` says which, see ``record_backlog_failure``)."""
    rows = conn.execute(
        "SELECT file_path, error_id, error_params, park_kind FROM backlog "
        "WHERE status = 'parked' OR attempts >= ? OR park_kind IS NOT NULL ORDER BY file_path",
        (max_attempts,),
    ).fetchall()
    return [_with_error_params(row) for row in rows]


def refund_backlog_attempts(conn: sqlite3.Connection, file_paths: list[str]) -> None:
    """Give back the attempt each of these files lost, because the failure turned out not to be theirs (a drive that went away)."""
    for start in range(0, len(file_paths), _CHUNK):
        chunk = file_paths[start:start + _CHUNK]
        placeholders = ", ".join("?" for _ in chunk)
        conn.execute(f"UPDATE backlog SET attempts = MAX(attempts - 1, 0) WHERE file_path IN ({placeholders})", chunk)


def park_backlog(conn: sqlite3.Connection, file_path: str, kind: str, error_id: str, retry_when: str,
                 versions: dict | None = None, params: dict | None = None) -> None:
    """Set a file aside because every step for its kind of failure was tried.

    ``error_id`` and ``params`` say why (see ``pipeline/error_messages.py``); only these are stored, not the wording.

    ``retry_when`` says what would make another try worthwhile ('model_changed': a different model or prompt, judged
    by ``versions``, the versions at the time; 'file_changed'; 'on_request'). The file is queued again when it changes
    (``upsert_backlog``), when its versions change (``requeue_changed_parked``) or on request (``requeue_parked``).
    """
    conn.execute(
        "UPDATE backlog SET status = 'parked', park_kind = ?, error_id = ?, error_params = ?, retry_when = ?, park_versions = ? "
        "WHERE file_path = ?",
        (kind, error_id, _dump_params(params), retry_when, json.dumps(versions, sort_keys=True) if versions else None, file_path),
    )


def requeue_changed_parked(conn: sqlite3.Connection, versions: dict) -> int:
    """Queue again the files parked because of the model's replies, if the versions they were parked with are not ``versions`` any more."""
    now = json.dumps(versions, sort_keys=True)
    rows = conn.execute(
        "SELECT file_path FROM backlog WHERE status = 'parked' AND retry_when = 'model_changed' "
        "AND COALESCE(park_versions, '') != ?", (now,)).fetchall()
    return requeue_parked(conn, paths=[row["file_path"] for row in rows])


def list_parked(conn: sqlite3.Connection) -> list[dict]:
    """Every parked file with the kind of failure and the error (``error_id`` and its ``error_params`` as a dict)."""
    rows = conn.execute(
        "SELECT file_path, park_kind, error_id, error_params, retry_when FROM backlog WHERE status = 'parked' ORDER BY park_kind, file_path"
    ).fetchall()
    return [_with_error_params(row) for row in rows]


def requeue_parked(conn: sqlite3.Connection, kind: str | None = None, paths: list[str] | None = None) -> int:
    """Queue parked files again with a clean slate: all of them, those of one ``kind``, or the given ``paths``. Returns how many."""
    where, params = "status = 'parked'", []
    if kind is not None:
        where += " AND park_kind = ?"
        params.append(kind)
    sql = ("UPDATE backlog SET status = 'queued', attempts = 0, park_kind = NULL, retry_when = NULL, "
           "park_versions = NULL, error_id = NULL, error_params = NULL WHERE ")
    if paths is None:
        return conn.execute(sql + where, params).rowcount
    requeued = 0
    for start in range(0, len(paths), _CHUNK):  # SQLite allows only so many ? in one statement
        chunk = paths[start:start + _CHUNK]
        placeholders = ", ".join("?" for _ in chunk)
        requeued += conn.execute(f"{sql}{where} AND file_path IN ({placeholders})", [*params, *chunk]).rowcount
    return requeued


def add_failure_history(conn: sqlite3.Connection, file_path: str, attempts: list[dict]) -> None:
    """Record the requests made for one image, if any of them had a reply that had to be asked for again.

    ``attempts`` is the record kept by ``vlm.analyzer``. A request that worked after others failed is stored as 'resolved'.
    """
    if not any(attempt["kind"] for attempt in attempts):
        return
    conn.executemany(
        """
        INSERT INTO failure_history
            (file_path, kind, step, options, done_reason, eval_count, prompt_eval_count, detail, raw_reply, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (file_path, attempt["kind"] or "resolved", attempt["step"], json.dumps(attempt["options"]), attempt["done_reason"],
             attempt["eval_count"], attempt["prompt_eval_count"], attempt["detail"] or None, attempt["raw_reply"], _now())
            for attempt in attempts
        ],
    )


def get_backlog_paths_for_root(conn: sqlite3.Connection, root_id: int) -> list[str]:
    rows = conn.execute("SELECT file_path FROM backlog WHERE root_id = ?", (root_id,)).fetchall()
    return [row["file_path"] for row in rows]


def delete_backlog(conn: sqlite3.Connection, file_path: str) -> None:
    """Remove a backlog row once it has been processed (or no longer needs indexing)."""
    conn.execute("DELETE FROM backlog WHERE file_path = ?", (file_path,))


def delete_backlog_paths(conn: sqlite3.Connection, file_paths: list[str]) -> None:
    for start in range(0, len(file_paths), _CHUNK):
        chunk = file_paths[start:start + _CHUNK]
        placeholders = ", ".join("?" for _ in chunk)
        conn.execute(f"DELETE FROM backlog WHERE file_path IN ({placeholders})", chunk)


def record_backlog_failure(conn: sqlite3.Connection, file_path: str, error_id: str, count_attempt: bool,
                           kind: str | None = None, params: dict | None = None) -> None:
    """Remember why indexing a file failed, as an ``error_id`` and the ``params`` for its message; ``count_attempt`` is False when the file is not to blame.

    ``kind`` is given when every try for the failure was made and the file stays queued for the next run all the same
    (the machine was to blame, not the file): it is what lets the user be told that the file could not be indexed."""
    conn.execute(
        "UPDATE backlog SET attempts = attempts + ?, error_id = ?, error_params = ?, park_kind = ? WHERE file_path = ?",
        (1 if count_attempt else 0, error_id, _dump_params(params), kind, file_path),
    )


# ---------------------------------------------------------------------------
# text_embeddings
# ---------------------------------------------------------------------------

def upsert_text_embedding(conn: sqlite3.Connection, file_id: int, embedding_bytes: bytes,
                          model: str | None = None, digest: str | None = None, dim: int | None = None) -> None:
    """Insert or update the serialised embedding for an image, with the model that made it (name, digest) and its size.

    ``embedding_bytes`` should be a packed array of little-endian float32
    (see ``pipeline.serialisation.serialize_embedding``).
    """
    conn.execute(
        """
        INSERT INTO text_embeddings (file_id, embedding, created_at, embedding_model, embedding_digest, embedding_dim)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            embedding        = excluded.embedding,
            created_at       = excluded.created_at,
            embedding_model  = excluded.embedding_model,
            embedding_digest = excluded.embedding_digest,
            embedding_dim    = excluded.embedding_dim
        """,
        (file_id, embedding_bytes, _now(), model, digest, dim),
    )


def count_embeddings(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM text_embeddings").fetchone()["n"]


def iter_embedding_batches(conn: sqlite3.Connection, size: int) -> Iterator[list[sqlite3.Row]]:
    """Every stored (file_id, embedding blob) row, ``size`` rows at a time, in file_id order.

    Only one batch is in memory at a time, however many images are indexed."""
    last = _BEFORE_EVERY_ID
    while True:
        rows = conn.execute("SELECT file_id, embedding FROM text_embeddings WHERE file_id > ? ORDER BY file_id LIMIT ?", (last, size)).fetchall()
        if not rows:
            return
        yield rows
        last = rows[-1]["file_id"]


def get_image_summary(conn: sqlite3.Connection, file_id: int) -> dict | None:
    """Return display/reconstruction fields for a single image.

    Used by search to present results: its descriptions, the file's properties and EXIF facts, and the folder it was found under.
    """
    row = conn.execute(
        """
        SELECT i.file_id, i.file_path, i.short_description, i.description, i.vlm_metadata_json,
               i.width, i.height, i.file_size, i.mime_type, i.taken_at, i.modified_at,
               i.camera_make, i.camera_model, i.gps_lat, i.gps_lon, r.path AS root_path
        FROM image_data i JOIN scan_roots r ON r.root_id = i.root_id
        WHERE i.file_id = ?
        """,
        (file_id,),
    ).fetchone()
    return dict(row) if row else None
