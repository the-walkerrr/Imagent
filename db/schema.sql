-- The latest schema, applied to a new database by db/connection.py (version SCHEMA_VERSION there).
-- Changing it means raising that version and adding the steps that upgrade older databases to _MIGRATIONS.
-- Foreign keys are switched on for every connection by db/connection.py (the pragma does nothing inside a transaction).

-- ============================================================================
-- TABLE: model_settings
--
-- What: The models this database was built with (a single row).
-- Why:  Embedding vectors from different models cannot be compared, so the
--       models are locked together with the vectors they produced. The row is
--       inserted once by install.py and never updated; the CHECK keeps it to one row.
-- When: Written as the last step of setup, once all models are downloaded and verified.
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),

    -- Ollama model names including the tag (e.g. 'nomic-embed-text:latest').
    -- vlm_model describes images (indexing only); judge_model understands queries
    -- (search only; it never sees an image); embedding_model is used by both.
    -- The judge may be the same model as the VLM.
    vlm_model TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    judge_model TEXT NOT NULL,

    -- Ollama's digest of each model when it was set up. A different digest later
    -- means the same tag now points at different weights; a warning is shown.
    vlm_digest TEXT,
    embedding_digest TEXT,
    judge_digest TEXT,

    -- Number of dimensions of the embedding model's vectors, measured at setup.
    embedding_dim INTEGER NOT NULL,

    locked_at DATETIME NOT NULL
);


-- ============================================================================
-- TABLE: device_settings
--
-- What: What setup measured about this computer with the locked models (a single row).
-- Why:  How long to wait for a reply, how large a context and an image the models can take, and whether to
--       page models in from disk depend on the machine, so they are measured (vlm/calibration.py), not guessed.
-- When: Written with model_settings at the end of setup, and again (settings.vlm_digest differs) when the
--       vision model changes. Read by config.apply_device_settings before every run.
-- ============================================================================
CREATE TABLE IF NOT EXISTS device_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),

    -- The values every request to the models is sent with (see config.py: they must not change between requests).
    num_ctx INTEGER NOT NULL,
    use_mmap INTEGER NOT NULL CHECK (use_mmap IN (0, 1)),

    -- Longest side, in pixels, of the image sent to the vision model.
    image_long_edge INTEGER NOT NULL,

    -- Seconds to wait for the first reply of a request; each wait that runs out is doubled (vlm/analyzer.py).
    timeout_seconds INTEGER NOT NULL,

    -- The vision model these were measured with (its digest) and the numbers they came from, as JSON.
    vlm_digest TEXT,
    measurements TEXT NOT NULL,

    app_version TEXT NOT NULL,
    calibrated_at DATETIME NOT NULL
);


-- ============================================================================
-- TABLE: scan_roots
--
-- What: The folders the user has pointed Imagent at.
-- Why:  A scan only reconciles the files of its own folder, so scanning one
--       folder can never delete what was indexed from another. Overlapping
--       folders are rejected, so every file belongs to exactly one root.
-- When: A row is added the first time a folder is scanned.
-- ============================================================================
CREATE TABLE IF NOT EXISTS scan_roots (
    root_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Resolved absolute path of the folder.
    path TEXT NOT NULL UNIQUE,

    added_at DATETIME NOT NULL,
    last_scanned_at DATETIME
);


-- ============================================================================
-- TABLE: image_data
--
-- What: Stores primary metadata, file properties, and VLM analysis results
--       for every indexed image.
-- Why:  Acts as the central registry of images, facilitating duplicate detection,
--       change detection, and serving as the primary source of natural language descriptions.
-- When: Populated or updated whenever an image is scanned, analyzed, or re-indexed.
-- ============================================================================
CREATE TABLE IF NOT EXISTS image_data (
    -- Surrogate key, stable when the file is moved or renamed.
    -- Referenced by text_embeddings and error_logs.
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- The folder this image was found under. Deleting a root deletes its images.
    root_id INTEGER NOT NULL REFERENCES scan_roots(root_id) ON DELETE CASCADE,

    -- Absolute path to the image on disk.
    -- Used to locate the image and must be unique to prevent indexing the same path multiple times.
    file_path TEXT NOT NULL UNIQUE,

    -- SHA-256 content hash of the image.
    -- Used for change detection and for recognising a moved or renamed file.
    hash TEXT NOT NULL,

    -- One-line summary generated by the Vision-Language Model (VLM).
    -- Used for quick previews and search result summaries.
    short_description TEXT,

    -- Detailed natural language description generated by the VLM.
    -- Used for comprehensive search matches and details views.
    description TEXT,

    -- JSON string containing all remaining VLM metadata (e.g. tags, objects, people, OCR).
    -- Excludes short_description and description to avoid redundancy.
    -- Used as a flexible, evolvable storage to avoid schema migrations for structured metadata.
    vlm_metadata_json TEXT NOT NULL,

    -- Image width in pixels.
    -- Used for filtering search results by dimensions or rendering UI layouts.
    width INTEGER,

    -- Image height in pixels.
    -- Used for filtering search results by dimensions or rendering UI layouts.
    height INTEGER,

    -- File size of the image on disk in bytes.
    -- Used for file stats, size filtering, and validating file status.
    file_size INTEGER,

    -- MIME type of the image (e.g., 'image/jpeg', 'image/png').
    -- Used to identify format and determine decoder requirements.
    mime_type TEXT,

    -- File creation timestamp if the OS reports one (not available on every OS / Python version).
    -- Used to sort or filter images chronologically by their capture/creation time.
    created_at DATETIME,

    -- Last modification timestamp of the image file on disk.
    -- Used to check if a file has changed since it was last indexed.
    modified_at DATETIME,

    -- Timestamp indicating when this image was last indexed by Imagent.
    -- Used to monitor index freshness and schedule maintenance/updates.
    indexed_at DATETIME,

    -- Total time taken to index this image (vision model + embeddings + DB write) in milliseconds.
    -- Used for performance monitoring and metrics.
    processing_time_ms INTEGER,

    -- Version of the stored metadata structure.
    -- Used to identify legacy metadata formats and trigger updates or migrations.
    metadata_version INTEGER NOT NULL DEFAULT 1,

    -- The name of the Vision-Language Model used to generate the metadata.
    vlm_model TEXT,

    -- Local capture date and time from the image's EXIF data (ISO 8601, no timezone), e.g. '2026-09-06T07:24:12'.
    -- Used for date and time-of-day search; NULL when the image has no EXIF capture time.
    taken_at TEXT,

    -- Camera manufacturer and model from EXIF (e.g. 'iQOO', 'iQOO Neo9 Pro').
    -- Used to filter or describe images by the device that took them; NULL when absent.
    camera_make TEXT,
    camera_model TEXT,

    -- GPS position from EXIF in signed decimal degrees (south and west are negative).
    -- Used for location search; NULL when the image has no GPS data.
    gps_lat REAL,
    gps_lon REAL,

    -- What produced this record, so that records made with something older can be told apart and redone:
    -- the vision model's digest (its exact weights), fingerprints of the prompt and of the reply schema,
    -- the version of the way the search text is built, and the versions of the OCR library, Ollama and Imagent.
    vlm_digest TEXT,
    prompt_hash TEXT,
    schema_hash TEXT,
    search_text_version INTEGER,
    ocr_version TEXT,
    ollama_version TEXT,
    app_version TEXT
);

-- Index on root_id: a scan loads every stored image of one root.
CREATE INDEX IF NOT EXISTS idx_image_data_root_id ON image_data(root_id);

-- Index on hash to optimize duplicate and moved-file detection queries.
CREATE INDEX IF NOT EXISTS idx_image_data_hash ON image_data(hash);

-- Index on modified_at to optimize delta-indexing scans detecting modified files.
CREATE INDEX IF NOT EXISTS idx_image_data_modified_at ON image_data(modified_at);


-- ============================================================================
-- TABLE: backlog
--
-- What: Stores files that need to be indexed or re-indexed after a filesystem scan.
-- Why:  Decouples cheap filesystem scanning from expensive VLM processing.
-- When: Populated by scan_and_index.py for new or modified files.
-- ============================================================================
CREATE TABLE IF NOT EXISTS backlog (
    -- Keyed by path: a new file has no image_data row (and so no file_id) yet.
    file_path TEXT PRIMARY KEY,

    root_id INTEGER NOT NULL REFERENCES scan_roots(root_id) ON DELETE CASCADE,

    -- Modification time seen by the scan that queued the file. When a later scan sees a
    -- different one, the file changed again, so its failed attempts are forgotten.
    modified_at DATETIME NOT NULL,

    -- Number of times indexing this file failed. At config.BACKLOG_MAX_ATTEMPTS the file
    -- is skipped until it changes. Failures caused by Ollama being unreachable do not count.
    attempts INTEGER NOT NULL DEFAULT 0,

    -- 'queued': indexed on the next run. 'parked': set aside because every step for its kind of
    -- failure was tried; it is queued again when the file changes (see upsert_backlog).
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'parked')),

    -- Why a file was parked (e.g. 'cut_off', 'invalid_reply') and what would make it worth another try.
    park_kind TEXT,
    retry_when TEXT,

    -- Message of the most recent failure, shown in the indexing summary.
    last_error TEXT,

    -- The model and prompt versions (JSON) at the time the file was parked: when they change, a file parked for
    -- its model's replies (retry_when = 'model_changed') is queued again.
    park_versions TEXT,

    queued_at DATETIME NOT NULL
);


-- ============================================================================
-- TABLE: failure_history
--
-- What: Every request whose reply had to be asked for again, with the exact reason and the
--       settings used, and (as kind 'resolved') the request that finally worked.
-- Why:  Shows which steps put right which kind of failure, and what a parked file already went through.
--       Unlike error_logs it is never trimmed.
-- When: Written when an image's reply needed more than one request, whether or not it was finally indexed.
-- ============================================================================
CREATE TABLE IF NOT EXISTS failure_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,

    -- What was wrong with the reply: 'cut_off', 'invalid_reply', or 'resolved' for the request that worked.
    kind TEXT NOT NULL,

    -- Which step of the ladder for that kind made the request ('first try', 'changed sampling', ...).
    step TEXT NOT NULL,

    -- The options sent to Ollama (JSON), and what Ollama reported back.
    options TEXT,
    done_reason TEXT,
    eval_count INTEGER,
    prompt_eval_count INTEGER,

    -- The problems found in the reply, and the reply itself (NULL for 'resolved').
    detail TEXT,
    raw_reply TEXT,

    created_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_failure_history_file_path ON failure_history(file_path);
CREATE INDEX IF NOT EXISTS idx_failure_history_kind ON failure_history(kind);


-- ============================================================================
-- TABLE: text_embeddings
--
-- What: Stores high-dimensional vector embeddings generated from image descriptions.
-- Why:  Enables semantic search by allowing vector similarity matches.
-- When: Populated when an image's description is indexed or updated.
-- ============================================================================
CREATE TABLE IF NOT EXISTS text_embeddings (
    -- Unique identifier of the image this embedding belongs to.
    -- References image_data(file_id) to link vector embeddings to image files.
    file_id INTEGER PRIMARY KEY,

    -- Serialized embedding vector: little-endian float32, embedding_dim values
    -- (see model_settings). Every vector comes from the single locked model.
    embedding BLOB NOT NULL,

    -- Timestamp indicating when the embedding was generated.
    -- Used for auditing and assessing cache/data freshness.
    created_at DATETIME,

    -- The model that made the vector (name, digest of its exact weights) and the vector's size.
    embedding_model TEXT,
    embedding_digest TEXT,
    embedding_dim INTEGER,

    -- Maintain database integrity by cascading deletions when an image is deleted.
    FOREIGN KEY (file_id) REFERENCES image_data(file_id) ON DELETE CASCADE
);


-- ============================================================================
-- TABLE: error_logs
--
-- What: Tracks failures and unexpected errors encountered while indexing or searching.
-- Why:  Provides a structured audit trail for debugging, monitoring, and retrying
--       failed operations without requiring log file parsing.
-- When: A row is inserted whenever the pipeline catches an error. Only the newest
--       config.ERROR_LOG_MAX_ROWS rows are kept.
-- ============================================================================
CREATE TABLE IF NOT EXISTS error_logs (
    -- Auto-incrementing primary key uniquely identifying each error record.
    -- Used to paginate, sort, and reference individual error entries.
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- The file_id of the image being processed when the error occurred, if it has an image_data row.
    -- NULL for files that were never indexed successfully (see file_path) and for errors outside a file.
    file_id INTEGER,

    -- Absolute path to the image file being processed when the error occurred.
    -- Stored separately from file_id so errors can be reported even when no
    -- image_data row exists yet.
    file_path TEXT,

    -- The named stage in the application where the error was caught.
    -- Valid values: 'pipeline' (indexing one image), 'scan', 'search',
    -- 'query_understanding'.
    -- Used to filter errors by stage and identify systemic bottlenecks.
    stage TEXT NOT NULL,

    -- The class or type name of the exception (e.g. 'FileNotFoundError', 'ValidationError').
    -- Used to group and triage errors by category without parsing the message.
    error_type TEXT NOT NULL,

    -- Human-readable description of what went wrong.
    -- Used for display in dashboards and log summaries without needing the stack trace.
    error_message TEXT NOT NULL,

    -- Full stack trace captured at the point of the exception.
    -- Used for in-depth debugging when error_message alone is insufficient.
    stack_trace TEXT,

    -- Optional JSON blob containing additional contextual information relevant to the failure.
    -- Examples: model name in use, timing data, partial payloads, retry count.
    -- Kept as JSON to remain flexible without adding new schema columns for every context type.
    metadata TEXT,

    -- Timestamp (UTC) when this error was recorded.
    -- Used for chronological filtering, rate analysis, and correlating errors with deployments.
    created_at DATETIME NOT NULL,

    -- If file_id is present, cascade deletions from image_data to keep error_logs consistent.
    -- Errors for deleted images are removed automatically to prevent orphaned records.
    FOREIGN KEY (file_id) REFERENCES image_data(file_id) ON DELETE CASCADE
);

-- Index on file_id to efficiently retrieve all errors associated with a specific image.
CREATE INDEX IF NOT EXISTS idx_error_logs_file_id ON error_logs(file_id);

-- Index on stage to filter and aggregate errors by stage.
CREATE INDEX IF NOT EXISTS idx_error_logs_stage ON error_logs(stage);

-- Index on created_at to support time-range queries and chronological browsing of errors.
CREATE INDEX IF NOT EXISTS idx_error_logs_created_at ON error_logs(created_at);
