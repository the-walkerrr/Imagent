import json
import os
import platform
from pathlib import Path


# ---------------------------------------------------------------------------
# Storage location. Every setting lives in this file; no environment variables are read.
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    """Per-user folder for the database and config.json, following each OS's convention.

    It must be on a local disk: SQLite's WAL mode is not safe on network drives.
    """
    try:
        home = Path.home()
    except (RuntimeError, KeyError):  # no HOME / user profile, as in some containers and services
        raise SystemExit("[imagent] Error: cannot find your home folder, where Imagent keeps its data.")
    system = platform.system()
    if system == "Darwin":
        return home / "Library" / "Application Support" / "Imagent"
    if system == "Windows":
        return home / "AppData" / "Local" / "Imagent"  # Local, not Roaming: the database must not sync between PCs
    return home / ".local" / "share" / "imagent"  # Linux and other Unix systems


IMAGENT_HOME: Path = _data_dir()
DB_PATH: Path = IMAGENT_HOME / "imagent.db"
CONFIG_PATH: Path = IMAGENT_HOME / "config.json"


def load_config() -> dict:
    """Return config.json as a dict, or {} if the file is missing, unreadable or not a JSON object."""
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` beside the real file and then swap it in, so a crash or a power cut
    leaves either the old file or the new one, never a half-written one."""
    IMAGENT_HOME.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def save_config(data: dict) -> None:
    """Write config.json (user preferences). The chosen models are not here: they are locked in the database."""
    _write_json(CONFIG_PATH, data)


def _install_record_path() -> Path:
    return IMAGENT_HOME / "install_record.json"


def load_install_record() -> dict:
    """Return what install.py added to this computer ({"ollama_models": [...]}), or {} if nothing is recorded.

    The uninstaller removes only what is listed here, so a model you already had before Imagent is never touched.
    """
    try:
        data = json.loads(_install_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_install_record(data: dict) -> None:
    _write_json(_install_record_path(), data)


# User preferences written by install.py.
OLLAMA_HOST: str = load_config().get("ollama_host", "")

# --- Ollama ------------------------------------------------------------------
OLLAMA_CONNECT_TIMEOUT: int = 10  # seconds to reach the server
OLLAMA_TIMEOUT: int = 300  # seconds to wait for a reply (a cold model load counts) or between download updates
OLLAMA_NUM_CTX: int = 8192
OLLAMA_MAX_OUTPUT_TOKENS: int = 2200  # cap on the image-analysis reply; without it a runaway reply fills the whole context window
OLLAMA_USE_MMAP: bool = False  # load each model fully into memory instead of paging it in from disk
ANALYSIS_RETRIES: int = 1  # extra tries when the vision model's reply is not valid JSON for the schema (cut off, wrong shape)
ANALYSIS_RETRY_TEMPERATURE: float = 0.4  # a retry must differ from the first try: at temperature 0 the same input gives the same reply

# Sent with every request to the VLM and to the judge model. num_ctx and use_mmap must be
# identical across all requests to one model: Ollama reloads it whenever one of them changes.
LLM_OPTIONS: dict = {"num_ctx": OLLAMA_NUM_CTX, "use_mmap": OLLAMA_USE_MMAP, "temperature": 0, "seed": 0}
EMBED_OPTIONS: dict = {"use_mmap": OLLAMA_USE_MMAP}

# --- Images ------------------------------------------------------------------
OCR_MIN_CONFIDENCE: float = 0.6  # OCR blocks scoring below this are dropped
OCR_MAX_CHARS: int = 1500  # total OCR text kept per image; more would push the end of the search text past the embedding model's context
IMAGE_MAX_LONG_EDGE: int = 1024
IMAGE_SIZE_MULTIPLE: int = 1  # resized sides are rounded down to a multiple of this; 1 = no rounding
IMAGE_JPEG_QUALITY: int = 85

SAMPLE_IMAGES_DIR: Path = Path(__file__).parent.parent / "sample_images"  # folder used by benchmarks/all_sample_images.py

# --- Database and indexing ---------------------------------------------------
SQLITE_BUSY_TIMEOUT_S: int = 30  # how long a connection waits for another process's write lock
SCAN_GUARD_MIN_FILES: int = 10  # a scan that finds at least this many indexed images gone ...
SCAN_GUARD_MAX_SHARE: float = 0.5  # ... and more than this share of the folder's indexed images asks before forgetting them
BACKLOG_MAX_ATTEMPTS: int = 3  # a file that fails this many times is left alone until it changes
MAX_CONSECUTIVE_SERVICE_FAILURES: int = 3  # indexing stops after this many Ollama failures in a row
ERROR_LOG_MAX_ROWS: int = 1000  # oldest error_logs rows beyond this are deleted

# --- Search ------------------------------------------------------------------
UNCERTAIN_TOP_K: int = 3  # results returned per search (the closest images)
SEARCH_MAX_MISSING_SKIPPED: int = 100  # closest images skipped because their file is gone, before the search gives up looking

# --- Launchers ---------------------------------------------------------------
# What messages tell the user to run, for the OS they are on.
INSTALL_COMMAND: str = "install.bat" if platform.system() == "Windows" else "sh install.sh"
RUN_COMMAND: str = "run.bat" if platform.system() == "Windows" else "sh run.sh"

# --- Prompts -----------------------------------------------------------------
PROMPTS_DIR: Path = Path(__file__).parent.parent / "prompts"
ANALYSIS_PROMPT: str = (PROMPTS_DIR / "analysis_prompt.txt").read_text(encoding="utf-8")
QUERY_PROMPT: str = (PROMPTS_DIR / "query_prompt.txt").read_text(encoding="utf-8")
