from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Storage location. Every setting lives in this file; no environment variables are read.
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    """Per-user folder for the database and config.json: ``~/.imagent`` on every OS.

    It must be on a local disk: SQLite's WAL mode is not safe on network drives.
    """
    try:
        home = Path.home()
    except (RuntimeError, KeyError):  # no HOME / user profile, as in some containers and services
        raise SystemExit("Error: cannot find your home folder, where Imagent keeps its data.")
    return home / ".imagent"


IMAGENT_HOME: Path = _data_dir()
DB_PATH: Path = IMAGENT_HOME / "imagent.db"
CONFIG_PATH: Path = IMAGENT_HOME / "config.json"
# Programs that setup installs for Imagent (Ollama, when the computer has none) live here, beside the data folder and not in it:
# uninstall deletes the data folder first and then asks whether to keep each program.
TOOLS_DIR: Path = IMAGENT_HOME.with_name(IMAGENT_HOME.name + "-tools")


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
    """Return what imagent setup added to this computer ({"ollama_models": [...]}), or {} if nothing is recorded.

    The uninstaller removes only what is listed here, so a model you already had before Imagent is never touched.
    """
    try:
        data = json.loads(_install_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_install_record(data: dict) -> None:
    _write_json(_install_record_path(), data)


def record_pinned_model(pinned: str, base: str) -> None:
    """Note that setup copied the model ``base`` under the name ``pinned``: the uninstaller removes the copy, and it can be made again from ``base``."""
    record = load_install_record()
    models = [m for m in record.get("ollama_models", []) if isinstance(m, str)]
    if pinned not in models:
        models.append(pinned)
    pins = record["pinned_models"] if isinstance(record.get("pinned_models"), dict) else {}
    save_install_record({**record, "ollama_models": models, "pinned_models": {**pins, pinned: base}})


# --- Programs installed by setup -----------------------------------------------
# Each lives in its own folder of TOOLS_DIR, with a small file saying what it is, so uninstall can describe it and offer to
# remove it even after the data folder (and the install record in it) is gone.
TOOL_INFO_FILE = "imagent-tool.json"
OLLAMA_DIR: Path = TOOLS_DIR / "ollama"  # holds program/ (what the archive unpacks to), models/ and the info file


def save_tool_info(folder: Path, name: str, description: str, version: str) -> None:
    (folder / TOOL_INFO_FILE).write_text(json.dumps({"name": name, "description": description, "version": version}, indent=2) + "\n",
                                         encoding="utf-8")


def installed_tools() -> list[dict]:
    """The programs setup installed that are still on this computer: [{"name", "description", "version", "path"}]."""
    tools = []
    for folder in sorted(TOOLS_DIR.iterdir()) if TOOLS_DIR.is_dir() else []:
        try:
            info = json.loads((folder / TOOL_INFO_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(info, dict) and isinstance(info.get("name"), str):
            tools.append({"name": info["name"], "description": str(info.get("description", "")),
                          "version": str(info.get("version", "")), "path": folder})
    return tools


# --- Ollama program ------------------------------------------------------------
# The Ollama that setup installs when the computer has none (or an older one than Imagent needs): the official standalone build
# for this OS and CPU, from Ollama's own release, pinned to one version. The release lists the SHA-256 of every file in its
# sha256sum.txt; that list is checked against the one sum pinned here, so the file names and their sums are read from the list
# and none of them is written in this code. Running it needs no administrator rights and no change to PATH, and removing it is
# deleting its folder.
OLLAMA_VERSION = "0.40.17"
OLLAMA_MANIFEST_SHA256 = "dc2f480e48d4f8dfd37d127e6a54347cf7f3fd80a52beb1e1ca611d47bdf4dcd"  # of sha256sum.txt in that release
OLLAMA_MANIFEST = "sha256sum.txt"
OLLAMA_RELEASE_URL = "https://github.com/ollama/ollama/releases/download/v{version}/{file}"
OLLAMA_ARCHIVE_ENDINGS = (".tgz", ".tar.zst", ".zip")


def ollama_release_url(file: str) -> str:
    return OLLAMA_RELEASE_URL.format(version=OLLAMA_VERSION, file=file)


def ollama_build_prefix() -> str | None:
    """The start of the file names of Ollama's build for this OS and CPU (``ollama-linux-amd64``), or None when it has none.
    The macOS build is one file for both CPUs, so its name has no CPU in it."""
    system = {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}.get(platform.system())
    cpu = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine().lower())
    if system is None or cpu is None:
        return None
    return "ollama-darwin" if system == "darwin" else f"ollama-{system}-{cpu}"


def ollama_checksums(manifest: str) -> dict:
    """File name -> SHA-256 from the text of a release's sha256sum.txt (lines like ``<sum>  ./<name>``)."""
    sums = {}
    for line in manifest.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0].lower()):
            sums[parts[1].strip().lstrip("*").removeprefix("./")] = parts[0].lower()
    return sums


def ollama_downloads(manifest: str, extras: tuple[str, ...] = ()) -> list[tuple[str, str, str]]:
    """(url, file name, sha256) of every file to install, from the text of the release's sha256sum.txt: the build for this computer
    first, then the extra file of each name in ``extras`` (``rocm``, ``jetpack6``…) that the release has, to be unpacked over it.
    Empty when the release has no build for this computer's OS and CPU."""
    prefix, sums = ollama_build_prefix(), ollama_checksums(manifest)
    if prefix is None:
        return []
    found = []
    for stem in (prefix, *(f"{prefix}-{extra}" for extra in extras)):
        name = next((stem + ending for ending in OLLAMA_ARCHIVE_ENDINGS if stem + ending in sums), None)
        if name is not None:
            found.append((ollama_release_url(name), name, sums[name]))
        elif stem == prefix:
            return []  # without the build itself an extra file has nothing to go over
    return found


def private_ollama_binary() -> str | None:
    """The `ollama` program that setup installed, or None if it did not (or it is gone)."""
    name = "ollama.exe" if platform.system() == "Windows" else "ollama"
    program = OLLAMA_DIR / "program"
    for path in (program / name, program / "bin" / name):  # Windows and macOS unpack it at the top, Linux in bin/
        if path.is_file():
            return str(path)
    return None


def ollama_binary() -> str | None:
    """The `ollama` program Imagent runs: its own copy when setup installed one (the models are in that copy's folder), otherwise the one on PATH."""
    return private_ollama_binary() or shutil.which("ollama")


def ollama_environment(host: str | None) -> dict | None:
    """The environment to start ``ollama serve`` with: the address to listen on and, when the program is Imagent's own, a home
    inside its folder. Ollama keeps its models, its key pair and its cache in a ``.ollama`` folder in the user's home (the models
    can be moved with OLLAMA_MODELS, the rest cannot), so without this, removing Imagent's copy would leave files behind in a folder
    shared with any Ollama the user has. None when there is nothing to add."""
    extra = {}
    if host:  # otherwise the server listens on its default port, not the saved one
        extra["OLLAMA_HOST"] = host
    if private_ollama_binary():
        extra["OLLAMA_MODELS"] = str(OLLAMA_DIR / "models")  # explicit, so an OLLAMA_MODELS of the user's own cannot send them elsewhere
        extra["HOME"] = extra["USERPROFILE"] = str(OLLAMA_DIR / "home")  # USERPROFILE is the home on Windows
    return {**os.environ, **extra} if extra else None


# User preferences written by imagent setup.
OLLAMA_HOST: str = load_config().get("ollama_host", "")

# --- Ollama ------------------------------------------------------------------
OLLAMA_CONNECT_TIMEOUT: int = 10  # seconds to reach the server
# These four depend on the computer. The values here are only where setup starts measuring: setup stores what it measured
# (vlm/calibration.py) and apply_device_settings puts it in their place before every run.
OLLAMA_TIMEOUT: int = 300  # seconds to wait for a reply (a cold model load counts) or between download updates
OLLAMA_NUM_CTX: int = 8192
OLLAMA_USE_MMAP: bool = False  # load each model fully into memory instead of paging it in from disk
OLLAMA_MAX_OUTPUT_TOKENS: int = 2200  # cap on the image-analysis reply; without it a runaway reply fills the whole context window
# What happens when the vision model's reply cannot be used. Each kind of trouble has its own steps (vlm/analyzer.py):
#  - a reply that ended at the output cap: one retry with different sampling, then the cap is raised step by step;
#  - a complete reply that does not fit the schema: the problems are sent back to the model and it answers again.
ANALYSIS_RETRIES: int = 1  # extra tries when a complete reply does not fit the schema
ANALYSIS_RETRY_TEMPERATURE: float = 0.4  # a retry must differ from the first try: at temperature 0 the same input gives the same reply
ANALYSIS_RETRY_REPEAT_PENALTY: float = 1.15  # sampling after a cut-off reply, which may be a loop; Ollama's own default is 1.1
ANALYSIS_CAP_STEP: float = 1.5  # each raise multiplies the output cap by this ...
ANALYSIS_CAP_RAISES: int = 3  # ... up to this many times, and never beyond what fits in the context window after the prompt
# Trouble reaching the model, each with its own step (vlm/analyzer.py): no reply in time -> wait longer; the server crashed
# or ran out of memory -> restart it, or free memory, and ask the same again once.
OLLAMA_TIMEOUT_STEP: int = 2  # each wait is this many times the one before ...
OLLAMA_TIMEOUT_RAISES: int = 2  # ... up to this many times
OLLAMA_RECOVERY_WAIT: int = 5  # seconds to wait after freeing memory before asking again

# Sent with every request to the VLM and to the judge model. num_ctx and use_mmap must be
# identical across all requests to one model: Ollama reloads it whenever one of them changes.
LLM_OPTIONS: dict = {"num_ctx": OLLAMA_NUM_CTX, "use_mmap": OLLAMA_USE_MMAP, "temperature": 0, "seed": 0}
EMBED_OPTIONS: dict = {"use_mmap": OLLAMA_USE_MMAP}

# --- Images ------------------------------------------------------------------
OCR_MIN_CONFIDENCE: float = 0.6  # OCR blocks scoring below this are dropped
OCR_TILE_SHARE: float = 0.5  # a tile is this share of the largest image the OCR engine reads without shrinking it: it finds text only above about 2% of the image side, so smaller tiles read smaller text
OCR_TILE_OVERLAP: float = 0.1  # share of a tile that the next one covers again, so text on a border is read whole in one of them
IMAGE_MEMORY_SHARE: float = 0.5  # share of the memory free right now that one image may use at its peak (image_processing/memory.py)
IMAGE_MEMORY_SHARE_LAST_TRY: float = 0.75  # the share for one last check of an image still too large after every model is unloaded (vlm/analyzer.py)


IMAGE_EDGE_CEILING: int = 1024  # the largest long side setup tries for the image sent to the vision model: beyond it the extra detail is not worth the time
IMAGE_MAX_LONG_EDGE: int = IMAGE_EDGE_CEILING  # the long side actually used: the largest the computer managed, never above the ceiling
IMAGE_JPEG_QUALITY: int = 85
ONE_MODEL_AT_A_TIME: bool = False  # True when setup found that the vision and embedding models do not fit in memory together (vlm/calibration.py)

# --- Measuring the computer at setup (vlm/calibration.py) ---------------------
CALIBRATION_EDGE_STEP: float = 0.75  # each smaller image tried is this share of the one before
CALIBRATION_MIN_EDGE: int = 336  # below this long side the vision model gets too little detail to describe a picture usefully
CALIBRATION_MEMORY_HEADROOM: float = 0.25  # share of the free memory that must stay free after a model is loaded into it; else it is paged in from disk
CALIBRATION_TIMEOUT_MARGIN: float = 2.0  # the wait is the measured time for the largest first reply times this
CALIBRATION_EMBED_TOKENS: int = 500  # length, in tokens, of the text the embedding model is given to see whether it fits in memory beside the vision model: about one stored description
CALIBRATION_TRIAL_TOKENS: int = 48  # tokens a trial reply is asked for: enough to time the model, not to finish a description


def apply_device_settings(device: dict) -> None:
    """Put what setup measured about this computer in place of the starting values, for this run.

    The two option dicts are changed in place: code that took them earlier sees the new values."""
    global OLLAMA_TIMEOUT, OLLAMA_NUM_CTX, OLLAMA_USE_MMAP, IMAGE_MAX_LONG_EDGE, ONE_MODEL_AT_A_TIME
    OLLAMA_TIMEOUT = device["timeout_seconds"]
    OLLAMA_NUM_CTX = device["num_ctx"]
    OLLAMA_USE_MMAP = device["use_mmap"]
    IMAGE_MAX_LONG_EDGE = device["image_long_edge"]
    ONE_MODEL_AT_A_TIME = bool(device.get("one_model_at_a_time"))  # None: not measured yet, so the two models are left to share the memory
    LLM_OPTIONS.update(num_ctx=OLLAMA_NUM_CTX, use_mmap=OLLAMA_USE_MMAP)
    EMBED_OPTIONS.update(use_mmap=OLLAMA_USE_MMAP)


def device_settings() -> dict:
    """The device settings in place right now, in the form ``apply_device_settings`` takes."""
    return {"timeout_seconds": OLLAMA_TIMEOUT, "num_ctx": OLLAMA_NUM_CTX, "use_mmap": OLLAMA_USE_MMAP,
            "image_long_edge": IMAGE_MAX_LONG_EDGE, "one_model_at_a_time": ONE_MODEL_AT_A_TIME}


@contextlib.contextmanager
def temporary_device_settings(**changes):
    """Run the block with some of the device settings changed (``timeout_seconds``, ``num_ctx``, ``use_mmap``,
    ``image_long_edge``), then put back the ones that were in place: a try with other settings never changes the usual ones."""
    before = device_settings()
    apply_device_settings({**before, **changes})
    try:
        yield
    finally:
        apply_device_settings(before)


# --- Database and indexing ---------------------------------------------------
SQLITE_BUSY_TIMEOUT_S: int = 30  # how long a connection waits for another process's write lock
SCAN_GUARD_MIN_FILES: int = 10  # a scan that finds at least this many indexed images gone ...
SCAN_GUARD_MAX_SHARE: float = 0.5  # ... and more than this share of the folder's indexed images asks before forgetting them
BACKLOG_MAX_ATTEMPTS: int = 3  # a file that fails this many times is left alone until it changes
MAX_CONSECUTIVE_SERVICE_FAILURES: int = 3  # indexing stops after this many Ollama failures in a row
SERVICE_TRIES: int = 3  # tries for an image whose failure is Ollama's or the machine's (each try runs the steps in vlm/analyzer.py), before the user is told
CONVERT_TIMEOUT: int = 300  # seconds the operating system's tool may take to convert one picture Pillow cannot open
RETRY_RATE_WINDOW: int = 10  # indexing stops when, among this many of the latest analysed images ...
RETRY_RATE_STOP: float = 0.5  # ... at least this share needed the model's reply retried: the model or the prompt is the problem, not the files
ERROR_LOG_MAX_ROWS: int = 1000  # oldest error_logs rows beyond this are deleted

# --- Search ------------------------------------------------------------------
DB_BATCH_ROWS: int = 1000  # rows read from the database at a time by search: the memory it uses is about this many rows, however many images are indexed
UNCERTAIN_TOP_K: int = 3  # results returned per search (the closest images)
SEARCH_MAX_MISSING_SKIPPED: int = 100  # closest images skipped because their file is gone, before the search gives up looking

# --- Versions ----------------------------------------------------------------
APP_VERSION: str = "0.1.0"  # stored with every indexed record (pipeline/versions.py)

# --- Launchers ---------------------------------------------------------------
# What messages tell the user to run, for the OS they are on.
SETUP_COMMAND: str = "imagent setup"
SCAN_COMMAND: str = "imagent scan"
ADD_COMMAND: str = "imagent add"

# --- Prompts -----------------------------------------------------------------
PROMPTS_DIR: Path = Path(__file__).parent / "prompts"
ANALYSIS_PROMPT: str = (PROMPTS_DIR / "analysis_prompt.txt").read_text(encoding="utf-8")
# The shape of the vision model's reply: a JSON schema sent to Ollama and used to check the reply. The fields live in this
# file and in the prompt only; no code names them, except the one below.
ANALYSIS_SCHEMA: dict = json.loads((PROMPTS_DIR / "analysis_schema.json").read_text(encoding="utf-8"))
SUMMARY_FIELD: str = "short_description"  # the field of the reply stored as the image's one-line summary, shown in search results
QUERY_PROMPT: str = (PROMPTS_DIR / "query_prompt.txt").read_text(encoding="utf-8")
