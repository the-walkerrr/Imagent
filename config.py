import json
import os
from pathlib import Path

# User-level storage directory for Imagent's database (analogous to ~/.claude).
# Overridable via IMAGENT_HOME for testing or non-standard setups.
IMAGENT_HOME: Path = Path(os.getenv("IMAGENT_HOME", str(Path.home() / ".imagent")))
DB_PATH: Path = IMAGENT_HOME / "imagent.db"
CONFIG_PATH: Path = IMAGENT_HOME / "config.json"

# User settings written by setup.py.
_settings: dict = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}

OLLAMA_HOST: str = _settings.get("ollama_host", "")
OLLAMA_MODEL: str = _settings.get("vlm_model", "")
OLLAMA_EMBEDDING_MODEL: str = _settings.get("embedding_model", "bge-m3")
OLLAMA_TIMEOUT: int = 300
OLLAMA_NUM_CTX: int = 8192
OLLAMA_USE_MMAP: bool = False  # mmap made the VLM's image encoder run out of GPU memory on 8 GB Macs (empty responses)

IMAGE_MAX_LONG_EDGE: int = int(os.getenv("IMAGE_MAX_LONG_EDGE", "1120"))
IMAGE_JPEG_QUALITY: int = int(os.getenv("IMAGE_JPEG_QUALITY", "85"))

ANALYSIS_PROMPT: str = (Path(__file__).parent / "analysis_prompt.txt").read_text(encoding="utf-8")
QUERY_PROMPT: str = (Path(__file__).parent / "query_prompt.txt").read_text(encoding="utf-8")
JUDGE_PROMPT: str = (Path(__file__).parent / "judge_prompt.txt").read_text(encoding="utf-8")

# --- Search confidence gating -----------------------------------------------
# Provisional, data-informed starting points derived from a 10-image / 4-query
# sample (see imagent's search-design notes) — NOT calibrated against a
# labeled golden set. Expect to need tightening once more real usage exists.
COSINE_ABS_FLOOR: float = 0.55   # below this, nothing is even plausibly related — skip the judge entirely
Z_CONFIDENT: float = 3.0         # diagnostic only (kept for future calibration)
MARGIN_CONFIDENT: float = 0.08   # diagnostic only (kept for future calibration)
Z_POOL_SIZE: int = 20            # candidates ranked 2..21 form the background pool for the z-score
Z_SIGMA_FLOOR: float = 0.02      # guards divide-by-~0 when the pool is unusually homogeneous
Z_MIN_POOL: int = 5              # below this many pool candidates, skip the z computation
UNCERTAIN_TOP_K: int = 3         # max candidates sent to the relevance judge per search
JUDGE_CONFIDENCE_MIN: float = 0.7  # minimum judge confidence required to accept a match
