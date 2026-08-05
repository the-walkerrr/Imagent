import os
from pathlib import Path
import dotenv

dotenv.load_dotenv(".env")

# User-level storage directory for Imagent's database (analogous to ~/.claude).
# Overridable via IMAGENT_HOME for testing or non-standard setups.
IMAGENT_HOME: Path = Path(os.getenv("IMAGENT_HOME", str(Path.home() / ".imagent")))
DB_PATH: Path = IMAGENT_HOME / "imagent.db"

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "")
OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
OLLAMA_TIMEOUT: int = 300
OLLAMA_NUM_CTX: int = 8192

IMAGE_MAX_LONG_EDGE: int = int(os.getenv("IMAGE_MAX_LONG_EDGE", "1120"))
IMAGE_JPEG_QUALITY: int = int(os.getenv("IMAGE_JPEG_QUALITY", "85"))

ANALYSIS_PROMPT: str = (Path(__file__).parent / "analysis_prompt.txt").read_text(encoding="utf-8")
