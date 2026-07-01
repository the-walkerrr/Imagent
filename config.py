import os
from pathlib import Path
import dotenv

dotenv.load_dotenv(".env")

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "")
OLLAMA_TIMEOUT: int = 300
OLLAMA_NUM_CTX: int = 8192

ANALYSIS_PROMPT: str = (Path(__file__).parent / "analysis_prompt.txt").read_text(encoding="utf-8")
