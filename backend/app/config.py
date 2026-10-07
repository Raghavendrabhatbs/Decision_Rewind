import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = BASE_DIR / "app" / "ml" / "trained"
SQLITE_PATH = BASE_DIR / "decision_rewind.db"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


APP_NAME = get_env("APP_NAME", "DECISION-REWIND")
LLM_PROVIDER = get_env("LLM_PROVIDER", "groq")
LLM_MODEL = get_env("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_API_KEY = get_env("GROQ_API_KEY") or get_env("LLM_API_KEY")
LLM_TEMPERATURE = float(get_env("TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(get_env("MAX_TOKENS", "2048"))
