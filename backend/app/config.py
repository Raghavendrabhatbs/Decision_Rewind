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
    value = os.getenv(key, default)
    return value.strip() if isinstance(value, str) else value


APP_NAME = get_env("APP_NAME", "DECISION-REWIND")
LLM_PROVIDER = get_env("LLM_PROVIDER", "groq")
LLM_BASE_URL = get_env("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = get_env("LLM_MODEL", "openai/gpt-oss-20b")
LLM_API_KEY = get_env("GROQ_API_KEY") or get_env("LLM_API_KEY")
LLM_TEMPERATURE = float(get_env("TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(get_env("MAX_TOKENS", "1024"))
