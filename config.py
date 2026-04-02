import os
from dotenv import load_dotenv

load_dotenv(override=True)

def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        raise EnvironmentError(
            f"Missing required env var: {key}\n"
            f"Copy .env.example → .env and fill in values."
        )
    return val

GROQ_API_KEY = _require("GROQ_API_KEY")
SERPAPI_KEY  = _require("SERPAPI_KEY")

MODEL_NAME       = os.getenv("GROQ_MODEL",       "llama-3.3-70b-versatile")
CURRENCY         = os.getenv("TRAVEL_CURRENCY",  "USD")
LANGUAGE         = os.getenv("TRAVEL_LANG",      "en")
PROVIDER_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "20"))
MAX_HISTORY      = int(os.getenv("MAX_HISTORY",    "10"))
LITE_MODE        = os.getenv("LITE_MODE", "false").lower() == "true"

SEARCH_CONFIGS = [
    {"sort_by": "1", "label": "best",     "emoji": "🏆"},
    {"sort_by": "2", "label": "cheapest", "emoji": "💰"},
    {"sort_by": "5", "label": "fastest",  "emoji": "⚡"},
]

FORCE_FAIL_SEARCHES: set = set()