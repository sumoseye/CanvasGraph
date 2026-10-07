import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
STORE_PATH = RESULTS_DIR / "memory_store.pkl"
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key) or default


class Settings:
    GROQ_API_KEY = _env("GROQ_API_KEY")
    GROQ_MODEL = _env("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_BASE_URL = _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    TG_HOST = _env("TG_HOST")
    TG_USER = _env("TG_USER")
    TG_PASSWORD = _env("TG_PASSWORD")
    TG_GRAPH = _env("TG_GRAPH")
    GRAPH_BACKEND = _env("GRAPH_BACKEND", "memory")

    TOP_K = int(_env("TOP_K", "3"))
    HOPS = int(_env("HOPS", "2"))
    CHUNK_TOKENS = int(_env("CHUNK_TOKENS", "200"))
    CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "30"))
    PAGERANK_K = int(_env("PAGERANK_K", "8"))
    MAX_AGENT_STEPS = int(_env("MAX_AGENT_STEPS", "3"))
    TEST_LIMIT = int(_env("TEST_LIMIT", "6"))
    EVIDENCE_CONF_THRESHOLD = float(_env("EVIDENCE_CONF_THRESHOLD", "0.65"))
    ROUTER_ARGMAX_MIN = float(_env("ROUTER_ARGMAX_MIN", "0.6"))
    LLM_RETRIES = int(_env("LLM_RETRIES", "3"))

    # Generous limits for gpt-oss reasoning models
    CONTEXT_CHAR_LIMIT = int(_env("CONTEXT_CHAR_LIMIT", "2000"))
    ANSWER_MAX_TOKENS = int(_env("ANSWER_MAX_TOKENS", "1000"))
    JSON_MAX_TOKENS = int(_env("JSON_MAX_TOKENS", "800"))
    JUDGE_MAX_TOKENS = int(_env("JUDGE_MAX_TOKENS", "800"))

    Q_ID_FIELD = _env("Q_ID_FIELD", "id")
    Q_TEXT_FIELD = _env("Q_TEXT_FIELD", "question")
    Q_ANSWER_FIELD = _env("Q_ANSWER_FIELD", "ground_truth")
    Q_TYPE_FIELD = _env("Q_TYPE_FIELD", "type")


settings = Settings()