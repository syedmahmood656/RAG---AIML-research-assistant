"""
Centralised config for the RAG backend.

Reads from the environment (docker-compose or a local .env populate it).
Missing optional values degrade gracefully (e.g. no token -> extractive answers).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of where uvicorn is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _read(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name, default)
    return val if val not in (None, "") else default


def _read_int(name: str, default: int) -> int:
    raw = _read(name)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _read_float(name: str, default: float) -> float:
    raw = _read(name)
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _clean_token(raw: str | None) -> str | None:
    """Return a usable token, or None for empty values and copy-pasted placeholders."""
    if not raw:
        return None
    token = raw.strip().strip('"').strip("'")
    placeholders = {"", "api key", "api_key", "your_token_here", "your-token-here"}
    if token.lower() in placeholders or token.lower().startswith("hf_xxxx"):
        return None
    return token


# ----- paths -----
FAISS_INDEX_PATH = _read("FAISS_INDEX_PATH", str(_PROJECT_ROOT / "faiss_local_index"))

# ----- retrieval -----
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match the index
DEFAULT_K = _read_int("DEFAULT_K", 4)
# Fetch k * FETCH_MULTIPLIER candidates, then drop duplicate chunks and keep k.
# (The prebuilt index contains duplicate chunks from papers cross-listed in cs.AI + cs.LG.)
FETCH_MULTIPLIER = max(1, _read_int("FETCH_MULTIPLIER", 3))

# ----- LLM (OpenRouter, OpenAI-compatible) -----
HF_API_TOKEN = _clean_token(_read("OPENROUTER_API_KEY"))  # name kept so rag.py needs no other edits
LLM_BASE_URL = _read("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_REPO_ID = _read("LLM_REPO_ID", "qwen/qwen3-next-80b-a3b-instruct:free")
LLM_PROVIDER = "openrouter"
LLM_TEMPERATURE = _read_float("LLM_TEMPERATURE", 0.2)
LLM_MAX_TOKENS = _read_int("LLM_MAX_TOKENS", 700)

# ----- server -----
API_HOST = _read("API_HOST", "0.0.0.0")
API_PORT = _read_int("API_PORT", 7860)
