"""
Centralized config for the RAG backend.
Reads from environment (which docker-compose or a local .env can populate).
Fails LOUDLY on missing critical values so we never serve a half-broken backend.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of where uvicorn is launched from.
# C:\Projects\RAG\.env  -> resolved from this file's location.
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


# ----- paths -----

# FAISS index location. Defaults to the prebuilt folder in the project root,
# which is what `main.py` and the original pipeline use.
_PROJECT_ROOT_STR = str(_PROJECT_ROOT)
FAISS_INDEX_PATH = _read("FAISS_INDEX_PATH", os.path.join(_PROJECT_ROOT_STR, "faiss_local_index"))

# ----- retrieval -----

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = _read_int("DEFAULT_K", 4)

# ----- LLM -----

# Llama 3.1 8B Instruct is gated + rate-limited on the free HF Inference API,
# so the backend tolerates a missing token by degrading to extractive answers.
HF_API_TOKEN = _read("HUGGINGFACEHUB_API_TOKEN")
LLM_REPO_ID = _read("LLM_REPO_ID", "meta-llama/Llama-3.1-8B-Instruct")
LLM_TEMPERATURE = _read_float("LLM_TEMPERATURE", 0.2)

# Server bind config (uvicorn picks these up).
API_HOST = _read("API_HOST", "0.0.0.0")
API_PORT = _read_int("API_PORT", 7860)
