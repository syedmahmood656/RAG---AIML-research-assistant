"""
RAG core: load FAISS, retrieve, and stream an answer.

Design notes:
- This module must work whether it's imported as `backend.rag` (package) or
  imported as a top-level `rag` (script). The original code used
  `from backend.config import ...`, which silently broke the second case.
- If the LLM is unreachable (no token, gated model, rate-limit, network),
  we fall back to an extractive answer built from the top retrieved chunks
  so the UI still works end-to-end.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import AsyncIterator, Dict, List

# Make `config` importable whether this file is run as a package member
# (backend.rag) or directly as a script (python backend/rag.py).
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
for p in (str(_THIS_DIR), str(_PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (  # noqa: E402  (intentional: after sys.path tweak)
    DEFAULT_K,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    HF_API_TOKEN,
    LLM_REPO_ID,
    LLM_TEMPERATURE,
)

# LangChain imports are deferred so that a missing optional dep doesn't
# kill the health endpoint. We only need them inside the actual functions.
_EMBEDDINGS = None
_DB = None
_MODEL = None
_LLM_INIT_ERROR: Exception | None = None


def _load_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        print(f"[rag] loading embedding model: {EMBEDDING_MODEL_NAME}", flush=True)
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS


def _load_db():
    """Load FAISS once and cache it."""
    global _DB
    if _DB is None:
        from langchain_community.vectorstores import FAISS
        index_path = Path(FAISS_INDEX_PATH)
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                f"Build it first (see load_and_vectorbase/load.py)."
            )
        print(f"[rag] loading FAISS index from {index_path}", flush=True)
        _DB = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=_load_embeddings(),
            allow_dangerous_deserialization=True,
        )
        print("[rag] FAISS index ready", flush=True)
    return _DB


def _load_model():
    """Load the LLM. Returns None on failure (extractive fallback will be used)."""
    global _MODEL, _LLM_INIT_ERROR
    if _MODEL is not None or _LLM_INIT_ERROR is not None:
        return _MODEL

    if not HF_API_TOKEN or HF_API_TOKEN.strip().lower() in {"", "api key", "api_key"}:
        _LLM_INIT_ERROR = RuntimeError("HUGGINGFACEHUB_API_TOKEN not set")
        print(f"[rag] LLM disabled: {_LLM_INIT_ERROR}", flush=True)
        return None

    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", HF_API_TOKEN)
        print(f"[rag] connecting to LLM: {LLM_REPO_ID}", flush=True)
        endpoint = HuggingFaceEndpoint(
            repo_id=LLM_REPO_ID,
            huggingfacehub_api_token=HF_API_TOKEN,
            temperature=LLM_TEMPERATURE,
        )
        _MODEL = ChatHuggingFace(llm=endpoint)
        print("[rag] LLM ready", flush=True)
    except Exception as e:  # noqa: BLE001  we want to swallow & degrade
        _LLM_INIT_ERROR = e
        print(f"[rag] LLM init failed, using extractive fallback: {e}", flush=True)
    return _MODEL


def _build_chain(k: int):
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    db = _load_db()
    model = _load_model()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    prompt = PromptTemplate(
        template=(
            "You are a helpful research assistant.\n"
            "Answer ONLY from the provided context, which was retrieved from "
            "research papers. If the context is insufficient, say you don't know.\n\n"
            "{context}\n"
            "Question: {question}\n"
        ),
        input_variables=["context", "question"],
    )

    parallel = RunnableParallel(
        {
            "context": retriever
            | RunnableLambda(lambda docs: "\n\n---\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
    )
    parser = StrOutputParser()
    return parallel | prompt | model | parser


def _extractive_answer(question: str, k: int) -> str:
    """
    No-LLM fallback. Returns a concise answer stitched from the top snippets
    so the UI demo is never empty when the LLM is down.
    """
    docs = _load_db().similarity_search(question, k=k)
    if not docs:
        return "I couldn't find any relevant context in the indexed papers."
    bullets = []
    for i, d in enumerate(docs, 1):
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
        title = (d.metadata or {}).get("Title", "Untitled")
        bullets.append(f"[{i}] {title}\n{snippet}")
    return (
        "_(extractive fallback — set HUGGINGFACEHUB_API_TOKEN for a generated answer)_\n\n"
        + "\n\n".join(bullets)
    )


def retrieve_sources(question: str, k: int = DEFAULT_K) -> List[Dict]:
    docs = _load_db().similarity_search(question, k=k)
    out: List[Dict] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        out.append(
            {
                "rank": i,
                "title": meta.get("Title", "Untitled"),
                "authors": meta.get("Authors", "Unknown"),
                "published": meta.get("Published", ""),
                "category": meta.get("category", ""),
                "entry_id": meta.get("entry_id", ""),
                "pdf_url": meta.get("pdf_url", ""),
                "snippet": doc.page_content[:800],
            }
        )
    return out


async def stream_answer(question: str, k: int = DEFAULT_K) -> AsyncIterator[str]:
    """
    Yield answer fragments. If the LLM is unavailable, yield the extractive
    fallback as a single chunk so the UI still renders something useful.
    """
    model = _load_model()
    if model is None:
        yield _extractive_answer(question, k)
        return

    chain = _build_chain(k)
    try:
        for chunk in chain.stream(question):
            if chunk:
                yield chunk
    except Exception as e:  # noqa: BLE001
        # LLM call failed at runtime (e.g. HF rate limit / 503). Don't kill
        # the stream — fall back to extractive and keep going.
        print(f"[rag] generation failed, falling back: {e}", flush=True)
        yield "\n\n_(LLM call failed, showing retrieved context instead)_\n\n"
        yield _extractive_answer(question, k)


def warmup() -> None:
    """Touch the index once at startup so the first request is fast."""
    try:
        _load_db().similarity_search("warmup", k=1)
    except Exception as e:  # noqa: BLE001
        print(f"[rag] warmup failed: {e}", flush=True)
