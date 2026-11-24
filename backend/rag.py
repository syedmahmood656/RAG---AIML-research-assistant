"""
RAG core: load FAISS, retrieve (with de-duplication), and stream a grounded answer.

Design notes
- Retrieval runs ONCE per request; the same chunks feed both the "sources" panel and the LLM
  prompt (the original code retrieved twice and could show different chunks than it used).
- The LLM is called through huggingface_hub.InferenceClient (Inference Providers router).
  The old api-inference.huggingface.co endpoint used by huggingface-hub 0.25 is decommissioned.
- If the LLM is unreachable (no token, gated model, rate limit, network) we fall back to an
  extractive answer built from the retrieved chunks so the UI always shows something useful.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
from pathlib import Path
from typing import AsyncIterator, Dict, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from langchain_core.embeddings import Embeddings  # noqa: E402

from config import (  # noqa: E402
    DEFAULT_K,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    FETCH_MULTIPLIER,
    HF_API_TOKEN,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_REPO_ID,
    LLM_TEMPERATURE,
)

Result = Tuple[object, float]  # (langchain Document, L2 distance)

_EMBEDDINGS = None
_DB = None
_CLIENT = None
_LLM_ERROR: str | None = None
_INFO: Dict | None = None

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the question using ONLY the numbered "
    "context excerpts, which were retrieved from research papers. Cite the excerpts you use "
    "as [1], [2], ... If the excerpts do not contain the answer, say you don't know based on "
    "the indexed papers. Be concise and do not invent facts."
)


# --------------------------------------------------------------------------- loading

class _MiniLMEmbeddings(Embeddings):
    """sentence-transformers wrapper; normalised vectors, identical to the ones used to build the index."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        print(f"[rag] loading embedding model: {EMBEDDING_MODEL_NAME}", flush=True)
        _EMBEDDINGS = _MiniLMEmbeddings(EMBEDDING_MODEL_NAME)
    return _EMBEDDINGS


def _load_db():
    """Load FAISS once and cache it."""
    global _DB
    if _DB is None:
        from langchain_community.vectorstores import FAISS

        index_path = Path(FAISS_INDEX_PATH)
        if not (index_path / "index.faiss").exists():
            raise FileNotFoundError(
                f"FAISS index not found in {index_path}. "
                "Build it first: python data_ingestion_pipeline/main.py && python load_and_vectorbase/load.py"
            )
        print(f"[rag] loading FAISS index from {index_path}", flush=True)
        _DB = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=get_embeddings(),
            allow_dangerous_deserialization=True,  # the .pkl is our own file
        )
        print("[rag] FAISS index ready", flush=True)
    return _DB


class _OpenRouterClient:
    """OpenRouter adapter: walks through free models until one actually answers."""

    BAD = ("reasoning", "thinking", "preview", "-vl", "omni", "1.7b", "embed")
    MAX_ROUNDS = 3     # full passes over the model list
    ROUND_WAIT = 10    # seconds to wait between passes
    MAX_MODELS = 10    # models tried per pass

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI

        self._c = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=0)
        self._preferred = model
        self._models = None

    def _candidates(self):
        if self._models is None:
            import os

            names = [self._preferred]
            names += [m.strip() for m in os.environ.get("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
            try:
                for m in self._c.models.list().data:
                    mid = m.id
                    if mid.endswith(":free") and not any(b in mid.lower() for b in self.BAD):
                        names.append(mid)
            except Exception as e:  # noqa: BLE001
                print(f"[rag] could not list free models: {e}", flush=True)
            seen, ordered = set(), []
            for n in names:
                if n not in seen:
                    seen.add(n)
                    ordered.append(n)
            self._models = ordered[: self.MAX_MODELS]
            print(f"[rag] LLM candidates: {self._models}", flush=True)
        return self._models

    def chat_completion(self, messages, max_tokens, temperature, stream):
        import itertools
        import time

        last = None
        for rnd in range(self.MAX_ROUNDS):
            for model in self._candidates():
                try:
                    it = iter(
                        self._c.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=True,
                        )
                    )
                    buffered, got = [], False
                    for chunk in it:  # wait for the first real text
                        buffered.append(chunk)
                        if chunk.choices and chunk.choices[0].delta.content:
                            got = True
                            break
                    if not got:
                        raise RuntimeError("empty response")
                    print(f"[rag] answering with {model}", flush=True)
                    return itertools.chain(buffered, it)
                except Exception as e:  # noqa: BLE001
                    last = e
                    if getattr(e, "status_code", None) in (401, 403):
                        raise  # bad key: other models will not help
                    print(f"[rag] {model} failed ({type(e).__name__}); trying next", flush=True)
            if rnd < self.MAX_ROUNDS - 1:
                print(f"[rag] all models busy, waiting {self.ROUND_WAIT}s...", flush=True)
                time.sleep(self.ROUND_WAIT)
        raise last or RuntimeError("no model available")

def _get_client():
    """Return an OpenRouter client, or None (-> excerpts-only fallback)."""
    global _CLIENT, _LLM_ERROR
    if _CLIENT is not None or _LLM_ERROR is not None:
        return _CLIENT
    if not HF_API_TOKEN:
        _LLM_ERROR = "No OpenRouter API key configured - showing retrieved excerpts only."
        print(f"[rag] LLM disabled: {_LLM_ERROR}", flush=True)
        return None
    try:
        from config import LLM_BASE_URL

        _CLIENT = _OpenRouterClient(HF_API_TOKEN, LLM_BASE_URL, LLM_REPO_ID)
        print(f"[rag] LLM client ready: {LLM_REPO_ID} via {LLM_BASE_URL}", flush=True)
    except Exception as e:  # noqa: BLE001
        _LLM_ERROR = f"LLM client init failed ({type(e).__name__}: {e})"
        print(f"[rag] {_LLM_ERROR}", flush=True)
    return _CLIENT


# --------------------------------------------------------------------------- retrieval

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def dedupe_results(results: List[Result], k: int) -> List[Result]:
    """Drop chunks with identical text (keeping the best-ranked copy) and return at most k."""
    seen, out = set(), []
    for doc, score in results:
        key = hashlib.md5(_norm(doc.page_content).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append((doc, score))
        if len(out) == k:
            break
    return out


def l2_to_similarity(distance: float) -> float:
    """FAISS IndexFlatL2 returns squared L2; for unit vectors cosine = 1 - d/2."""
    return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))


def retrieve(question: str, k: int = DEFAULT_K) -> List[Result]:
    raw = _load_db().similarity_search_with_score(question, k=k * FETCH_MULTIPLIER)
    return dedupe_results(raw, k)


def format_sources(results: List[Result]) -> List[Dict]:
    out: List[Dict] = []
    for i, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata or {}
        out.append(
            {
                "rank": i,
                "title": meta.get("Title") or "Untitled excerpt",
                "has_metadata": bool(meta.get("Title")),
                "authors": meta.get("Authors", ""),
                "published": meta.get("Published", ""),
                "category": meta.get("category", ""),
                "entry_id": meta.get("entry_id", ""),
                "pdf_url": meta.get("pdf_url", ""),
                "score": round(l2_to_similarity(score), 3),
                "snippet": doc.page_content[:900],
            }
        )
    return out


# --------------------------------------------------------------------------- generation

def build_messages(question: str, results: List[Result]) -> List[Dict[str, str]]:
    blocks = []
    for i, (doc, _) in enumerate(results, start=1):
        title = (doc.metadata or {}).get("Title")
        header = f"[{i}] {title}" if title else f"[{i}]"
        blocks.append(f"{header}\n{doc.page_content.strip()}")
    context = "\n\n---\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context excerpts:\n\n{context}\n\nQuestion: {question}"},
    ]


def extractive_answer(results: List[Result]) -> str:
    """No-LLM fallback: the top excerpts, trimmed, as a readable answer."""
    if not results:
        return "I couldn't find any relevant context in the indexed papers."
    parts = []
    for i, (doc, _) in enumerate(results, start=1):
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
        title = (doc.metadata or {}).get("Title")
        label = f"**[{i}] {title}**" if title else f"**[{i}]**"
        parts.append(f"{label}\n\n{snippet}")
    return "\n\n".join(parts)


def _describe_error(e: Exception) -> str:
    msg = f"{type(e).__name__}: {str(e)[:200]}"
    low = str(e).lower()
    if "401" in low or "403" in low or "gated" in low or "access" in low:
        msg += " (check the token permissions and that you accepted the model licence)"
    elif "429" in low or "rate" in low:
        msg += " (rate limited - wait a minute or try another LLM_REPO_ID)"
    return msg


async def stream_answer(question: str, results: List[Result]) -> AsyncIterator[Tuple[str, str]]:
    """
    Yield ("token", text) fragments, or ("notice", message) when falling back to the
    extractive answer. Blocking HF calls run in worker threads so the event loop stays free.
    """
    client = _get_client()
    if client is None:
        yield ("notice", _LLM_ERROR or "LLM unavailable - showing retrieved excerpts only.")
        yield ("token", extractive_answer(results))
        return

    emitted = False
    try:
        messages = build_messages(question, results)
        stream = await asyncio.to_thread(
            lambda: client.chat_completion(
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                stream=True,
            )
        )
        it = iter(stream)
        while True:
            chunk = await asyncio.to_thread(next, it, None)
            if chunk is None:
                break
            delta = chunk.choices[0].delta.content if getattr(chunk, "choices", None) else None
            if delta:
                emitted = True
                yield ("token", delta)
        if not emitted:
            raise RuntimeError("the model returned an empty response")
    except Exception as e:  # noqa: BLE001 - degrade instead of killing the stream
        print(f"[rag] generation failed, falling back: {e}", flush=True)
        yield ("notice", f"LLM call failed - {_describe_error(e)}. Showing retrieved excerpts instead.")
        yield ("token", ("\n\n---\n\n" if emitted else "") + extractive_answer(results))


# --------------------------------------------------------------------------- misc

def warmup() -> None:
    """Load the index + embedder and run one query. Raises if the index is unusable."""
    _load_db().similarity_search("warmup", k=1)


def info() -> Dict:
    """Facts about the loaded index / LLM, shown in the UI sidebar."""
    global _INFO
    if _INFO is None:
        db = _load_db()
        sample = list(db.docstore._dict.values())[:25]
        _INFO = {
            "chunks": int(db.index.ntotal),
            "has_paper_metadata": any((d.metadata or {}).get("Title") for d in sample),
            "embedding_model": EMBEDDING_MODEL_NAME,
        }
    return {
        **_INFO,
        "llm_model": LLM_REPO_ID,
        "llm_configured": bool(HF_API_TOKEN),
        "llm_error": _LLM_ERROR,
    }
