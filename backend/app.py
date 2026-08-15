"""
FastAPI entrypoint for the RAG backend.

Endpoints:
    GET  /health   -> liveness probe
    POST /query    -> Server-Sent Events stream

SSE wire protocol:
    event: sources
    data: {"sources": [{rank, title, authors, ...}, ...]}

    event: token
    data: "<text fragment>"

    event: done
    data: {}

    event: error        (only on unrecoverable failure)
    data: {"message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

import rag
from config import API_HOST, API_PORT, DEFAULT_K

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-api")

app = FastAPI(title="ArXiv RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's natural-language question.")
    k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve.")


def _sse(event: str, payload) -> str:
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False)
    # SSE requires that newlines in `data` are escaped.
    return f"event: {event}\ndata: {payload.replace(chr(10), '\\n')}\n\n"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe: returns 503 if the index can't be loaded yet."""
    try:
        rag.warmup()
        return {"status": "ready"}
    except Exception as e:  # noqa: BLE001
        log.exception("readiness failed")
        return JSONResponse({"status": "not-ready", "error": str(e)}, status_code=503)


@app.post("/query")
async def query(req: QueryRequest):
    k = req.k or DEFAULT_K

    async def event_stream():
        # 1. Sources first.
        try:
            sources = await asyncio.to_thread(rag.retrieve_sources, req.question, k)
            yield _sse("sources", {"sources": sources})
        except Exception as e:  # noqa: BLE001
            log.exception("retrieval failed")
            yield _sse("error", {"message": f"Retrieval failed: {e}"})
            return

        # 2. Stream answer tokens.
        try:
            async for chunk in rag.stream_answer(req.question, k):
                yield _sse("token", chunk)
                await asyncio.sleep(0)  # flush
        except Exception as e:  # noqa: BLE001
            log.exception("generation failed")
            yield _sse("error", {"message": f"Generation failed: {e}"})
            return

        yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.on_event("startup")
async def _startup():
    log.info("startup: warming up FAISS index...")
    try:
        await asyncio.to_thread(rag.warmup)
        log.info("startup: warmup OK")
    except Exception as e:  # noqa: BLE001
        log.warning("startup: warmup failed (%s) — will load lazily on first request", e)


if __name__ == "__main__":
    # Allows: python backend/app.py
    import uvicorn
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=False)
