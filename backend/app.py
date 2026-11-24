"""
FastAPI entrypoint for the RAG backend.

Endpoints
    GET  /health  liveness probe (cheap)
    GET  /ready   readiness probe (touches FAISS), 503 if the index is unusable
    GET  /info    index + LLM facts for the UI
    POST /query   Server-Sent Events stream

SSE protocol - every `data:` line is JSON:
    event: sources   data: {"sources": [{rank, title, score, snippet, ...}, ...]}
    event: notice    data: {"message": "..."}         (optional: LLM unavailable -> fallback)
    event: token     data: "<text fragment>"
    event: done      data: {"elapsed_ms": 1234, "mode": "llm" | "extractive"}
    event: error     data: {"message": "..."}          (unrecoverable failure)
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import rag
from config import API_HOST, API_PORT, DEFAULT_K
from sse import encode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("startup: warming up FAISS index and embedder...")
    try:
        await asyncio.to_thread(rag.warmup)
        log.info("startup: warmup OK")
    except Exception as e:  # noqa: BLE001
        log.warning("startup: warmup failed (%s) - /ready will report 503", e)
    yield


app = FastAPI(title="ArXiv RAG API", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Natural-language question.")
    k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    try:
        await asyncio.to_thread(rag.warmup)
        return {"status": "ready"}
    except Exception as e:  # noqa: BLE001
        log.exception("readiness failed")
        return JSONResponse({"status": "not-ready", "error": str(e)}, status_code=503)


@app.get("/info")
async def info():
    try:
        return await asyncio.to_thread(rag.info)
    except Exception as e:  # noqa: BLE001
        log.exception("info failed")
        return JSONResponse({"error": str(e)}, status_code=503)


@app.post("/query")
async def query(req: QueryRequest):
    question = req.question.strip()
    k = req.k or DEFAULT_K

    async def event_stream():
        started = time.perf_counter()

        # 1. Retrieve once; sources are sent before the first token.
        try:
            results = await asyncio.to_thread(rag.retrieve, question, k)
            yield encode("sources", {"sources": rag.format_sources(results)})
        except Exception as e:  # noqa: BLE001
            log.exception("retrieval failed")
            yield encode("error", {"message": f"Retrieval failed: {e}"})
            return

        # 2. Stream the answer.
        mode = "llm"
        try:
            async for kind, text in rag.stream_answer(question, results):
                if kind == "notice":
                    mode = "extractive"
                    yield encode("notice", {"message": text})
                else:
                    yield encode("token", text)
        except Exception as e:  # noqa: BLE001
            log.exception("generation failed")
            yield encode("error", {"message": f"Generation failed: {e}"})
            return

        yield encode("done", {"elapsed_ms": int((time.perf_counter() - started) * 1000), "mode": mode})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=False)
