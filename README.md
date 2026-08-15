# ArXiv RAG — Frontend + API

Retrieval-Augmented-Generation over ~150 recent ArXiv cs.AI / cs.LG papers.

The original CLI lives in `main.py` (and the data-ingestion modules under
`data_ingestion_pipeline/` / `load_and_vectorbase/`). This repo also ships
a FastAPI backend that exposes the same RAG chain over HTTP, plus a
Streamlit UI to drive it.

```
.
├── backend/                  FastAPI app (RAG chain + SSE streaming)
│   ├── app.py                endpoints: /health, /ready, /query
│   ├── rag.py                FAISS load + retriever + LLM chain (extractive fallback)
│   ├── config.py             env-driven config, reads .env from project root
│   ├── requirements.txt      backend-only deps (for Docker)
│   └── Dockerfile
├── frontend/                 Streamlit UI
│   ├── app.py
│   └── Dockerfile
├── faiss_local_index/        Pre-built FAISS index (index.faiss + index.pkl)
├── data_ingestion_pipeline/  Original LangChain-based paper fetch + chunking
├── load_and_vectorbase/      Builds the FAISS index from `data/`
├── data/                     Raw + chunked paper JSONs (gitignored)
├── docker-compose.yml
├── .env.example
├── requirements.txt          Consolidated deps for local (non-Docker) installs
└── main.py                   The original CLI; still runs standalone
```

---

## Quick start (local, no Docker) — recommended

This is the path I verified end-to-end. Open **two terminals** in
`C:\Projects\RAG`.

### 1. Configure

Copy `.env.example` to `.env` and put a real `hf_...` token in it:

```bash
cp .env.example .env
# then edit .env and replace hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The token needs read access to `meta-llama/Llama-3.1-8B-Instruct`. The
model is gated — accept its license at
<https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> from your
HuggingFace account first.

If you don't have (or don't want to use) a gated model, set
`LLM_REPO_ID=mistralai/Mistral-7B-Instruct-v0.3` in `.env` — the backend
will use it instead.

### 2. Install deps into your venv

```bash
cd C:\Projects\RAG
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Terminal 1 — start the backend

```bash
cd C:\Projects\RAG
.\venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 7860
```

- First start takes **~30 seconds** because the embedding model
  (`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB) downloads into the
  HF cache. Subsequent starts are instant.
- Wait for `INFO:     Application startup complete.`
- If the LLM is unreachable (no token, gated, rate-limited), the backend
  still starts and serves an **extractive fallback** (top-k snippets)
  instead of a generated answer, so the UI is never empty.

Sanity check (third terminal, or browser):

```bash
curl http://127.0.0.1:7860/health
# {"status":"ok"}
```

### 4. Terminal 2 — start the UI

```bash
cd C:\Projects\RAG
.\venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```

Open **http://localhost:8501** in your browser. The UI talks to the API
on `http://localhost:7860` automatically.

---

## Quick start (Docker)

If you prefer containers:

```bash
cp .env.example .env      # then edit .env
docker compose up --build
```

Open <http://localhost:8501>. The API runs on
<http://localhost:7860/docs> (Swagger UI). The compose file mounts
`./faiss_local_index` read-only into the api container, so you can
rebuild the index on the host and restart the api to pick it up.

---

## API contract

`POST /query` accepts:

```json
{ "question": "What are Seiberg dualities?", "k": 4 }
```

Returns a Server-Sent Events stream:

```
event: sources
data: {"sources":[{"rank":1,"title":"...","authors":"...","snippet":"..."}]}

event: token
data: "first chunk of the answer"

event: token
data: " next chunk"

event: done
data: {}
```

On unrecoverable failure you get a single `event: error` with the
message. The UI surfaces these inline rather than hanging.

Other endpoints:

- `GET /health` → `{"status":"ok"}` (cheap liveness probe)
- `GET /ready`  → `{"status":"ready"}` or 503 (touches FAISS)

---

## Configuration (env vars)

All of these are optional — sensible defaults are baked in.

| Var | Default | Purpose |
|---|---|---|
| `HUGGINGFACEHUB_API_TOKEN` | — | HF token. If missing/empty, the backend falls back to extractive answers. |
| `LLM_REPO_ID` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace chat model. |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature. |
| `FAISS_INDEX_PATH` | `<project>/faiss_local_index` | Where to load the index from. |
| `DEFAULT_K` | `4` | Chunks retrieved per query. |
| `API_HOST` | `0.0.0.0` | uvicorn bind host. |
| `API_PORT` | `7860` | uvicorn bind port. |

---

## Rebuilding the index

The prebuilt `faiss_local_index/` is committed. To rebuild from scratch:

```bash
python data_ingestion_pipeline/main.py        # fetch + chunk papers into data/
python load_and_vectorbase/load.py            # embed + write FAISS index
```

Then restart the backend (or the API container) — the new index will be
loaded on the next startup warmup.

---

## Notes

- **Extractive fallback**: if the LLM call fails for any reason (no
  token, 401/403 on a gated model, HF rate-limit 503, network error),
  the backend returns the top retrieved snippets as the answer. You'll
  see a small italic note at the top of the answer in the UI.
- **Rate limits**: HuggingFace's free inference tier is heavily
  rate-limited. If you see intermittent `Generation failed` in the UI,
  either wait a minute or switch to a different `LLM_REPO_ID`.
- **CLI**: `main.py` is preserved as-is — `python main.py` still works
  for one-off scripted queries against the same FAISS index.
