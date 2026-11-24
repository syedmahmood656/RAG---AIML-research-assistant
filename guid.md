# ArXiv RAG Research Assistant - Project Guide

A question-answering app over ~150 recent arXiv papers (cs.AI + cs.LG). You ask a question, the app finds the most relevant paper excerpts, and an LLM writes an answer **using only those excerpts**, with the sources shown next to it.

All commands below are for **Windows PowerShell**.

---

## 1. How it works

```
 PDFs from arXiv ──► extract text ──► split into 1000-char chunks ──► embed (MiniLM) ──► FAISS index
   (one-time ingestion pipeline)                                                           │
                                                                                           ▼
 Streamlit UI  ──question──►  FastAPI  ──►  embed question ──► top-k similar chunks (de-duplicated)
      ▲                          │                                        │
      │   SSE stream:            └──► LLM prompt = numbered excerpts + question ──► Llama 3.1
      └── sources, tokens, done ◄──────────── streamed answer (or extractive fallback) ◄──┘
```

**One question, step by step**

1. The UI sends `POST /query {question, k}` to the API.
2. The API embeds the question with `sentence-transformers/all-MiniLM-L6-v2` and asks FAISS for the nearest chunks. It fetches `3 x k` candidates, drops exact-duplicate chunks, and keeps `k`.
3. The API immediately streams a `sources` event (titles, match %, excerpts) so the UI can show them.
4. The excerpts are numbered `[1]..[k]` and sent with the question to the LLM, which is told to answer only from them and cite `[n]`.
5. Tokens stream back as `token` events and appear live in the chat bubble.
6. If the LLM is unavailable (no token, rate limit, network), the API sends a `notice` and shows the retrieved excerpts as the answer, so the UI is never empty.

### Folder map

| Path | Purpose |
|---|---|
| `backend/app.py` | FastAPI app: `/health`, `/ready`, `/info`, `/query` (SSE) |
| `backend/rag.py` | Core logic: load index, retrieve, de-duplicate, prompt, stream answer, fallback |
| `backend/config.py` | Environment-driven settings (reads `.env` from the project root) |
| `backend/sse.py` | SSE message encoder (always JSON-encodes the payload) |
| `frontend/app.py` | Streamlit chat UI (dark theme, source cards, status panel) |
| `frontend/sse_client.py` | SSE parser + API helpers used by the UI |
| `faiss_local_index/` | Pre-built vector index (`index.faiss` + `index.pkl`) |
| `data_ingestion_pipeline/` | Downloads papers from arXiv, extracts text, chunks it |
| `load_and_vectorbase/load.py` | Embeds chunks and writes the FAISS index |
| `evaluation/` | Retrieval evaluation script and question set |
| `tests/` | Unit tests (no model / network needed) |
| `scripts/setup.ps1`, `scripts/start.ps1` | One-command setup and start for PowerShell |
| `main.py` | Command-line version of the same pipeline |

---

## 2. Prerequisites

- **Windows 10/11** with **PowerShell**
- **Python 3.10 - 3.13** (3.11 or 3.12 recommended). Check: `python --version`
- ~3 GB free disk (PyTorch + the embedding model)
- *Optional but recommended:* a free **Hugging Face account** for generated answers. Without a token the app still works in retrieval-only mode.
- *Optional:* Docker Desktop, if you prefer containers (section 6)

---

## 3. Installation - step by step

### Step 1 - Open the project folder

```powershell
cd C:\path\to\RAG-research-assistant
```

(Adjust the path to wherever you unzipped it. You should see `guid.md`, `backend`, `frontend`, ...)

### Step 2 - Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell says *"running scripts is disabled on this system"*, allow scripts **for this window only** and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Your prompt should now start with `(venv)`.

### Step 3 - Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first install downloads PyTorch and takes several minutes. Verify:

```powershell
python -c "import fastapi, streamlit, faiss, sentence_transformers, huggingface_hub; print('imports OK')"
```

### Step 4 - Add your Hugging Face token (optional)

```powershell
Copy-Item .env.example .env
notepad .env
```

1. Go to <https://huggingface.co/settings/tokens> -> **Create new token** -> type **Fine-grained**.
2. Tick **"Make calls to Inference Providers"** and create it.
3. Open <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and accept the licence (the model is gated).
4. Paste the token into `.env`:

```
HUGGINGFACEHUB_API_TOKEN=hf_yourRealTokenHere
```

To use a different chat model, uncomment and edit `LLM_REPO_ID` in `.env` (pick any chat model that lists an Inference Provider on its Hugging Face page). Leave the token empty to run retrieval-only.

### Step 5 - Start the backend (terminal 1)

```powershell
python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 7860
```

The first start downloads the ~90 MB embedding model and loads the index (about 30 s). Wait for `Application startup complete.`

Check it from a **new** PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:7860/health
Invoke-RestMethod http://127.0.0.1:7860/info
```

### Step 6 - Start the UI (terminal 2)

```powershell
cd C:\path\to\RAG-research-assistant
.\venv\Scripts\Activate.ps1
python -m streamlit run frontend/app.py --server.port 8501
```

Open **<http://localhost:8501>**. The sidebar shows API status, chunk count and whether the LLM is connected.

### Shortcut - steps 2-6 in two commands

```powershell
.\scripts\setup.ps1     # venv + dependencies + .env (one time)
.\scripts\start.ps1     # opens the API and the UI in two windows
```

### Test the API directly (optional)

```powershell
$body = @{ question = "What is retrieval augmented generation?"; k = 3 } | ConvertTo-Json
Invoke-WebRequest -Uri http://127.0.0.1:7860/query -Method Post -ContentType "application/json" -Body $body |
    Select-Object -ExpandProperty Content
```

(`Invoke-WebRequest` waits for the whole stream, then prints all events at once. Interactive docs: <http://127.0.0.1:7860/docs>.)

### Command-line version

```powershell
python main.py "What are the computational challenges of Seiberg dualities?" -k 4
```

---

## 4. Run the tests and the evaluation

```powershell
# Unit tests - fast, no model or network needed
python -m unittest discover -s tests -v

# Retrieval evaluation - needs the embedding model (backend need not be running)
python evaluation/eval_retrieval.py
python evaluation/eval_retrieval.py -k 5 -n 100
```

The evaluation reports:

| Check | What it tells you |
|---|---|
| **Keyword hit@k** | For each question in `evaluation/questions.json`, does a top-k excerpt contain an expected keyword? Add your own questions to that file. |
| **Self-retrieval** | Searches for a sentence taken from a chunk; the chunk should come back at rank 1. A sanity check that embedder and index match (expect ~100%). |
| **Paper-level** | Searches by paper title and checks that paper's chunks appear. Only runs once the index has titles (section 5). |
| **Latency** | Median / p95 retrieval time. |

These measure *retrieval*. To evaluate *answer quality* (faithfulness, relevance), the next step is an LLM-as-judge or a framework such as Ragas, using a hand-written set of question / reference-answer pairs.

---

## 5. Rebuild the index (recommended once)

The **pre-built index in the zip contains no paper titles, authors or links** - only the chunk text (it was built with a loader that dropped the metadata), so the UI can only show "Untitled excerpt". It also contains ~10% duplicate chunks. The fixed pipeline rebuilds it correctly:

```powershell
# 1. Download ~150 papers and chunk them (10-30 min: arXiv rate-limits to one request per 3 s)
python data_ingestion_pipeline/main.py

# 2. Embed the chunks and write faiss_local_index/ (a few minutes on CPU)
python load_and_vectorbase/load.py
```

Then restart the backend. The sidebar warning disappears and sources show titles, authors, dates and arXiv links.

Notes: this fetches the *newest* papers, so the corpus will differ from the shipped one (update `evaluation/questions.json` to match). Change `CATEGORIES`, `PAPERS_PER_CATEGORY`, `CHUNK_SIZE` etc. in `data_ingestion_pipeline/config.py`.

---

## 6. Run with Docker (alternative)

```powershell
Copy-Item .env.example .env
notepad .env                      # optional: add your token
docker compose up --build         # first build takes a while
```

- UI: <http://localhost:8501> - API docs: <http://localhost:7860/docs>
- Logs: `docker compose logs -f api`
- Stop: `docker compose down`

The index folder is mounted read-only, so after rebuilding the index just run `docker compose restart api`.

---

## 7. Configuration (`.env` or environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `HUGGINGFACEHUB_API_TOKEN` (or `HF_TOKEN`) | empty | Token for generated answers. Empty -> retrieval-only. |
| `LLM_REPO_ID` | `meta-llama/Llama-3.1-8B-Instruct` | Chat model to use. |
| `LLM_PROVIDER` | `auto` | Inference provider (`auto` lets Hugging Face route the request). |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature. |
| `LLM_MAX_TOKENS` | `700` | Maximum answer length. |
| `DEFAULT_K` | `4` | Excerpts retrieved per question. |
| `FETCH_MULTIPLIER` | `3` | Candidates fetched before de-duplication (`k x this`). |
| `FAISS_INDEX_PATH` | `<project>/faiss_local_index` | Index location. |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `7860` | Server bind settings. |

For the UI: `API_URL` (default `http://localhost:7860`).

---

## 8. API reference

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status":"ok"}` - liveness |
| `GET /ready` | `{"status":"ready"}` or 503 if the index cannot be loaded |
| `GET /info` | Chunk count, whether the index has paper metadata, LLM model / status |
| `POST /query` | Body `{"question": "...", "k": 4}` -> Server-Sent Events stream |

`/query` events - **every `data:` payload is JSON**:

```
event: sources   data: {"sources":[{"rank":1,"title":"...","score":0.83,"authors":"...","snippet":"..."}]}
event: notice    data: {"message":"..."}            (only if falling back to retrieval-only)
event: token     data: "first chunk of the answer"
event: token     data: " next chunk\n"
event: done      data: {"elapsed_ms":2841,"mode":"llm"}
event: error     data: {"message":"..."}            (unrecoverable failure)
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Activate.ps1 cannot be loaded ... scripts is disabled` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or skip activation and call `.\venv\Scripts\python.exe` directly. |
| Sidebar shows **API offline** | Start the backend (step 5). Confirm with `Invoke-RestMethod http://127.0.0.1:7860/health`. |
| Port already in use | `Stop-Process -Id (Get-NetTCPConnection -LocalPort 7860).OwningProcess -Force` (use 8501 for the UI). |
| Yellow notice "No Hugging Face token configured" | Put a real token in `.env` (step 4) and restart the backend. |
| Notice "LLM call failed ... 401 / 403" | Token lacks the *Inference Providers* permission, or the Llama licence has not been accepted. |
| Notice "LLM call failed ... 429" | Rate limit or free credits used up. Wait, or set another `LLM_REPO_ID`. |
| Sources show "Untitled excerpt" | The shipped index has no metadata - rebuild it (section 5). |
| `faiss` / `pymupdf` fail to install | Use Python 3.11 or 3.12 for the venv: `py -3.12 -m venv venv`. |
| `/ready` returns 503 | Index missing or corrupt; check `FAISS_INDEX_PATH` and rebuild (section 5). |
| First start is very slow | One-time download of the embedding model (~90 MB). Later starts take seconds. |

---

## 10. Known limitations and ideas to improve

- Each paper is truncated to its first ~20,000 characters (about 8-10 pages), so late sections and results tables are often missing.
- Retrieval is plain dense similarity. Adding a **cross-encoder re-ranker** and/or **hybrid BM25 + vector** search would improve precision.
- PDF text extraction keeps headers, footnotes and equations as noise; a cleaner parser (e.g. `pymupdf4llm`) would help chunk quality.
- The corpus is a small snapshot, so many questions are simply out of scope. The assistant is instructed to say it doesn't know in that case.
- Conversations are single-turn: earlier messages are shown in the UI but are not sent to the LLM as context.
- There is no answer-quality evaluation yet (see section 4).

---

## 11. What was fixed in this version

| Area | Problem | Fix |
|---|---|---|
| **Docker build** | `backend/` and `frontend/` Dockerfiles copy a `requirements.txt` that did not exist - `docker compose build` failed. | Added `backend/requirements.txt` and `frontend/requirements.txt`. |
| **`.gitignore`** | Ignored `faiss_local_index/`, `docker-compose.yml`, both Dockerfiles and the Streamlit config - the repo could not run when cloned. | Ignore only secrets, venvs, caches and `data/`. |
| **Python 3.11** | `app.py` used a backslash inside an f-string expression - a `SyntaxError` before Python 3.12, so the API crashed in the Docker image (Python 3.11). | New `sse.py` encoder. |
| **Broken newlines** | Token newlines were replaced by a literal `\n` and never decoded, so answers showed `\n` text. | Every SSE payload is JSON-encoded and decoded. |
| **Index metadata** | The index has only `source` and `seq_num`: no titles, authors or links, so the Sources panel was always "Untitled". `load.py` used `JSONLoader` without a `metadata_func`. | `load.py` keeps all metadata; UI warns until the index is rebuilt. |
| **`jq` missing** | `load.py` needed the `jq` package, absent from requirements and hard to install on Windows. | Replaced `JSONLoader` with plain `json`. |
| **Duplicate chunks** | ~10% of indexed chunks (383 of 3,592) were exact duplicates (papers cross-listed in cs.AI and cs.LG), wasting top-k slots. | De-duplicate at ingestion, index build and query time. |
| **LLM never worked** | Pinned `huggingface-hub==0.25.1` calls the retired `api-inference.huggingface.co`, so every call failed and silently fell back to extractive answers. | `InferenceClient` with a current `huggingface-hub` (Inference Providers). |
| **Event loop blocked** | Streaming used a synchronous loop inside an async function, stalling the server during generation. | Blocking calls run in worker threads. |
| **Double retrieval** | Retrieval ran twice per question (once for sources, once for the prompt). | Retrieve once and reuse. |
| **Weak prompt** | Excerpts were not numbered, so the model could not cite. | Numbered excerpts, citation instruction, system prompt. |
| **Token handling** | `main.py` hard-coded `'API KEY'` in source; the copied `.env.example` placeholder looked like a real token. | `main.py` rewritten to use `.env`; placeholders ignored. |
| **Required token** | `docker-compose.yml` refused to start without a token, contradicting the fallback design. | Token optional. |
| **UI** | Latest answer vanished on any widget change; corrupted emoji (`�`); one byte read at a time. | Rewritten chat UI with persistent history. |
| **Pipeline paths** | Relative paths depended on the working directory; stale comments. | Absolute paths from the project root. |
