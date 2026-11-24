# ArXiv RAG Research Assistant - Project Guide

A question-answering app over **138 recent arXiv papers** (cs.AI + cs.LG). You ask a question, the app finds the most relevant paper excerpts, and an LLM writes an answer **using only those excerpts**, with the sources shown next to it.

All commands are for **Windows PowerShell**. Replace `C:\Projects\RAG` with wherever your project folder is.

---

## 1. How it works

```
 PDFs from arXiv ─► extract text ─► split into 700-char chunks ─► embed (MiniLM) ─► FAISS index
   (one-time ingestion pipeline)                                                        │
                                                                                        ▼
 Streamlit UI ──question──► FastAPI ──► embed question ──► top-k similar chunks (de-duplicated)
      ▲                        │                                      │
      │   SSE stream:          └──► LLM prompt = numbered excerpts + question ──► Llama 3.1
      └── sources, tokens, done ◄──────────── streamed answer (or excerpts-only fallback) ◄─┘
```

**One question, step by step**

1. The UI sends `POST /query {question, k}` to the API.
2. The API embeds the question with `sentence-transformers/all-MiniLM-L6-v2` and asks FAISS for the nearest chunks. It fetches `3 x k` candidates, drops exact-duplicate chunks and keeps `k`.
3. The API immediately streams a `sources` event (titles, match %, excerpts) so the UI shows them first.
4. The excerpts are numbered `[1]..[k]` and sent with the question to the LLM, which is told to answer only from them and cite `[n]`.
5. Tokens stream back as `token` events and appear live in the chat bubble.
6. If the LLM is unavailable (no token, no credits, rate limit, network), the API sends a `notice` and shows the retrieved excerpts as the answer, so the UI is never empty.

### Folder map

| Path | Purpose |
|---|---|
| `backend/app.py` | FastAPI app: `/health`, `/ready`, `/info`, `/query` (SSE) |
| `backend/rag.py` | Core logic: load index, retrieve, de-duplicate, prompt, stream answer, fallback |
| `backend/config.py` | Environment-driven settings (reads `.env` from the project root) |
| `backend/sse.py` | SSE message encoder (payloads are always JSON) |
| `frontend/app.py` | Streamlit chat UI (dark theme, source cards, status panel) |
| `frontend/sse_client.py` | SSE parser and API helpers for the UI |
| `faiss_local_index/` | Vector index (`index.faiss` + `index.pkl`) |
| `data_ingestion_pipeline/` | Downloads papers from arXiv, extracts text, chunks it |
| `load_and_vectorbase/load.py` | Embeds chunks and writes the FAISS index |
| `load_and_vectorbase/rechunk.py` | Re-splits downloaded papers with a new chunk size (no re-download) |
| `load_and_vectorbase/test.py` | Quick retrieval smoke test |
| `evaluation/` | Retrieval benchmark and question set |
| `tests/` | Unit tests (no model or network needed) |
| `scripts/setup.ps1`, `scripts/start.ps1` | One-command setup and start |
| `main.py` | Command-line version of the pipeline |

---

## 2. Prerequisites

- **Windows 10/11** with **PowerShell**
- **Python 3.10 - 3.13** (3.11 or 3.12 recommended). Check with `python --version`
- About 3 GB free disk (PyTorch and the embedding model)
- *Optional:* a free **Hugging Face account** for generated answers. Without a token the app runs in retrieval-only mode.
- *Optional:* Docker Desktop (section 6)

---

## 3. Installation - step by step

### Step 1 - Open the project folder

```powershell
cd C:\Projects\RAG
```

You should see `guid.md`, `backend`, `frontend`, `scripts`, and so on:

```powershell
Get-ChildItem
```

### Step 2 - Allow scripts for this window

PowerShell blocks unsigned scripts by default. This lasts only for the current window and changes nothing permanently. **Repeat it every time you open a new PowerShell window.**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 3 - Create the virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Your prompt should now start with `(venv)`.

### Step 4 - Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first install downloads PyTorch and takes several minutes. Verify:

```powershell
python -c "import fastapi, streamlit, faiss, sentence_transformers, huggingface_hub; print('imports OK')"
```

### Step 5 - Add your Hugging Face token (optional)

```powershell
Copy-Item .env.example .env
notepad .env
```

1. Open <https://huggingface.co/settings/tokens> and create a **fine-grained** token.
2. Tick **"Make calls to Inference Providers"**.
3. Open <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and accept the licence (the model is gated).
4. Paste the token into `.env`:

```
HUGGINGFACEHUB_API_TOKEN=hf_yourRealTokenHere
```

Leave it empty to run retrieval-only. To use another chat model, uncomment and edit `LLM_REPO_ID` in `.env`.

### Step 6 - Start the backend (terminal 1)

```powershell
python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 7860
```

The first start downloads the ~90 MB embedding model and loads the index (about 30 s). Wait for `Application startup complete.` Then check from a **new** PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:7860/health
Invoke-RestMethod http://127.0.0.1:7860/info
```

### Step 7 - Start the UI (terminal 2)

```powershell
cd C:\Projects\RAG
.\venv\Scripts\Activate.ps1
python -m streamlit run frontend/app.py --server.port 8501
```

Open **<http://localhost:8501>**. The sidebar shows API status, chunk count and LLM status.

### Shortcut - steps 3 to 7 in two commands

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup.ps1     # venv + dependencies + .env (one time)
.\scripts\start.ps1     # opens the API and the UI in two windows
```

### Test the API directly (optional)

```powershell
$body = @{ question = "What is retrieval augmented generation?"; k = 3 } | ConvertTo-Json
Invoke-WebRequest -Uri http://127.0.0.1:7860/query -Method Post -ContentType "application/json" -Body $body |
    Select-Object -ExpandProperty Content
```

`Invoke-WebRequest` waits for the whole stream, then prints all events at once. Interactive API docs: <http://127.0.0.1:7860/docs>.

### Command-line version

```powershell
python main.py "How do spiking neural networks save energy?" -k 4
```

---

## 4. Tests and evaluation

```powershell
# Unit tests - fast, no model or network needed
python -m unittest discover -s tests -v

# Retrieval benchmark - needs the embedding model; the backend need not be running
python evaluation/eval_retrieval.py
python evaluation/eval_retrieval.py -k 5 -n 100
```

| Check | What it measures |
|---|---|
| **Keyword hit@k** | For each question in `evaluation/questions.json`, does a top-k excerpt contain an expected keyword? |
| **Self-retrieval, exact chunk** | Search with a sentence from a chunk; is that exact chunk in the top k? Harsh: overlapping sibling chunks can legitimately outrank the original. |
| **Self-retrieval, same paper** | Is any chunk from the right paper in the top k? The fairer score. |
| **Paper-level** | Search by paper title; is that paper's chunk in the top k? |
| **Latency** | Median and p95 retrieval time. |

**Results on the current index** (4,619 chunks, 700-char, k = 4):

| Metric | 1000-char chunks | 700-char chunks |
|---|---|---|
| Exact-chunk hit | 74% | **86%** |
| Exact-chunk top-1 | 54% | **58%** |
| MRR | 0.629 | **0.684** |
| Same-paper hit | not measured | **98%** |
| Paper-level hit | 100% | **100%** |
| Median latency | 15 ms | 17 ms |

**Why 700 characters?** `all-MiniLM-L6-v2` only reads the first 256 tokens of its input. With 1000-character chunks, about 13% exceeded that, so the tail of those chunks was never embedded.

**Customising the questions.** `evaluation/questions.json` was written for an earlier corpus, so some questions may fail on yours. Replace them with questions about papers that are in your index; each entry is `{"question": "...", "any_of": ["keyword1", "keyword2"]}`.

**What this does not measure.** These tests show retrieval is healthy, not that generated answers are correct. For answer quality (faithfulness and relevance), the next step is an LLM-as-judge or a framework such as Ragas, with a hand-written set of question and reference-answer pairs.

---

## 5. Rebuild or update the index

You need this when you want a fresh set of papers, a different category, or a different chunk size.

**Full rebuild (downloads papers from arXiv, 10-30 minutes):**

```powershell
python data_ingestion_pipeline/main.py     # download, extract text, chunk
python load_and_vectorbase/load.py         # embed and write faiss_local_index/
```

Then restart the backend. This fetches the *newest* papers, so the corpus differs from your previous one. Settings (`CATEGORIES`, `PAPERS_PER_CATEGORY`, `CHUNK_SIZE`, `CHUNK_OVERLAP`) are in `data_ingestion_pipeline/config.py`.

**Change only the chunk size (no re-download, a few minutes):**

```powershell
python load_and_vectorbase/rechunk.py 700 100     # chunk size, overlap
python load_and_vectorbase/load.py
```

Always run `load.py` after changing the chunks, then restart the backend and re-run the evaluation.

---

## 6. Run with Docker (alternative)

```powershell
Copy-Item .env.example .env
notepad .env                      # optional: add your token
docker compose up --build         # the first build takes a while
```

- UI: <http://localhost:8501> - API docs: <http://localhost:7860/docs>
- Logs: `docker compose logs -f api`
- Stop: `docker compose down`

The index folder is mounted read-only, so after rebuilding the index just run `docker compose restart api`.

---

## 7. Configuration (`.env` or environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `HUGGINGFACEHUB_API_TOKEN` (or `HF_TOKEN`) | empty | Token for generated answers. Empty means retrieval-only. |
| `LLM_REPO_ID` | `meta-llama/Llama-3.1-8B-Instruct` | Chat model to use. |
| `LLM_PROVIDER` | `auto` | Inference provider (`auto` lets Hugging Face route the request). |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature. |
| `LLM_MAX_TOKENS` | `700` | Maximum answer length. |
| `DEFAULT_K` | `4` | Excerpts retrieved per question. |
| `FETCH_MULTIPLIER` | `3` | Candidates fetched before de-duplication (`k x this`). |
| `FAISS_INDEX_PATH` | `<project>/faiss_local_index` | Index location. |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `7860` | Server bind settings. |

For the UI: `API_URL` (default `http://localhost:7860`). Restart the backend after editing `.env`.

---

## 8. API reference

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status":"ok"}` - liveness |
| `GET /ready` | `{"status":"ready"}` or 503 if the index cannot be loaded |
| `GET /info` | Chunk count, whether the index has paper metadata, LLM model and status |
| `POST /query` | Body `{"question": "...", "k": 4}` returns a Server-Sent Events stream |

`/query` events - **every `data:` payload is JSON**:

```
event: sources   data: {"sources":[{"rank":1,"title":"...","score":0.83,"authors":"...","snippet":"..."}]}
event: notice    data: {"message":"..."}            (only when falling back to retrieval-only)
event: token     data: "first chunk of the answer"
event: token     data: " next chunk\n"
event: done      data: {"elapsed_ms":2841,"mode":"llm"}
event: error     data: {"message":"..."}            (unrecoverable failure)
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `cd` says the path does not exist | Use your real project folder, for example `cd C:\Projects\RAG`. |
| `... is not digitally signed` / `scripts is disabled` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then retry. Repeat in each new window. |
| `can't open file '...\load'` | Include the extension: `python load_and_vectorbase/load.py`. |
| `can't open file '...eval_retrieval.py'` | The file must be inside the `evaluation` folder, next to `questions.json`. |
| Sidebar shows **API offline** | Start the backend (step 6). Confirm with `Invoke-RestMethod http://127.0.0.1:7860/health`. |
| Port already in use | `Stop-Process -Id (Get-NetTCPConnection -LocalPort 7860).OwningProcess -Force` (use 8501 for the UI). Also close old API windows before restarting. |
| Notice "No Hugging Face token configured" | Put a real token in `.env` (step 5) and restart the backend. |
| Notice "LLM call failed ... 401 / 403" | The token lacks the *Inference Providers* permission, or the Llama licence has not been accepted. |
| Notice "LLM call failed ... **402 Payment Required**" | Your Hugging Face inference credits are used up. Wait for the monthly reset or add credits at <https://huggingface.co/settings/billing>. The app keeps working in retrieval-only mode. |
| Notice "LLM call failed ... 429" | Rate limited. Wait, or set another `LLM_REPO_ID`. |
| Sources show "Untitled excerpt" | The index has no paper metadata. Rebuild it (section 5). |
| `faiss` / `pymupdf` fail to install | Create the venv with Python 3.11 or 3.12: `py -3.12 -m venv venv`. |
| `/ready` returns 503 | Index missing or corrupt. Check `FAISS_INDEX_PATH` and rebuild (section 5). |
| First start is very slow | One-time download of the embedding model (~90 MB). |
| Changes to the index do not show up | Restart the backend; the index is loaded once at startup. |

---

## 10. Known limitations and ideas

- Each paper is truncated to its first ~20,000 characters (about 8-10 pages), so late sections and result tables are often missing.
- Retrieval is plain dense similarity. A **cross-encoder re-ranker** and/or **hybrid BM25 + vector** search would improve precision.
- PDF text extraction keeps headers, footnotes and equations as noise; a cleaner parser (for example `pymupdf4llm`) would help.
- The corpus is a small snapshot, so many questions are out of scope. The assistant is instructed to say it doesn't know in that case.
- Conversations are single-turn: earlier messages are displayed but not sent to the LLM.
- Answer quality is not yet evaluated (section 4).
- Generated answers need Hugging Face inference credits; a local LLM option (for example Ollama) would remove that dependency.

---

## 11. Engineering notes

Changes made while preparing this version, for reference:

| Area | Change |
|---|---|
| Index | Rebuilt with full paper metadata (title, authors, date, arXiv link) and duplicates removed; chunk size tuned from 1000 to 700 characters |
| LLM client | Switched to `huggingface_hub.InferenceClient` with a current `huggingface-hub` (Inference Providers) |
| Streaming | All SSE payloads are JSON-encoded, so newlines and Unicode survive; blocking calls run in worker threads |
| Retrieval | Single pass, de-duplicated, reused for both the sources panel and the prompt; numbered excerpts so the model can cite |
| Packaging | Added per-service `requirements.txt` files, fixed `.gitignore`, made the API token optional, Python 3.11 compatible |
| UI | Rewritten chat interface with persistent history and source cards |
| Ingestion | Cross-category de-duplication, absolute paths, no `jq` dependency |
| Quality | Unit tests, retrieval benchmark, PowerShell setup and start scripts |
