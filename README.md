# 📚 ArXiv RAG - Research-Paper Q&A

Ask questions about recent AI research and get answers grounded in the papers themselves, with the supporting excerpts shown next to every answer.

The app indexes **138 recent arXiv papers** (cs.AI + cs.LG, full text, **4,619 chunks**). It retrieves the most relevant excerpts with FAISS, and an LLM writes an answer from them with `[1]`, `[2]` citations. If no LLM is available, it still works as a semantic search over the papers.

## Features

- **Grounded answers** - the LLM sees only retrieved excerpts and cites them
- **Source cards** - title, authors, date, match %, excerpt and arXiv link for every source
- **Live streaming** - answers appear token by token (Server-Sent Events)
- **Graceful fallback** - without a token or credits, the app shows the retrieved excerpts instead of failing
- **Evaluation included** - a retrieval benchmark and unit tests ship with the project
- **Runs anywhere** - PowerShell scripts for Windows, or `docker compose up`

## Tech stack

Python · FastAPI · Streamlit · LangChain · FAISS · Sentence-Transformers (all-MiniLM-L6-v2) · Hugging Face Inference Providers (Llama 3.1 8B) · PyMuPDF · arXiv API · Docker

## Architecture

```
arXiv PDFs ─► PyMuPDF text ─► 700-char chunks ─► MiniLM embeddings ─► FAISS index
                                                                          │
Streamlit UI ──question──► FastAPI ──► embed query ──► top-k chunks (de-duplicated)
     ▲                        │                               │
     └── SSE: sources, ◄──────┴── LLM prompt (numbered excerpts + question) ──► Llama 3.1
         tokens, done
```

## Retrieval results

Measured with `evaluation/eval_retrieval.py` on the current index (k = 4):

| Metric | Result |
|---|---|
| Paper-level hit (search by title → right paper) | **100%** (138 / 138) |
| Same-paper hit (probe sentence → chunk from the right paper) | **98%** |
| Exact-chunk hit | **86%** (74% with 1000-char chunks) |
| Median retrieval latency | **17 ms** (p95 29 ms) |

These are self-retrieval probes (sentences taken from the indexed text). They show the index is healthy, not that answers are accurate. Answer-quality evaluation is on the roadmap.

## Quick start (Windows PowerShell)

```powershell
cd C:\Projects\RAG                     # your project folder
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup.ps1                    # venv + dependencies + .env (one time)
notepad .env                           # optional: paste your Hugging Face token
.\scripts\start.ps1                    # API :7860 and UI :8501
```

Open <http://localhost:8501>. Full instructions, Docker, rebuilding the index and troubleshooting are in **[guid.md](guid.md)**.

## Project structure

```
backend/                 FastAPI app, RAG core, config, SSE encoder
frontend/                Streamlit chat UI
data_ingestion_pipeline/ arXiv download, text extraction, chunking
load_and_vectorbase/     index builder, re-chunking, retrieval smoke test
evaluation/              retrieval benchmark + question set
tests/                   unit tests (no model or network needed)
scripts/                 setup.ps1, start.ps1
faiss_local_index/       the vector index
```

## Tests and evaluation

```powershell
python -m unittest discover -s tests -v     # unit tests
python evaluation/eval_retrieval.py         # retrieval benchmark
```

## Roadmap

- Answer-quality evaluation (LLM-as-judge / Ragas)
- Cross-encoder re-ranker and hybrid BM25 + vector search
- Multi-turn conversation memory
- Local LLM option (Ollama) so no API credits are needed

## License

Add your license here (e.g. MIT).
