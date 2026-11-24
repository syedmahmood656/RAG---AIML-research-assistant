# ArXiv RAG - research-paper Q&A

Retrieval-Augmented Generation over ~150 recent arXiv cs.AI / cs.LG papers:
**FAISS** retrieval -> **Llama 3.1** (Hugging Face Inference Providers) -> **FastAPI** SSE stream -> **Streamlit** chat UI.

Everything you need - architecture, step-by-step PowerShell install, Docker, rebuilding the index,
evaluation, troubleshooting - is in **[guid.md](guid.md)**.

Quick start (Windows PowerShell):

```powershell
.\scripts\setup.ps1     # venv + dependencies + .env   (one time)
.\scripts\start.ps1     # API on :7860, UI on :8501
```

Tests: `python -m unittest discover -s tests -v`
