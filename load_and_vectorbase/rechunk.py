"""
Re-split the already-downloaded papers (data/raw_papers.json) with a different chunk size,
WITHOUT re-downloading anything from arXiv.

    python load_and_vectorbase/rechunk.py              # 700 / 100
    python load_and_vectorbase/rechunk.py 600 80       # size / overlap

Then rebuild the index:  python load_and_vectorbase/load.py
Why: all-MiniLM-L6-v2 only reads the first 256 tokens; ~13% of 1000-char chunks exceed that,
so the tail of those chunks is never embedded.
"""

import json
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
size = int(sys.argv[1]) if len(sys.argv) > 1 else 700
overlap = int(sys.argv[2]) if len(sys.argv) > 2 else 100

raw = json.loads((ROOT / "data" / "raw_papers.json").read_text(encoding="utf-8"))
docs = [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in raw]
chunks = RecursiveCharacterTextSplitter(
    chunk_size=size, chunk_overlap=overlap, add_start_index=True
).split_documents(docs)

out = ROOT / "data" / "chunked_papers.json"
out.write_text(
    json.dumps([{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
               indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)
print(f"{len(docs)} papers -> {len(chunks)} chunks (size={size}, overlap={overlap}) -> {out}")
