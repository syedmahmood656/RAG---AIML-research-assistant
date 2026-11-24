"""
Build the FAISS index from data/chunked_papers.json.

Run from anywhere:  python load_and_vectorbase/load.py

Differences from the original script
- Plain json instead of JSONLoader: JSONLoader needs the `jq` package (missing from
  requirements, and hard to install on Windows) and - without a metadata_func - it DROPPED the
  paper metadata, so the prebuilt index had no titles/authors/links.
- Exact-duplicate chunks are removed before embedding.
- The final message reports the real chunk count instead of a hard-coded number.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from rag import get_embeddings  # noqa: E402  (same embedder the API uses)

DATA_PATH = ROOT / "data" / "chunked_papers.json"
OUT_PATH = ROOT / "faiss_local_index"
BATCH_SIZE = 500

if not DATA_PATH.exists():
    sys.exit(f"{DATA_PATH} not found. Run first: python data_ingestion_pipeline/main.py")

records = json.loads(DATA_PATH.read_text(encoding="utf-8"))

docs, seen = [], set()
for r in records:
    text = (r.get("page_content") or "").strip()
    key = hashlib.md5(re.sub(r"\s+", " ", text).lower().encode("utf-8")).hexdigest()
    if not text or key in seen:
        continue
    seen.add(key)
    docs.append(Document(page_content=text, metadata=r.get("metadata", {})))

print(f"{len(records)} chunks read, {len(docs)} unique after de-duplication.")
embeddings = get_embeddings()

db = None
for start in range(0, len(docs), BATCH_SIZE):
    batch = docs[start:start + BATCH_SIZE]
    print(f"Embedding chunks {start + 1}-{start + len(batch)} of {len(docs)} ...")
    if db is None:
        db = FAISS.from_documents(batch, embeddings)
    else:
        db.add_documents(batch)

db.save_local(str(OUT_PATH))
papers = len({d.metadata.get("entry_id") for d in docs})
print(f"Done: {len(docs)} chunks from {papers} papers saved to {OUT_PATH}")
