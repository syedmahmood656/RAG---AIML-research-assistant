"""Quick retrieval smoke test:  python load_and_vectorbase/test.py "your question" """

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import rag  # noqa: E402

query = " ".join(sys.argv[1:]) or "What are the computational challenges of Seiberg Dualities?"
print(f"Searching for: {query!r}\n")
for i, (doc, score) in enumerate(rag.retrieve(query, k=3), 1):
    meta = doc.metadata or {}
    print(f"--- Result {i}  (match {rag.l2_to_similarity(score):.0%}) ---")
    print("Title:", meta.get("Title", "(no metadata in this index)"))
    print(doc.page_content[:600].replace("\n", " "), "...\n")
