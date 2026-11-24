"""
Command-line RAG over the FAISS index (same pipeline the API uses).

    python main.py "What are the computational challenges of Seiberg dualities?" -k 4

The Hugging Face token is read from .env / the environment - never hard-code it here.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
import rag  # noqa: E402


async def run(question: str, k: int) -> None:
    results = rag.retrieve(question, k)
    print("\nSources:")
    for s in rag.format_sources(results):
        print(f"  [{s['rank']}] {s['title']}  ({s['score']:.0%} match)")
    print("\nAnswer:\n")
    async for kind, text in rag.stream_answer(question, results):
        if kind == "notice":
            print(f"(note: {text})\n")
        else:
            print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", default="What are the computational challenges of Seiberg Dualities?")
    ap.add_argument("-k", type=int, default=4, help="number of excerpts to retrieve")
    args = ap.parse_args()
    asyncio.run(run(args.question, args.k))
