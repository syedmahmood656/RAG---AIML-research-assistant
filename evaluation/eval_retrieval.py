"""
Retrieval-quality evaluation (no LLM, no API token needed).

    python evaluation/eval_retrieval.py            # k=4, 60 self-retrieval probes
    python evaluation/eval_retrieval.py -k 5 -n 100
"""

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import rag  # noqa: E402


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [s.strip() for s in parts if 60 <= len(s.strip()) <= 250]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("-n", type=int, default=60, help="number of self-retrieval probes")
    args = ap.parse_args()

    db = rag._load_db()
    docs = list(db.docstore._dict.values())
    latencies = []

    # 1. keyword hit@k
    questions = json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    hits = 0
    print(f"== Keyword hit@{args.k} ==")
    for q in questions:
        t = time.perf_counter()
        res = rag.retrieve(q["question"], args.k)
        latencies.append(time.perf_counter() - t)
        blob = " ".join(norm(d.page_content) for d, _ in res)
        ok = any(kw in blob for kw in q["any_of"])
        hits += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {q['question']}")
    print(f"  -> {hits}/{len(questions)} = {hits / len(questions):.0%}\n")

    # 2. self-retrieval (exact chunk and same paper)
    rng = random.Random(7)
    probes = rng.sample(docs, min(args.n, len(docs)))
    ranks, paper_hits, paper_total = [], 0, 0
    for d in probes:
        sents = sentences(d.page_content)
        if not sents:
            continue
        t = time.perf_counter()
        res = rag.retrieve(rng.choice(sents), args.k)
        latencies.append(time.perf_counter() - t)
        target = norm(d.page_content)
        rank = next((i + 1 for i, (x, _) in enumerate(res) if norm(x.page_content) == target), None)
        ranks.append(rank)
        eid = (d.metadata or {}).get("entry_id")
        if eid:
            paper_total += 1
            paper_hits += any((x.metadata or {}).get("entry_id") == eid for x, _ in res)
    found = [r for r in ranks if r]
    n = max(len(ranks), 1)
    print(f"== Self-retrieval ({len(ranks)} probes) ==")
    print(f"  exact chunk  hit@{args.k}: {len(found) / n:.0%}   top-1: {sum(r == 1 for r in found) / n:.0%}   "
          f"MRR: {sum(1 / r for r in found) / n:.3f}")
    if paper_total:
        print(f"  same paper   hit@{args.k}: {paper_hits / paper_total:.0%}")
    print()

    # 3. paper-level (needs metadata)
    titled = {}
    for d in docs:
        m = d.metadata or {}
        if m.get("Title") and m.get("entry_id"):
            titled.setdefault(m["entry_id"], m["Title"])
    if titled:
        ok = 0
        for entry_id, title in titled.items():
            res = rag.retrieve(title, args.k)
            ok += any((x.metadata or {}).get("entry_id") == entry_id for x, _ in res)
        print(f"== Paper-level (title -> paper) over {len(titled)} papers ==")
        print(f"  hit@{args.k}: {ok / len(titled):.0%}\n")
    else:
        print("== Paper-level == skipped: this index has no Title metadata\n")

    print(f"Retrieval latency: median {statistics.median(latencies) * 1000:.0f} ms, "
          f"p95 {sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
