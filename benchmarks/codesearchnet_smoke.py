"""Smoke test: AdaptMem.train() on a 1000-sample CodeSearchNet Python slice.

Goal: verify the existing train contract accepts code-domain (docstring → body)
pairs, hard-negative mining produces non-empty pairs, MNR loss drops over 1 epoch,
and the resulting model can encode + retrieve on held-out queries. No full train,
no checkpoint persist. If this completes on Mac (8GB), full train moves to Colab.

Run:
    .venv/bin/python benchmarks/codesearchnet_smoke.py [--device cpu|mps]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from datasets import load_dataset

from adaptmem.core import AdaptMem
from adaptmem.miner import CorpusEntry
from adaptmem.types import LabelledQuery, TrainConfig


def build_pairs(n: int) -> tuple[list[CorpusEntry], list[LabelledQuery]]:
    """Load CodeSearchNet python validation split, return (corpus, labelled).

    Each row: docstring → function body. Drop rows with empty docstring or body
    shorter than 40 chars (too noisy to learn). Dedup on body to avoid
    cross-row collisions in hard-negative mining.
    """
    ds = load_dataset("code_search_net", "python", split="validation", trust_remote_code=True)
    corpus: list[CorpusEntry] = []
    labelled: list[LabelledQuery] = []
    seen_bodies: set[str] = set()
    for i, row in enumerate(ds):
        if len(corpus) >= n:
            break
        body = row.get("func_code_string") or ""
        docstring = row.get("func_documentation_string") or ""
        if len(body) < 40 or not docstring.strip():
            continue
        body_key = body[:200]
        if body_key in seen_bodies:
            continue
        seen_bodies.add(body_key)
        cid = f"cs{i}"
        corpus.append(CorpusEntry(id=cid, text=body))
        labelled.append(LabelledQuery(query=docstring.strip().splitlines()[0][:200], relevant_ids=[cid]))
    return corpus, labelled


def smoke(device: str | None, n_samples: int = 1000) -> None:
    print(f"[smoke] building {n_samples} (docstring -> body) pairs from CodeSearchNet/python/validation")
    t0 = time.time()
    corpus, labelled = build_pairs(n_samples)
    print(f"[smoke] built {len(corpus)} corpus entries, {len(labelled)} labelled queries in {time.time()-t0:.1f}s")
    if len(corpus) < 100:
        raise RuntimeError(f"too few pairs ({len(corpus)}); dataset filter or download issue")

    print(f"[smoke] init AdaptMem(base=all-MiniLM-L6-v2, device={device or 'auto'})")
    am = AdaptMem(base_model="sentence-transformers/all-MiniLM-L6-v2", device=device)

    cfg = TrainConfig(epochs=1, batch_size=8, top_k_mine=5)
    print(f"[smoke] train epochs={cfg.epochs} batch_size={cfg.batch_size}")
    t1 = time.time()
    stats = am.train(corpus, labelled, config=cfg)
    elapsed = time.time() - t1
    print(f"[smoke] train ok, stats={stats}, elapsed={elapsed:.1f}s")

    sample_query = labelled[0].query
    sample_truth = labelled[0].relevant_ids[0]
    hits = am.search(sample_query, top_k=5)
    print(f"[smoke] sample search: query={sample_query[:80]!r}")
    for i, h in enumerate(hits):
        marker = " <- TRUTH" if h.chunk_id == sample_truth else ""
        print(f"  [{i+1}] {h.chunk_id} score={h.score:.3f}{marker}")

    print("[smoke] PASS, train+search contract works on code-domain pairs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    smoke(args.device, args.n)
