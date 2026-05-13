"""Eval an AdaptMem checkpoint on CodeSearchNet/python test split.

Loads a saved checkpoint, builds the test corpus (function bodies), then runs
each test query (docstring) and computes Recall@k + MRR. Writes per-query
ranks to a JSONL and a summary JSON next to the checkpoint.

Default test size is 5000 for fast iteration; pass --n -1 to use the full
~22k test set (recommended for the final report).

Run:
    .venv/bin/python benchmarks/codesearchnet_eval.py \
        --checkpoint checkpoints/ft-code/ft-code-1000 \
        --n 5000 \
        --out results/ft-code-1000.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def load_test(n: int) -> tuple[list[str], list[str], list[str]]:
    """Return (queries, ground_truth_ids, corpus_texts).

    Each test row contributes one query (docstring), one ground-truth body
    (added to corpus with a unique id). Dedup logic mirrors training-side
    cleanup: drop empty/short docstrings, dedup on body prefix.
    """
    ds = load_dataset("code_search_net", "python", split="test")
    queries: list[str] = []
    truths: list[str] = []
    corpus: list[str] = []
    corpus_ids: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(ds):
        if n > 0 and len(queries) >= n:
            break
        body = row.get("func_code_string") or ""
        doc = (row.get("func_documentation_string") or "").strip()
        if len(body) < 40 or len(doc) < 10:
            continue
        key = body[:200]
        if key in seen:
            continue
        seen.add(key)
        cid = f"ts{i}"
        queries.append(doc.splitlines()[0][:200])
        truths.append(cid)
        corpus.append(body)
        corpus_ids.append(cid)
    return queries, truths, corpus_ids


def evaluate(checkpoint: Path, n: int, out: Path) -> dict:
    print(f"[eval] loading model from {checkpoint}")
    # Bypass AdaptMem.load() to avoid the corpus.tsv newline-in-body issue
    # (function bodies can contain characters that break the tsv parser).
    # We only need the encoder for retrieval eval, not the training corpus.
    model = SentenceTransformer(str(Path(checkpoint) / "model"))
    print(f"[eval] loading test split (n={'all' if n < 0 else n})")
    t0 = time.time()
    queries, truths, corpus_ids = load_test(n)
    print(f"[eval] {len(queries)} queries, {len(corpus_ids)} corpus entries in {time.time()-t0:.1f}s")

    print("[eval] encoding test corpus with checkpoint model")
    from datasets import load_dataset as _ld
    ds = _ld("code_search_net", "python", split="test")
    body_by_id: dict[str, str] = {}
    for i, row in enumerate(ds):
        body_by_id[f"ts{i}"] = row.get("func_code_string") or ""
    test_corpus_texts = [body_by_id[cid] for cid in corpus_ids]

    t1 = time.time()
    corpus_emb = model.encode(test_corpus_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    query_emb = model.encode(queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    print(f"[eval] encoded in {time.time()-t1:.1f}s")

    sims = query_emb @ corpus_emb.T  # cosine since both normalised
    ranks = np.argsort(-sims, axis=1)  # descending

    cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    out.parent.mkdir(parents=True, exist_ok=True)
    r1 = r5 = r10 = 0
    mrr = 0.0
    with out.open("w") as f:
        for qi, truth in enumerate(truths):
            truth_idx = cid_to_idx[truth]
            ranked = ranks[qi]
            pos = int(np.where(ranked == truth_idx)[0][0])  # zero-indexed rank
            if pos == 0:
                r1 += 1
            if pos < 5:
                r5 += 1
            if pos < 10:
                r10 += 1
            mrr += 1.0 / (pos + 1)
            f.write(json.dumps({"qi": qi, "query": queries[qi][:200], "truth": truth, "rank": pos + 1, "top1": corpus_ids[int(ranked[0])]}) + "\n")

    n_q = len(queries)
    summary = {
        "checkpoint": str(checkpoint),
        "n_queries": n_q,
        "n_corpus": len(corpus_ids),
        "R@1": r1 / n_q,
        "R@5": r5 / n_q,
        "R@10": r10 / n_q,
        "MRR": mrr / n_q,
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[eval] R@1={summary['R@1']:.4f} R@5={summary['R@5']:.4f} R@10={summary['R@10']:.4f} MRR={summary['MRR']:.4f}")
    print(f"[eval] per-query → {out}")
    print(f"[eval] summary  → {summary_path}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--n", type=int, default=5000, help="-1 for full test split")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    evaluate(args.checkpoint, args.n, args.out)
