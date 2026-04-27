#!/usr/bin/env python3
"""Inline ST-only bench harness — bypasses longmemeval_eval.py deadlock.

Args:
    --split <path>           split_ids JSON
    --st-model <name>        SentenceTransformer model id
    --out <path>             results JSON path
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "/Users/macmini/Projects/adaptmem/benchmarks")

from longmemeval_eval import (
    load_dataset,
    make_per_question_corpus,
    recall_at_k,
    select_questions,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--st-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--dataset",
        default="/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json",
    )
    args = ap.parse_args()

    print(f"loading dataset", flush=True)
    data = load_dataset(args.dataset)
    split = json.load(open(args.split))
    test_qs = select_questions(data, split, "test", 100)
    print(f"  test questions: {len(test_qs)}", flush=True)

    from sentence_transformers import SentenceTransformer

    print(f"  loading {args.st_model}", flush=True)
    m = SentenceTransformer(args.st_model, device="cpu")
    print(f"  model loaded", flush=True)

    import numpy as np

    n = len(test_qs)
    r1 = r5 = r10 = 0.0
    skipped = 0
    per_type: dict[str, dict[str, float]] = {}
    t0 = time.time()
    for i, q in enumerate(test_qs):
        sids, docs = make_per_question_corpus(q)
        if not sids:
            skipped += 1
            continue
        embs = m.encode(
            docs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        qv = m.encode(
            [q["question"]], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        scores = embs @ qv
        order = np.argsort(-scores)
        ranked = [sids[j] for j in order]
        gt = set(q["answer_session_ids"])
        h1, h5, h10 = recall_at_k(ranked, gt, 1), recall_at_k(ranked, gt, 5), recall_at_k(ranked, gt, 10)
        r1 += h1
        r5 += h5
        r10 += h10
        qtype = q.get("question_type", "unknown")
        b = per_type.setdefault(qtype, {"n": 0.0, "r1": 0.0, "r5": 0.0, "r10": 0.0})
        b["n"] += 1
        b["r1"] += h1
        b["r5"] += h5
        b["r10"] += h10
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n - i - 1) / (i + 1)
            print(f"  q{i+1}/{n}  elapsed {elapsed:.0f}s  eta {eta:.0f}s", flush=True)

    per_type_summary = {
        t: {
            "n": int(b["n"]),
            "r1": round(b["r1"] / b["n"], 4) if b["n"] else 0.0,
            "r5": round(b["r5"] / b["n"], 4) if b["n"] else 0.0,
            "r10": round(b["r10"] / b["n"], 4) if b["n"] else 0.0,
        }
        for t, b in sorted(per_type.items())
    }
    result = {
        "n_test": n,
        "n_evaluated": n - skipped,
        "n_skipped_empty": skipped,
        "r1": round(r1 / n, 4) if n else 0.0,
        "r5": round(r5 / n, 4) if n else 0.0,
        "r10": round(r10 / n, 4) if n else 0.0,
        "per_question_type": per_type_summary,
        "wall_clock_s": round(time.time() - t0, 2),
        "model": args.st_model,
        "split": args.split,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"DONE r1={result['r1']}  r5={result['r5']}  r10={result['r10']}  wall={result['wall_clock_s']}s", flush=True)
    print(f"  → wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
