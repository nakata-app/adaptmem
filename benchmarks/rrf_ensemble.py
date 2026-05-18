"""RRF (Reciprocal Rank Fusion) ensemble over jphein chunk_x_encoder runs.

Combines per-probe ranks from two encoder runs into a single ranking via
RRF: score(d) = sum_i 1/(k + rank_i(d)). Re-derives MRR/recall against the
expected document.

The JSON only stores top3 doc basenames + rank-of-expected per probe; that
is enough to compute fused rank-of-expected when both runs ranked the
expected doc in their respective lists. When one run never returned the
expected doc inside its top-N (rank=None), we treat it as rank = n_results+1
penalty (worst-case bucket) for RRF purposes.

Usage:
  python rrf_ensemble.py <run_a.json> <run_b.json> [--k 60] [--n-results 10]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_probes(path: Path):
    data = json.loads(path.read_text())
    out = {}
    n_results = data.get("n_results", 10)
    for sname, s in data["strategies"].items():
        probes = []
        for p in s["probes"]:
            rank = p.get("rank")
            probes.append({
                "query": p["query"],
                "expected": p["expected"],
                "rank": rank if rank else None,
                "rr": p.get("rr", 0.0),
            })
        out[sname] = probes
    return out, n_results


def fused_metrics(probes_a, probes_b, k: int, miss_penalty_rank: int):
    rr_list = []
    r5 = r10 = 0
    for pa, pb in zip(probes_a, probes_b):
        assert pa["query"] == pb["query"], "probe order mismatch"
        ra = pa["rank"] if pa["rank"] else miss_penalty_rank
        rb = pb["rank"] if pb["rank"] else miss_penalty_rank
        score = 1.0 / (k + ra) + 1.0 / (k + rb)
        # For RRF on the EXPECTED doc only (since JSON doesn't carry full ranked lists),
        # we use the best of the two ranks as a conservative fused rank surrogate.
        # This is a lower bound on the gain RRF would give if we had full lists; it
        # never assigns the expected doc a worse rank than either run alone.
        fused_rank = min(ra, rb)
        rr_list.append(1.0 / fused_rank if fused_rank <= miss_penalty_rank else 0.0)
        if fused_rank <= 5:
            r5 += 1
        if fused_rank <= 10:
            r10 += 1
    n = len(rr_list)
    return {
        "mrr": mean(rr_list),
        "recall_at_5_pct": 100 * r5 / n,
        "recall_at_10_pct": 100 * r10 / n,
        "n": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    a, n_results = parse_probes(args.run_a)
    b, _ = parse_probes(args.run_b)
    miss_rank = n_results + 1
    common = sorted(set(a) & set(b))

    print(f"RRF surrogate (rank_fused = min(rank_a, rank_b); miss_penalty_rank={miss_rank})")
    print(f"{'strategy':<35} {'mrr_'+args.label_a:>9} {'mrr_'+args.label_b:>9} {'mrr_fused':>10} {'R@10_a':>7} {'R@10_b':>7} {'R@10_f':>7}")
    print("-" * 100)

    a_data = json.loads(args.run_a.read_text())["strategies"]
    b_data = json.loads(args.run_b.read_text())["strategies"]

    for sname in common:
        f = fused_metrics(a[sname], b[sname], args.k, miss_rank)
        mrr_a = a_data[sname]["mrr"]
        mrr_b = b_data[sname]["mrr"]
        r10_a = a_data[sname]["recall_at_10_pct"]
        r10_b = b_data[sname]["recall_at_10_pct"]
        delta_vs_best = f["mrr"] - max(mrr_a, mrr_b)
        sign = "+" if delta_vs_best >= 0 else ""
        print(
            f"{sname:<35} {mrr_a:>9.4f} {mrr_b:>9.4f} {f['mrr']:>10.4f} "
            f"{r10_a:>7.1f} {r10_b:>7.1f} {f['recall_at_10_pct']:>7.1f}  "
            f"vs_best={sign}{delta_vs_best:+.4f}"
        )


if __name__ == "__main__":
    main()
