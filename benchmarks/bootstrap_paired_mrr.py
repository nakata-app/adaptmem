"""Paired bootstrap %95 CI for MRR delta between two encoders on jphein
chunk_x_encoder probe outputs.

For each strategy: take probe-level rr arrays from two JSON runs (same probe
order = paired), resample N=10000 with replacement, compute delta=MRR_b-MRR_a
each iteration, report mean delta + %95 CI.

Usage:
  python bootstrap_paired_mrr.py <run_a.json> <run_b.json> [--n-boot 10000]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean


def load_strategy_rr(path: Path) -> dict[str, list[float]]:
    data = json.loads(path.read_text())
    return {
        sname: [p["rr"] for p in s["probes"]]
        for sname, s in data["strategies"].items()
    }


def percentile(xs: list[float], q: float) -> float:
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs2) - 1)
    if f == c:
        return xs2[f]
    return xs2[f] + (k - f) * (xs2[c] - xs2[f])


def bootstrap_delta(rr_a: list[float], rr_b: list[float], n_boot: int, seed: int = 42):
    assert len(rr_a) == len(rr_b), "paired arrays must be same length"
    n = len(rr_a)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        m_a = mean(rr_a[i] for i in idxs)
        m_b = mean(rr_b[i] for i in idxs)
        deltas.append(m_b - m_a)
    return {
        "mrr_a": mean(rr_a),
        "mrr_b": mean(rr_b),
        "delta_point": mean(rr_b) - mean(rr_a),
        "delta_boot_mean": mean(deltas),
        "ci_low": percentile(deltas, 0.025),
        "ci_high": percentile(deltas, 0.975),
        "p_positive": sum(1 for d in deltas if d > 0) / n_boot,
        "p_negative": sum(1 for d in deltas if d < 0) / n_boot,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path, help="baseline JSON (e.g. default)")
    ap.add_argument("run_b", type=Path, help="treatment JSON (e.g. FT-Code-5000)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    a = load_strategy_rr(args.run_a)
    b = load_strategy_rr(args.run_b)
    common = sorted(set(a) & set(b))

    header = f"{'strategy':<35} {'mrr_'+args.label_a:>10} {'mrr_'+args.label_b:>10} {'delta':>9} {'95% CI':>22} {'P(>0)':>7}"
    print(header)
    print("-" * len(header))
    for sname in common:
        r = bootstrap_delta(a[sname], b[sname], args.n_boot)
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        sig = ""
        if r["ci_low"] > 0:
            sig = " ✓ sig+"
        elif r["ci_high"] < 0:
            sig = " ✗ sig-"
        else:
            sig = " ~ ns"
        print(
            f"{sname:<35} {r['mrr_a']:>10.4f} {r['mrr_b']:>10.4f} "
            f"{r['delta_point']:>+.4f} {ci:>22} {r['p_positive']:>7.3f}{sig}"
        )


if __name__ == "__main__":
    main()
