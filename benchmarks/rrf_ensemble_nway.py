"""N-way RRF surrogate ensemble across jphein chunk_x_encoder runs.

Generalizes rrf_ensemble.py to accept any number of input JSON files.
Fused rank = min(rank_i) across all inputs (lower bound on true RRF gain,
since JSONs only store rank-of-expected, not full ranked lists).

Usage:
  python rrf_ensemble_nway.py --label default <default.json> \
                              --label ft300   <ft300.json>   \
                              --label ft1000  <ft1000.json>  \
                              --label ft5000  <ft5000.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse(path: Path):
    data = json.loads(path.read_text())
    n_results = data.get("n_results", 10)
    out = {}
    for sname, s in data["strategies"].items():
        probes = []
        for p in s["probes"]:
            rk = p.get("rank")
            probes.append(rk if rk else None)
        out[sname] = {"probes": probes, "mrr": s["mrr"], "r10": s["recall_at_10_pct"]}
    return out, n_results


def fused(per_run_probes: list[list], miss_rank: int):
    n = len(per_run_probes[0])
    rr = []
    r5 = r10 = 0
    for i in range(n):
        ranks = [run[i] if run[i] else miss_rank for run in per_run_probes]
        f_rank = min(ranks)
        rr.append(1.0 / f_rank if f_rank <= miss_rank else 0.0)
        if f_rank <= 5:
            r5 += 1
        if f_rank <= 10:
            r10 += 1
    return mean(rr), 100 * r5 / n, 100 * r10 / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--path", action="append", type=Path, required=True)
    ap.add_argument("--combo", action="append", default=[],
                    help="comma-separated label subset to fuse (repeatable). "
                         "If empty, fuses ALL provided runs.")
    args = ap.parse_args()
    assert len(args.label) == len(args.path), "--label and --path counts must match"

    runs = {lbl: parse(p) for lbl, p in zip(args.label, args.path)}
    n_results = max(v[1] for v in runs.values())
    miss = n_results + 1

    strategies = sorted(set.intersection(*[set(v[0]) for v in runs.values()]))

    combos = args.combo or [",".join(args.label)]

    header_labels = list(args.label)
    print(f"{'strategy':<33} " + " ".join(f"{l:>9}" for l in header_labels), end="  ")
    print("  ".join(f"fused[{c}]".ljust(28) for c in combos))
    sep = "-" * (33 + (10 * len(header_labels)) + 2 + sum(len(c) + 32 for c in combos))
    print(sep)

    for sname in strategies:
        row = [sname.ljust(33)]
        per_run_mrr = {lbl: runs[lbl][0][sname]["mrr"] for lbl in args.label}
        row.append(" ".join(f"{per_run_mrr[lbl]:>9.4f}" for lbl in args.label))

        fused_parts = []
        for combo in combos:
            sel = [c.strip() for c in combo.split(",")]
            probe_lists = [runs[s][0][sname]["probes"] for s in sel]
            mrr_f, r5_f, r10_f = fused(probe_lists, miss)
            best_solo = max(per_run_mrr[s] for s in sel)
            delta = mrr_f - best_solo
            sign = "+" if delta >= 0 else ""
            fused_parts.append(f"mrr={mrr_f:.4f} r10={r10_f:.1f}% Δvs_best={sign}{delta:+.4f}")
        row.append("  " + "  ".join(p.ljust(28) for p in fused_parts))
        print("".join(row))


if __name__ == "__main__":
    main()
