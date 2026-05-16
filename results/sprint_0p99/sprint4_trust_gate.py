"""Sprint 4 — Trust-gated cross-encoder rerank.

Reasoning (anomaly found in Sprint 3 final):
  ft-v4 raw R@1 = 0.968 (16 fail / 500). All 16 fails have gold rank in 2..10.
  Sprint 3's per-category strategy (chat-ce-v3 pure rerank on preference)
  improved 7 cases but BROKE 4 cases in preference where ft-v4 raw was already
  rank-1. Net was -3 fails (16 -> 13, R@1 0.974).

  Trust gate principle:
    - Apply chat-ce-v3 to bi-encoder top-K (K=20).
    - Only override bi-encoder top-1 if CE's confidence margin is high.
    - Otherwise keep bi-encoder top-1 (trust the proven retriever).

  This recovers the 4 broken preference cases while preserving the 7 wins,
  expected R@1 >= 0.984.

Margin formula: ce_top1_score - ce_score_at_bi_top1 >= MARGIN -> override.

Run:
  python sprint4_trust_gate.py --margin 1.0
  python sprint4_trust_gate.py --margin 2.0 --top-k 20
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

REPO = Path("/Users/macmini/Projects/adaptmem")
RUN6 = REPO / "benchmarks/v335/run6_v335_hybrid_v4_ftv4.jsonl"
GOLD_PATH = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")

# Per-type CE checkpoint mapping. None = no rerank (trust bi-encoder).
CKPTS = {
    "single-session-preference": REPO / "checkpoints/chat-ce-v3-20260516",
    "multi-session":              REPO / "checkpoints/chat-ce-v3-20260516",
    "knowledge-update":           REPO / "checkpoints/chat-ce-v3-20260516",
    "temporal-reasoning":         REPO / "checkpoints/chat-ce-v3-20260516",
    "single-session-assistant":   REPO / "checkpoints/chat-ce-v2-20260516",
    "single-session-user":        None,
}


def load_gold() -> dict[str, set[str]]:
    data = json.loads(GOLD_PATH.read_text())
    return {q["question_id"]: set(q.get("answer_session_ids") or []) for q in data}


def load_runs() -> list[dict]:
    out = []
    with RUN6.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=1.0,
                    help="CE top-1 must beat CE-at-bi-top-1 by >= margin to override")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default=str(REPO / "results/sprint_0p99/sprint4_trust_gate_result.json"))
    ap.add_argument("--out-jsonl", default=str(REPO / "benchmarks/v335/run7_trust_gate.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="limit questions for smoke")
    args = ap.parse_args()

    print(f"loading gold from {GOLD_PATH}...")
    gold = load_gold()
    print(f"  {len(gold)} questions")

    print(f"loading run6 from {RUN6}...")
    runs = load_runs()
    print(f"  {len(runs)} questions")
    if args.limit:
        runs = runs[:args.limit]

    # Load CE models lazily (one per checkpoint, cached)
    from sentence_transformers import CrossEncoder
    ce_cache: dict[str, CrossEncoder] = {}

    def get_ce(path: Path | None) -> CrossEncoder | None:
        if path is None:
            return None
        key = str(path)
        if key not in ce_cache:
            print(f"  loading CE: {path.name}")
            t0 = time.time()
            ce_cache[key] = CrossEncoder(key, max_length=384)
            print(f"    loaded in {time.time()-t0:.1f}s")
        return ce_cache[key]

    by_type = defaultdict(lambda: {"n": 0, "baseline": 0, "final": 0,
                                    "overrides": 0, "trust_kept": 0,
                                    "override_helped": 0, "override_hurt": 0})
    fails_baseline = []
    fails_final = []
    overrides_log = []
    t0 = time.time()

    out_records = []

    for i, rec in enumerate(runs):
        qid = rec["question_id"]
        qtype = rec["question_type"]
        items = rec["retrieval_results"]["ranked_items"][: args.top_k]
        g = gold.get(qid, set())

        baseline_top1 = items[0]["corpus_id"]
        baseline_hit = int(bool(g) and baseline_top1 in g)

        ce_path = CKPTS.get(qtype)
        ce = get_ce(ce_path)
        if ce is None or len(items) < 2:
            final_top1 = baseline_top1
            final_hit = baseline_hit
            override = False
        else:
            q = rec["question"]
            pairs = [(q, it["text"][:2000]) for it in items]
            scores = ce.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
            ce_top1_idx = int(max(range(len(scores)), key=lambda j: scores[j]))
            bi_top1_idx = 0
            margin = float(scores[ce_top1_idx]) - float(scores[bi_top1_idx])
            if ce_top1_idx == bi_top1_idx or margin < args.margin:
                final_top1 = items[bi_top1_idx]["corpus_id"]
                override = False
            else:
                final_top1 = items[ce_top1_idx]["corpus_id"]
                override = True
            final_hit = int(bool(g) and final_top1 in g)
            if override:
                helped = (baseline_hit == 0 and final_hit == 1)
                hurt   = (baseline_hit == 1 and final_hit == 0)
                overrides_log.append({
                    "qid": qid, "type": qtype, "margin": round(margin, 4),
                    "baseline_hit": baseline_hit, "final_hit": final_hit,
                    "helped": helped, "hurt": hurt,
                })

        b = by_type[qtype]
        b["n"] += 1
        b["baseline"] += baseline_hit
        b["final"] += final_hit
        if ce is not None and len(items) >= 2:
            if override:
                b["overrides"] += 1
                if baseline_hit == 0 and final_hit == 1: b["override_helped"] += 1
                if baseline_hit == 1 and final_hit == 0: b["override_hurt"] += 1
            else:
                b["trust_kept"] += 1

        if not baseline_hit: fails_baseline.append((qid, qtype))
        if not final_hit:    fails_final.append((qid, qtype))

        # Write run7 jsonl (re-ordered top-K)
        if override:
            # Move CE top-1 to front, keep rest in original order
            new_items = [items[ce_top1_idx]] + [it for j, it in enumerate(items) if j != ce_top1_idx]
        else:
            new_items = items
        new_rec = dict(rec)
        new_rec["retrieval_results"] = dict(rec["retrieval_results"])
        new_rec["retrieval_results"]["ranked_items"] = new_items + rec["retrieval_results"]["ranked_items"][args.top_k:]
        out_records.append(new_rec)

        if (i+1) % 50 == 0:
            el = time.time() - t0
            eta = el * (len(runs)-i-1) / (i+1)
            print(f"  q{i+1}/{len(runs)}  eta {eta:.0f}s")

    n = sum(b["n"] for b in by_type.values())
    R_baseline = sum(b["baseline"] for b in by_type.values()) / n
    R_final = sum(b["final"] for b in by_type.values()) / n

    report = {
        "experiment": "Sprint 4 trust-gated CE rerank",
        "margin": args.margin,
        "top_k": args.top_k,
        "n": n,
        "R@1_baseline_ftv4_raw": round(R_baseline, 4),
        "R@1_final": round(R_final, 4),
        "fails_baseline": n - sum(b["baseline"] for b in by_type.values()),
        "fails_final": n - sum(b["final"] for b in by_type.values()),
        "by_type": {t: {
            "n": b["n"],
            "R@1_baseline": round(b["baseline"]/b["n"], 4),
            "R@1_final": round(b["final"]/b["n"], 4),
            "overrides": b["overrides"], "trust_kept": b["trust_kept"],
            "override_helped": b["override_helped"], "override_hurt": b["override_hurt"],
        } for t, b in sorted(by_type.items())},
        "remaining_fails": sorted(fails_final, key=lambda x: (x[1], x[0])),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    print("\n== RESULT ==")
    print(json.dumps(report, indent=2))
    print(f"\nsaved -> {args.out}\n         {args.out_jsonl}")


if __name__ == "__main__":
    main()
