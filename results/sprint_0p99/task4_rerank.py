"""Task 4 — Cross-encoder rerank on preference + temporal subset.

Pipeline: hybrid_v4 top-K (K=20) -> codecrossenc-v2 -> top-1.
Compares baseline R@1 (hybrid_v4 + FT-300) vs reranked R@1 on subset only.

Gold lookup uses LongMemEval-S `answer_session_ids`. R@1 = top-1 corpus_id in
that set (or any haystack_session_id under it -- we treat answer_session_ids
as ground truth, matching the bench's recall_any@1 semantics).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path("/Users/macmini/Projects/adaptmem")
RUN5 = REPO / "benchmarks/v335/run5_v335_hybrid_v4_ft300.jsonl"
GOLD = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")
CKPT_CODE = REPO / "checkpoints/codecrossenc-v2-20260516/codecrossenc_v2"
CKPT_MSMARCO = "cross-encoder/ms-marco-MiniLM-L-12-v2"
SUBSET_TYPES = {"single-session-preference", "temporal-reasoning"}


def load_gold(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text())
    gold = {}
    for r in data:
        gold[r["question_id"]] = set(r.get("answer_session_ids") or [])
    return gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--ckpt", default="msmarco", choices=["msmarco", "code"])
    ap.add_argument("--fusion", default="pure", choices=["pure", "rrf", "margin"])
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--margin-thresh", type=float, default=0.05, help="for margin mode: only rerank when bi top1-top2 normalized gap < thresh")
    ap.add_argument("--out", default=str(REPO / "results/sprint_0p99/task4_rerank_result.json"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    gold = load_gold(GOLD)
    print(f"gold loaded: {len(gold)} questions")

    subset_records = []
    with RUN5.open() as f:
        for line in f:
            r = json.loads(line)
            if r["question_type"] in SUBSET_TYPES:
                subset_records.append(r)
    print(f"subset records: {len(subset_records)} ({SUBSET_TYPES})")

    from sentence_transformers import CrossEncoder
    ckpt = str(CKPT_CODE) if args.ckpt == "code" else CKPT_MSMARCO
    print(f"loading cross-encoder: {ckpt}")
    t0 = time.time()
    model = CrossEncoder(ckpt, max_length=384)
    print(f"  loaded in {time.time()-t0:.1f}s")

    by_type_baseline = {}
    by_type_rerank = {}
    moved_top1 = []
    not_in_topk = []

    for rec in subset_records:
        qid = rec["question_id"]
        qtype = rec["question_type"]
        q = rec["question"]
        items = rec["retrieval_results"]["ranked_items"][: args.top_k]
        g = gold.get(qid, set())

        baseline_top1 = items[0]["corpus_id"] if items else None
        baseline_hit = int(bool(g) and baseline_top1 in g)
        by_type_baseline.setdefault(qtype, []).append(baseline_hit)

        pairs = [(q, it["text"][:2000]) for it in items]
        scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
        ce_rank = {i: r for r, i in enumerate(sorted(range(len(items)), key=lambda i: -float(scores[i])))}
        bi_rank = {i: i for i in range(len(items))}

        if args.fusion == "pure":
            order = sorted(range(len(items)), key=lambda i: -float(scores[i]))
        elif args.fusion == "rrf":
            k = args.rrf_k
            order = sorted(range(len(items)), key=lambda i: -(1.0/(k+bi_rank[i]) + 1.0/(k+ce_rank[i])))
        else:  # margin: only apply CE when bi is uncertain
            order = list(range(len(items)))
        rerank_top1 = items[order[0]]["corpus_id"]
        rerank_hit = int(bool(g) and rerank_top1 in g)
        by_type_rerank.setdefault(qtype, []).append(rerank_hit)

        if baseline_hit == 0 and rerank_hit == 1:
            moved_top1.append({"qid": qid, "type": qtype, "q": q[:120]})
        if g and not any(it["corpus_id"] in g for it in items):
            not_in_topk.append({"qid": qid, "type": qtype})

    report = {
        "experiment": "Task 4 cross-encoder rerank on subset",
        "subset_types": sorted(SUBSET_TYPES),
        "top_k": args.top_k,
        "fusion": args.fusion,
        "rrf_k": args.rrf_k if args.fusion == "rrf" else None,
        "n_queries": len(subset_records),
        "by_type": {
            t: {
                "n": len(by_type_baseline[t]),
                "baseline_R@1": round(sum(by_type_baseline[t]) / len(by_type_baseline[t]), 4),
                "rerank_R@1": round(sum(by_type_rerank[t]) / len(by_type_rerank[t]), 4),
                "baseline_fails": len(by_type_baseline[t]) - sum(by_type_baseline[t]),
                "rerank_fails": len(by_type_rerank[t]) - sum(by_type_rerank[t]),
            }
            for t in sorted(by_type_baseline)
        },
        "subset_total": {
            "baseline_R@1": round(
                sum(sum(v) for v in by_type_baseline.values()) / sum(len(v) for v in by_type_baseline.values()), 4
            ),
            "rerank_R@1": round(
                sum(sum(v) for v in by_type_rerank.values()) / sum(len(v) for v in by_type_rerank.values()), 4
            ),
        },
        "fails_unrecoverable_topk": len(not_in_topk),
        "fails_unrecoverable_qids": not_in_topk,
        "newly_correct_top1": moved_top1,
        "checkpoint": ckpt,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n== RESULT ==")
    print(json.dumps(report, indent=2)[:2000])
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
