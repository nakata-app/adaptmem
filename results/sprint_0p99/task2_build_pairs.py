"""Task 2.1 — Build (q, doc, label) training pairs from LongMemEval train split.

For each train question:
  - Positive pairs: q with every gold session (text=answer text)
  - Hard negatives: q with top-K bi-encoder retrieved docs that are NOT in
    answer_session_ids (taken from run5_v335 jsonl, the FT-300 hybrid_v4 run).

Output: JSONL with one record per pair: {"q","doc","label"}.
Used to fine-tune cross-encoder/ms-marco-MiniLM-L-12-v2.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path("/Users/macmini/Projects/adaptmem")
RUN5 = REPO / "benchmarks/v335/run5_v335_hybrid_v4_ft300.jsonl"
GOLD = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")
SPLIT = REPO / "benchmarks/data/split_ids_100_400.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-neg-per-q", type=int, default=8, help="hard negatives per train query")
    ap.add_argument("--max-doc-chars", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "results/sprint_0p99/task2_train_pairs.jsonl"))
    ap.add_argument("--out-val", default=str(REPO / "results/sprint_0p99/task2_val_pairs.jsonl"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    split = json.loads(SPLIT.read_text())
    train_qids = set(split["train_question_ids"])
    print(f"train qids: {len(train_qids)}")

    gold_map = {r["question_id"]: r for r in json.loads(GOLD.read_text())}

    # build per-qid retrieval top-K and gold sessions
    pairs_pos = []
    pairs_neg = []
    by_type = {}
    with RUN5.open() as f:
        for line in f:
            r = json.loads(line)
            qid = r["question_id"]
            if qid not in train_qids:
                continue
            q = r["question"]
            gold = gold_map.get(qid, {})
            answer_ids = set(gold.get("answer_session_ids") or [])
            qtype = r["question_type"]
            by_type[qtype] = by_type.get(qtype, 0) + 1

            # positive: gold sessions' text (from ranked_items if present, else from haystack)
            items_by_id = {it["corpus_id"]: it for it in r["retrieval_results"]["ranked_items"]}
            haystack = dict(zip(gold.get("haystack_session_ids", []), gold.get("haystack_sessions", []) or []))

            for gid in answer_ids:
                text = None
                if gid in items_by_id:
                    text = items_by_id[gid]["text"]
                elif gid in haystack:
                    sess = haystack[gid]
                    if isinstance(sess, list):
                        text = "\n".join(t.get("content", "") if isinstance(t, dict) else str(t) for t in sess)
                    else:
                        text = str(sess)
                if text:
                    pairs_pos.append({"qid": qid, "qtype": qtype, "q": q, "doc": text[: args.max_doc_chars], "label": 1.0})

            # hard negatives: top-K non-gold from retrieval
            negs = [it for it in r["retrieval_results"]["ranked_items"][: args.hard_neg_per_q * 3]
                    if it["corpus_id"] not in answer_ids]
            rng.shuffle(negs)
            for it in negs[: args.hard_neg_per_q]:
                pairs_neg.append({"qid": qid, "qtype": qtype, "q": q,
                                  "doc": it["text"][: args.max_doc_chars], "label": 0.0})

    print(f"by type (train): {by_type}")
    print(f"positives: {len(pairs_pos)}, hard negatives: {len(pairs_neg)}, total: {len(pairs_pos)+len(pairs_neg)}")

    # train/val split by qid (so no query crosses)
    all_qids = sorted(train_qids)
    rng.shuffle(all_qids)
    n_val = max(1, int(len(all_qids) * args.val_frac))
    val_qids = set(all_qids[:n_val])

    def write(path: str, recs: list) -> int:
        with open(path, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        return len(recs)

    all_pairs = pairs_pos + pairs_neg
    rng.shuffle(all_pairs)
    train_recs = [r for r in all_pairs if r["qid"] not in val_qids]
    val_recs = [r for r in all_pairs if r["qid"] in val_qids]
    n_tr = write(args.out, train_recs)
    n_va = write(args.out_val, val_recs)
    print(f"train pairs: {n_tr} -> {args.out}")
    print(f"val pairs:   {n_va} -> {args.out_val}")


if __name__ == "__main__":
    main()
