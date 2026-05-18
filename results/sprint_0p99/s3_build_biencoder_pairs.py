"""Sprint 3 — S3-A: build (q, gold_doc) positive pairs for bi-encoder FT.

Sources:
  - 100 train queries from LongMemEval-S split (~200 positive pairs given multi-gold)
  - 5448 synthetic paraphrases from Sprint 2 (s2_syn_all.jsonl) — each paired
    with the same gold doc as its origin query.

Output JSONL: one record per pair {"q": ..., "doc": ..., "orig_qid": ..., "qtype": ...}
Used to continue-train FT-300 with MultipleNegativesRankingLoss (in-batch neg).
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
SYN_ALL = REPO / "results/sprint_0p99/s2_syn_all.jsonl"
SYN_PREF = REPO / "results/sprint_0p99/s2_syn_preferences.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-doc-chars", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out", default=str(REPO / "results/sprint_0p99/s3_biencoder_pairs.jsonl"))
    ap.add_argument("--out-val", default=str(REPO / "results/sprint_0p99/s3_biencoder_pairs_val.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    split = json.loads(SPLIT.read_text())
    train_qids = set(split["train_question_ids"])
    gold_map = {r["question_id"]: r for r in json.loads(GOLD.read_text())}

    # ---- Source 1: original 100 train queries ----
    run5 = {}
    with RUN5.open() as f:
        for line in f:
            r = json.loads(line)
            run5[r["question_id"]] = r

    base_records = []
    for qid in train_qids:
        g = gold_map.get(qid)
        if not g: continue
        ans = g.get("answer_session_ids") or []
        if not ans: continue
        items_by_id = {it["corpus_id"]: it for it in run5[qid]["retrieval_results"]["ranked_items"]}
        haystack = dict(zip(g.get("haystack_session_ids", []), g.get("haystack_sessions", []) or []))
        for gid in ans:
            text = None
            if gid in items_by_id:
                text = items_by_id[gid]["text"]
            elif gid in haystack:
                s = haystack[gid]
                text = "\n".join((t.get("content","") if isinstance(t, dict) else str(t)) for t in s) if isinstance(s, list) else str(s)
            if text:
                base_records.append({
                    "q": g["question"], "doc": text[: args.max_doc_chars],
                    "orig_qid": qid, "qtype": g["question_type"], "source": "base",
                })
    print(f"base (100 train q): {len(base_records)} positive pairs")

    # ---- Source 2: Sprint 2 syn_all (all types augmented) ----
    syn_records = []
    if SYN_ALL.exists():
        for line in SYN_ALL.open():
            r = json.loads(line)
            syn_records.append({
                "q": r["syn_q"], "doc": r["gold_text"][: args.max_doc_chars],
                "orig_qid": r["orig_qid"], "qtype": r["qtype"], "source": "syn_all",
            })
    print(f"syn_all: {len(syn_records)} pairs")

    # ---- Source 3: Sprint 2 syn_preferences (also include older 264 records) ----
    syn_pref = []
    if SYN_PREF.exists():
        seen_qkeys = {(r["q"], r["doc"][:80]) for r in syn_records}
        for line in SYN_PREF.open():
            r = json.loads(line)
            key = (r["syn_q"], r["gold_text"][:80])
            if key in seen_qkeys: continue
            syn_pref.append({
                "q": r["syn_q"], "doc": r["gold_text"][: args.max_doc_chars],
                "orig_qid": r["orig_qid"], "qtype": "single-session-preference", "source": "syn_pref",
            })
    print(f"syn_pref (extra): {len(syn_pref)} pairs")

    all_records = base_records + syn_records + syn_pref
    print(f"\nTOTAL: {len(all_records)} positive pairs")

    # split by orig_qid (so no q-leakage)
    orig_qids = sorted({r["orig_qid"] for r in all_records})
    rng.shuffle(orig_qids)
    n_val = max(1, int(len(orig_qids) * args.val_frac))
    val_orig = set(orig_qids[:n_val])
    train = [r for r in all_records if r["orig_qid"] not in val_orig]
    val = [r for r in all_records if r["orig_qid"] in val_orig]
    rng.shuffle(train); rng.shuffle(val)

    with open(args.out, "w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    with open(args.out_val, "w") as f:
        for r in val: f.write(json.dumps(r) + "\n")
    print(f"\nTrain: {len(train)} -> {args.out}")
    print(f"Val:   {len(val)} -> {args.out_val}")

    # by type breakdown
    from collections import Counter
    print("\nby qtype (train):", Counter(r["qtype"] for r in train).most_common())


if __name__ == "__main__":
    main()
