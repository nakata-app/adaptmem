"""K-curve sweep + router sub-question analysis.

Three outputs:
  1. K-curve: per-strategy R@K for K in {1,3,5,10} on the chunking ablation
     (already have per-probe ranks, compute R@K directly).
  2. Sub-Q1 (max vs union): on the 500q LongMemEval, do entity-graph and
     FT-stack pick the same gold candidate or different gold candidates?
     If different on a meaningful fraction, union > max.
  3. Sub-Q2 (confidence routing): does per-query entity-graph score
     predict whether entity-graph beats the encoder stack? If yes,
     entity-graph hit count is a runtime router signal.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

# ---- K-curve from existing ablation results ---------------------------
base = json.load(open("/tmp/chunk_strategy_ablation_baseline_result.json"))
ft = json.load(open("/tmp/chunk_strategy_ablation_ft300_result.json"))
strategies = ["A_paragraph_aware__cs800", "B_heading_aware_md__cs800", "C_plus_ast_python__cs800"]

def r_at_k(probes, k):
    n = sum(1 for p in probes if p["rank"] is not None and p["rank"] <= k)
    return 100 * n / len(probes)

def mrr_at_k(probes, k):
    s = 0.0
    for p in probes:
        if p["rank"] is not None and p["rank"] <= k:
            s += 1.0 / p["rank"]
    return s / len(probes)

print("=" * 90)
print("K-CURVE: per-strategy R@K across K (20 probes, cs=800)")
print("=" * 90)
print(f"{'encoder':<10} {'K':>4} {'A_R@K':>8} {'B_R@K':>8} {'C_R@K':>8} {'B-A':>8} {'C-A':>8}")
for label, src in [("baseline", base), ("FT-300", ft)]:
    for K in (1, 3, 5, 10):
        vals = [r_at_k(src["strategies"][s]["probes"], K) for s in strategies]
        print(f"{label:<10} {K:>4} {vals[0]:>8.1f} {vals[1]:>8.1f} {vals[2]:>8.1f} {vals[1]-vals[0]:>+8.1f} {vals[2]-vals[0]:>+8.1f}")
    print()

# Markdown-only sub-slice (5 probes)
print("Markdown-only sub-slice (5/20 probes, md-MRR truth):")
print(f"{'encoder':<10} {'K':>4} {'A':>8} {'B':>8} {'C':>8} {'B-A':>8} {'C-A':>8}")
for label, src in [("baseline", base), ("FT-300", ft)]:
    for K in (1, 3, 5, 10):
        md_probes = [[p for p in src["strategies"][s]["probes"] if p["expected"].endswith((".md",))] for s in strategies]
        vals = [r_at_k(md, K) for md in md_probes]
        print(f"{label:<10} {K:>4} {vals[0]:>8.1f} {vals[1]:>8.1f} {vals[2]:>8.1f} {vals[1]-vals[0]:>+8.1f} {vals[2]-vals[0]:>+8.1f}")
    print()

# ---- Sub-Q1 + Sub-Q2: entity-graph vs FT-stack on 500q LongMemEval ----
eg = json.load(open("/tmp/longmemeval_entity_graph_result.json"))
sprint4 = json.load(open("/Users/macmini/Projects/adaptmem/results/sprint_0p99/sprint4_trust_gate_result.json"))

# Map per-question gold-or-not + top1 from each system
# entity-graph data per_q already has hit@1
# For sprint4 we need ftv4 raw rank-1 (R@1=0.968) vs final after trust gate (R@1=0.978)
# sprint4 records remaining_fails — we have hit@1 per question via the difference.
# Easier: load ftv4 run jsonl directly to get per-question top-1
LME_GOLD = json.load(open("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json"))
gold_map = {q["question_id"]: set(q.get("answer_session_ids") or []) for q in LME_GOLD}
qtype_map = {q["question_id"]: q["question_type"] for q in LME_GOLD}

# ftv4 raw top-1
ftv4 = {}
with open("/Users/macmini/Projects/adaptmem/benchmarks/v335/run6_v335_hybrid_v4_ftv4.jsonl") as f:
    for line in f:
        r = json.loads(line)
        items = r["retrieval_results"]["ranked_items"]
        ftv4[r["question_id"]] = items[0]["corpus_id"] if items else None

# entity-graph per-question
eg_per = {p["qid"]: p for p in eg["per_q"]}

# Cross-table: per question, hit indicator for each system
both_hit = 0
only_eg = 0
only_ft = 0
neither = 0
same_gold = 0  # both hit AND picked same session
diff_gold = 0  # both hit AND picked different gold sessions (union > max only if multi-gold)

for qid, gold in gold_map.items():
    if qid not in eg_per or qid not in ftv4:
        continue
    eg_top1 = eg_per[qid]["top1"]
    ft_top1 = ftv4[qid]
    eg_hit = bool(gold) and eg_top1 in gold
    ft_hit = bool(gold) and ft_top1 in gold
    if eg_hit and ft_hit:
        both_hit += 1
        if eg_top1 == ft_top1:
            same_gold += 1
        else:
            diff_gold += 1
    elif eg_hit:
        only_eg += 1
    elif ft_hit:
        only_ft += 1
    else:
        neither += 1

print("=" * 90)
print("SUB-Q1: entity-graph top-1 vs ftv4-raw top-1 overlap (500q LongMemEval)")
print("=" * 90)
print(f"  both hit:           {both_hit:>4}  ({100*both_hit/500:.1f}%)")
print(f"    same gold:        {same_gold:>4}  ({100*same_gold/500:.1f}%)   <- max == single path")
print(f"    different gold:   {diff_gold:>4}  ({100*diff_gold/500:.1f}%)   <- union > max")
print(f"  only entity-graph:  {only_eg:>4}  ({100*only_eg/500:.1f}%)   <- graph contributes uniquely")
print(f"  only ftv4:          {only_ft:>4}  ({100*only_ft/500:.1f}%)   <- ftv4 dominant")
print(f"  neither:            {neither:>4}  ({100*neither/500:.1f}%)")
print(f"\n  MAX router R@1 (best-of-two):    {(both_hit + only_eg + only_ft)/500:.4f}")
print(f"  ftv4-alone R@1:                  {(both_hit + only_ft)/500:.4f}")
print(f"  entity-graph-alone R@1:          {(both_hit + only_eg)/500:.4f}")

# Sub-Q2: confidence routing — does entity-graph top1_score predict its own win?
import statistics
eg_win = [eg_per[qid]["top1_score"] for qid, gold in gold_map.items() if qid in eg_per and qid in ftv4 and bool(gold) and eg_per[qid]["top1"] in gold and ftv4[qid] not in gold]
eg_lose = [eg_per[qid]["top1_score"] for qid, gold in gold_map.items() if qid in eg_per and qid in ftv4 and bool(gold) and eg_per[qid]["top1"] not in gold]

print("\n" + "=" * 90)
print("SUB-Q2: entity-graph top1_score distribution — does score predict win?")
print("=" * 90)
print(f"  EG wins (eg-hit, ft-miss):  n={len(eg_win):>3}  median_score={statistics.median(eg_win) if eg_win else 0:.3f}  mean={sum(eg_win)/len(eg_win) if eg_win else 0:.3f}")
print(f"  EG loses (eg-miss):         n={len(eg_lose):>3}  median_score={statistics.median(eg_lose) if eg_lose else 0:.3f}  mean={sum(eg_lose)/len(eg_lose) if eg_lose else 0:.3f}")

# Per-category routing gain
print("\n  Per-category MAX-router gain vs ftv4-alone:")
print(f"  {'category':<30} {'n':>4} {'ftv4_alone':>11} {'max_router':>11} {'lift':>7}")
by_cat = defaultdict(lambda: {"n": 0, "ft_hit": 0, "max_hit": 0})
for qid, gold in gold_map.items():
    if qid not in eg_per or qid not in ftv4:
        continue
    b = by_cat[qtype_map[qid]]
    b["n"] += 1
    eg_hit = bool(gold) and eg_per[qid]["top1"] in gold
    ft_hit = bool(gold) and ftv4[qid] in gold
    if ft_hit: b["ft_hit"] += 1
    if eg_hit or ft_hit: b["max_hit"] += 1
for cat in sorted(by_cat.keys()):
    b = by_cat[cat]
    ft_r = b["ft_hit"]/b["n"]
    max_r = b["max_hit"]/b["n"]
    print(f"  {cat:<30} {b['n']:>4} {ft_r:>11.4f} {max_r:>11.4f} {max_r-ft_r:>+7.4f}")
