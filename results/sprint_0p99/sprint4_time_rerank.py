"""Sprint 4 — Phase 2: time-aware rerank on temporal-reasoning subset.

Consumes run7 (trust-gate output) and applies the same time-aware rerank logic
proven in Sprint 1 (task3_time_rerank.py): regex temporal phrase -> target date
-> Gaussian proximity boost over top-K, with gating to protect already-correct
top-1s.

Best config from Sprint 1: --base rrf --alpha 0.01 --gate-prox 0.3
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path("/Users/macmini/Projects/adaptmem")
RUN_IN = REPO / "benchmarks/v335/run7_trust_gate.jsonl"
GOLD = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")

WORD2NUM = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
            "seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,
            "half":0.5,"couple":2,"few":3,"several":4}
WEEKDAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]


def parse_dt(s):
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})\s*\([A-Za-z]{3}\)\s*(\d{2}):(\d{2})", s or "")
    if not m: return None
    y,mo,d,hh,mm = map(int, m.groups())
    return datetime(y,mo,d,hh,mm)


def to_int(tok):
    tok = tok.lower().strip()
    if tok.isdigit(): return int(tok)
    return int(WORD2NUM[tok]) if tok in WORD2NUM else None


def extract_target(question, qdate):
    if qdate is None: return None, 0
    q = question.lower()
    m = re.search(r"\b(\d+|" + "|".join(WORD2NUM) + r")\s+(day|week|month|year)s?\s+ago\b", q)
    if m:
        n = to_int(m.group(1)); unit = m.group(2)
        if n is not None:
            if unit == "day":   return qdate - timedelta(days=n), 1
            if unit == "week":  return qdate - timedelta(weeks=n), 3
            if unit == "month": return qdate - timedelta(days=int(round(n*30.4))), 7
            if unit == "year":  return qdate - timedelta(days=int(round(n*365.25))), 30
    if re.search(r"\byesterday\b", q): return qdate - timedelta(days=1), 1
    if re.search(r"\blast\s+week\b", q): return qdate - timedelta(weeks=1), 4
    if re.search(r"\blast\s+month\b", q): return qdate - timedelta(days=30), 8
    if re.search(r"\blast\s+year\b", q): return qdate - timedelta(days=365), 30
    m = re.search(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", q)
    if m:
        wd = WEEKDAYS.index(m.group(1))
        diff = (qdate.weekday() - wd) % 7 or 7
        return qdate - timedelta(days=diff), 1
    if re.search(r"\bthis\s+week\b", q): return qdate, 4
    if re.search(r"\bthis\s+month\b", q): return qdate, 10
    return None, 0


def proximity(doc_ts, target, sigma):
    if doc_ts is None: return 0.0
    dt = abs((doc_ts - target).total_seconds()) / 86400.0
    return math.exp(-((dt / max(1.0, sigma)) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(RUN_IN))
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--gate-prox", type=float, default=0.3)
    ap.add_argument("--out", default=str(REPO / "results/sprint_0p99/sprint4_time_rerank_result.json"))
    ap.add_argument("--out-jsonl", default=str(REPO / "benchmarks/v335/run8_time_rerank.jsonl"))
    args = ap.parse_args()

    gold = {r["question_id"]: r for r in json.loads(GOLD.read_text())}

    by_type = defaultdict(lambda: {"n":0,"in":0,"out":0})
    moved, lost = [], []
    out_records = []

    with open(args.inp) as f:
        for line in f:
            r = json.loads(line)
            qid = r["question_id"]; qtype = r["question_type"]
            items = r["retrieval_results"]["ranked_items"][: args.top_k]
            g = gold.get(qid, {})
            answer_ids = set(g.get("answer_session_ids") or [])
            in_top1 = items[0]["corpus_id"] if items else None
            in_hit = int(bool(answer_ids) and in_top1 in answer_ids)

            by_type[qtype]["n"] += 1
            by_type[qtype]["in"] += in_hit

            if qtype != "temporal-reasoning":
                out_records.append(r)
                by_type[qtype]["out"] += in_hit
                continue

            qdate = parse_dt(g.get("question_date", ""))
            target, sigma = extract_target(r["question"], qdate)

            if target is None:
                out_records.append(r)
                by_type[qtype]["out"] += in_hit
                continue

            # gating: trust baseline if top-1 doc already close to target
            if args.gate_prox > 0:
                top1_ts = parse_dt(items[0].get("timestamp", ""))
                if proximity(top1_ts, target, sigma) >= args.gate_prox:
                    out_records.append(r)
                    by_type[qtype]["out"] += in_hit
                    continue

            # blended score over top-K
            scored = []
            for idx, it in enumerate(items):
                doc_ts = parse_dt(it.get("timestamp", ""))
                base = 1.0 / (args.rrf_k + idx)
                prox = proximity(doc_ts, target, sigma)
                scored.append((base + args.alpha * prox, idx, it))
            scored.sort(key=lambda x: -x[0])
            new_order = [it for _, _, it in scored]
            new_top1 = new_order[0]["corpus_id"]
            new_hit = int(bool(answer_ids) and new_top1 in answer_ids)

            if in_hit == 0 and new_hit == 1:
                moved.append({"qid": qid, "target": str(target), "sigma": sigma})
            elif in_hit == 1 and new_hit == 0:
                lost.append({"qid": qid, "target": str(target), "sigma": sigma})

            by_type[qtype]["out"] += new_hit
            # write reordered jsonl record
            new_rec = dict(r)
            new_rec["retrieval_results"] = dict(r["retrieval_results"])
            full_items = r["retrieval_results"]["ranked_items"]
            new_rec["retrieval_results"]["ranked_items"] = new_order + full_items[args.top_k:]
            out_records.append(new_rec)

    n = sum(b["n"] for b in by_type.values())
    R_in = sum(b["in"] for b in by_type.values()) / n
    R_out = sum(b["out"] for b in by_type.values()) / n

    # remaining fails
    rem_fails = []
    for rec in out_records:
        qid = rec["question_id"]
        g = set(gold.get(qid, {}).get("answer_session_ids") or [])
        items = rec["retrieval_results"]["ranked_items"]
        top1 = items[0]["corpus_id"] if items else None
        if g and top1 not in g:
            rem_fails.append((qid, rec["question_type"]))

    report = {
        "experiment": "Sprint 4 phase 2 time-aware rerank on temporal",
        "input": args.inp,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "gate_prox": args.gate_prox,
        "R@1_in": round(R_in, 4),
        "R@1_out": round(R_out, 4),
        "fails_in": n - sum(b["in"] for b in by_type.values()),
        "fails_out": n - sum(b["out"] for b in by_type.values()),
        "moved": moved, "lost": lost,
        "by_type": {t: {"n":b["n"], "R@1_in":round(b["in"]/b["n"],4), "R@1_out":round(b["out"]/b["n"],4),
                        "fails_out": b["n"]-b["out"]} for t,b in sorted(by_type.items())},
        "remaining_fails": sorted(rem_fails, key=lambda x:(x[1],x[0])),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    with open(args.out_jsonl, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
