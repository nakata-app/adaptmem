"""Task 3 — Time-aware rerank for temporal-reasoning subset.

Idea: parse relative-time phrases in the question + the question_date, derive
a target date, then add a proximity boost to ranked_items whose timestamp is
close to that target. If no temporal keyword is found, fall back to baseline
(do not touch).

We deliberately keep this surgical: pure post-hoc reranking of
benchmarks/v335/run5_v335_hybrid_v4_ft300.jsonl top-K, no embedder change.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path("/Users/macmini/Projects/adaptmem")
RUN5 = REPO / "benchmarks/v335/run5_v335_hybrid_v4_ft300.jsonl"
GOLD = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")

WORD2NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "half": 0.5, "couple": 2, "few": 3, "several": 4,
}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def parse_question_date(s: str) -> datetime | None:
    # format: "2023/02/01 (Wed) 08:41"
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})\s*\([A-Za-z]{3}\)\s*(\d{2}):(\d{2})", s or "")
    if not m:
        return None
    y, mo, d, hh, mm = map(int, m.groups())
    return datetime(y, mo, d, hh, mm)


def parse_doc_ts(s: str) -> datetime | None:
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})\s*\([A-Za-z]{3}\)\s*(\d{2}):(\d{2})", s or "")
    if not m:
        return None
    y, mo, d, hh, mm = map(int, m.groups())
    return datetime(y, mo, d, hh, mm)


def to_int(tok: str) -> int | None:
    tok = tok.lower().strip()
    if tok.isdigit():
        return int(tok)
    return int(WORD2NUM[tok]) if tok in WORD2NUM else None


def extract_temporal_target(question: str, qdate: datetime | None) -> tuple[datetime | None, int]:
    """Return (target_date, sigma_days). sigma is the window half-width.
    If no temporal keyword, returns (None, 0) -- caller should skip boost.
    """
    if qdate is None:
        return None, 0
    q = question.lower()

    # "X days ago" / "X weeks ago" / "X months ago" / "X years ago"
    m = re.search(r"\b(\d+|" + "|".join(WORD2NUM) + r")\s+(day|week|month|year)s?\s+ago\b", q)
    if m:
        n = to_int(m.group(1))
        unit = m.group(2)
        if n is not None:
            if unit == "day":
                return qdate - timedelta(days=n), 1
            if unit == "week":
                return qdate - timedelta(weeks=n), 3
            if unit == "month":
                return qdate - timedelta(days=int(round(n * 30.4))), 7
            if unit == "year":
                return qdate - timedelta(days=int(round(n * 365.25))), 30

    # "yesterday"
    if re.search(r"\byesterday\b", q):
        return qdate - timedelta(days=1), 1

    # "last week"
    if re.search(r"\blast\s+week\b", q):
        return qdate - timedelta(weeks=1), 4

    # "last month"
    if re.search(r"\blast\s+month\b", q):
        return qdate - timedelta(days=30), 8

    # "last year"
    if re.search(r"\blast\s+year\b", q):
        return qdate - timedelta(days=365), 30

    # "last Tuesday" etc.
    m = re.search(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", q)
    if m:
        target_wd = WEEKDAYS.index(m.group(1))
        # most recent occurrence strictly before qdate
        diff = (qdate.weekday() - target_wd) % 7
        if diff == 0:
            diff = 7
        return qdate - timedelta(days=diff), 1

    # "this week" / "this month"
    if re.search(r"\bthis\s+week\b", q):
        return qdate, 4
    if re.search(r"\bthis\s+month\b", q):
        return qdate, 10

    return None, 0


def proximity(doc_ts: datetime | None, target: datetime, sigma_days: int) -> float:
    if doc_ts is None:
        return 0.0
    dt = abs((doc_ts - target).total_seconds()) / 86400.0
    return math.exp(-((dt / max(1.0, sigma_days)) ** 2))


def load_gold(p: Path) -> dict[str, dict]:
    return {r["question_id"]: r for r in json.loads(p.read_text())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=2.0, help="weight of temporal boost")
    ap.add_argument("--base", default="rrf", choices=["linear", "rrf"], help="orig rank score formula")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--gate-prox", type=float, default=0.0,
                    help="if top-1 doc proximity >= this, skip boost (protects correct baselines)")
    ap.add_argument("--out", default=str(REPO / "results/sprint_0p99/task3_time_rerank.json"))
    args = ap.parse_args()

    gold = load_gold(GOLD)
    print(f"gold: {len(gold)}")

    baseline_hits = 0
    rerank_hits = 0
    total = 0
    skipped = 0  # no temporal keyword
    moved = []
    lost = []
    fail_inspect = []

    with RUN5.open() as f:
        for line in f:
            r = json.loads(line)
            if r["question_type"] != "temporal-reasoning":
                continue
            qid = r["question_id"]
            q = r["question"]
            g = gold.get(qid, {})
            answer_ids = set(g.get("answer_session_ids") or [])
            qdate = parse_question_date(g.get("question_date", ""))
            items = r["retrieval_results"]["ranked_items"][: args.top_k]
            if not items:
                continue
            total += 1

            base_top1 = items[0]["corpus_id"]
            base_hit = int(bool(answer_ids) and base_top1 in answer_ids)
            baseline_hits += base_hit

            target, sigma = extract_temporal_target(q, qdate)
            if target is None:
                # no boost; rerank top1 == baseline top1
                rerank_hits += base_hit
                skipped += 1
                if base_hit == 0:
                    fail_inspect.append({"qid": qid, "q": q[:120], "reason": "no_temporal_keyword"})
                continue

            # gating: if baseline top-1 doc is already close to target, trust it
            if args.gate_prox > 0:
                top1_ts = parse_doc_ts(items[0].get("timestamp", ""))
                top1_prox = proximity(top1_ts, target, sigma)
                if top1_prox >= args.gate_prox:
                    rerank_hits += base_hit
                    if base_hit == 0:
                        fail_inspect.append({"qid": qid, "q": q[:120], "reason": "gated_top1_close"})
                    continue

            # blended score: orig + alpha * proximity
            scored = []
            for idx, it in enumerate(items):
                doc_ts = parse_doc_ts(it.get("timestamp", ""))
                prox = proximity(doc_ts, target, sigma)
                if args.base == "rrf":
                    base_score = 1.0 / (args.rrf_k + idx)
                else:
                    base_score = -idx / args.top_k
                blended = base_score + args.alpha * prox
                scored.append((blended, idx, it))
            scored.sort(key=lambda x: -x[0])
            new_top1 = scored[0][2]["corpus_id"]
            new_hit = int(bool(answer_ids) and new_top1 in answer_ids)
            rerank_hits += new_hit

            if base_hit == 0 and new_hit == 1:
                moved.append({"qid": qid, "q": q[:120], "target": str(target), "sigma": sigma})
            elif base_hit == 1 and new_hit == 0:
                lost.append({"qid": qid, "q": q[:120], "target": str(target), "sigma": sigma})
            if new_hit == 0:
                fail_inspect.append({"qid": qid, "q": q[:120], "reason": "still_fail", "target": str(target)})

    report = {
        "experiment": "Task 3 time-aware rerank (temporal)",
        "top_k": args.top_k,
        "alpha": args.alpha,
        "total": total,
        "baseline_R@1": round(baseline_hits / max(1, total), 4),
        "rerank_R@1": round(rerank_hits / max(1, total), 4),
        "baseline_hits": baseline_hits,
        "rerank_hits": rerank_hits,
        "skipped_no_keyword": skipped,
        "newly_correct_top1": moved,
        "newly_wrong_top1": lost,
        "still_fail": fail_inspect,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["total", "baseline_R@1", "rerank_R@1",
                                              "baseline_hits", "rerank_hits", "skipped_no_keyword"]},
                     indent=2))
    print(f"moved (recovered): {len(moved)}, lost: {len(lost)}")
    print(f"out -> {args.out}")


if __name__ == "__main__":
    main()
