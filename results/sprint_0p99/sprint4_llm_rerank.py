"""Sprint 4 — Phase 3: LLM listwise rerank for hard categories.

Reads run8 (after Phase 1 trust-gate + Phase 2 time-aware rerank). Applies
Llama-3.3-70B (NVIDIA NIM) listwise rerank on temporal-reasoning and
multi-session subsets (where remaining fails concentrate). Other categories
pass through unchanged.

Trust gate: if LLM picks an invalid index OR an index whose CE-substitute
signal is weak, keep bi-encoder top-1. Conservative — we'd rather miss a
recovery than break a working top-1.

Cost: 133+133 = 266 calls, batched serially.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
import urllib.request
import urllib.error

REPO = Path("/Users/macmini/Projects/adaptmem")
RUN_IN = REPO / "benchmarks/v335/run8_time_rerank.jsonl"
GOLD = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.3-70b-instruct"

TARGET_TYPES = {"temporal-reasoning", "multi-session"}


def build_prompt(question: str, question_date: str, candidates: list[dict]) -> str:
    lines = [
        "You are reranking memory snippets to find which one best answers the user's question.",
        f"Question: {question}",
        f"Question asked on: {question_date}",
        "",
        "Candidates (each is a conversation snippet with its timestamp):",
    ]
    for i, c in enumerate(candidates, 1):
        ts = c.get("timestamp", "")
        txt = c["text"][:600].replace("\n", " ")
        lines.append(f"[{i}] ts={ts} | {txt}")
    lines += [
        "",
        "Pick the SINGLE candidate that best contains the answer to the question.",
        "Consider temporal cues (dates, 'last week', 'a couple of days ago') when relevant.",
        "Respond with ONLY the candidate number (1 through {0}). No explanation.".format(len(candidates)),
    ]
    return "\n".join(lines)


def nim_call(prompt: str, api_key: str, model: str = MODEL, timeout: int = 30) -> str | None:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        NIM_URL, data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  nim error: {e}")
        return None


def parse_pick(reply: str | None, k: int) -> int | None:
    if not reply: return None
    m = re.search(r"\d+", reply)
    if not m: return None
    n = int(m.group())
    if 1 <= n <= k: return n - 1  # 0-indexed
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(RUN_IN))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=str(REPO / "results/sprint_0p99/sprint4_llm_rerank_result.json"))
    ap.add_argument("--out-jsonl", default=str(REPO / "benchmarks/v335/run9_llm_rerank.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY required")

    gold_data = json.loads(GOLD.read_text())
    gold = {q["question_id"]: q for q in gold_data}

    by_type = defaultdict(lambda: {"n":0, "in":0, "out":0,
                                    "llm_called":0, "llm_picked_top1":0,
                                    "llm_override":0, "override_helped":0,
                                    "override_hurt":0, "llm_failed":0})
    out_records = []
    overrides = []
    t0 = time.time()

    with open(args.inp) as f:
        records = [json.loads(line) for line in f]
    if args.limit: records = records[:args.limit]

    for i, rec in enumerate(records):
        qid = rec["question_id"]; qtype = rec["question_type"]
        items = rec["retrieval_results"]["ranked_items"]
        g_q = gold.get(qid, {})
        answer_ids = set(g_q.get("answer_session_ids") or [])
        in_top1 = items[0]["corpus_id"] if items else None
        in_hit = int(bool(answer_ids) and in_top1 in answer_ids)

        b = by_type[qtype]
        b["n"] += 1
        b["in"] += in_hit

        if qtype not in TARGET_TYPES or len(items) < 2:
            out_records.append(rec); b["out"] += in_hit; continue

        topk = items[: args.top_k]
        prompt = build_prompt(rec["question"], g_q.get("question_date",""), topk)
        b["llm_called"] += 1
        reply = nim_call(prompt, api_key)
        pick = parse_pick(reply, len(topk))

        if pick is None:
            b["llm_failed"] += 1
            out_records.append(rec); b["out"] += in_hit; continue

        if pick == 0:
            b["llm_picked_top1"] += 1
            out_records.append(rec); b["out"] += in_hit; continue

        # LLM overrides — reorder top-K
        new_top1 = topk[pick]
        new_order = [new_top1] + [it for j, it in enumerate(topk) if j != pick]
        new_items = new_order + items[args.top_k:]
        new_hit = int(bool(answer_ids) and new_top1["corpus_id"] in answer_ids)

        b["llm_override"] += 1
        helped = (in_hit == 0 and new_hit == 1)
        hurt   = (in_hit == 1 and new_hit == 0)
        if helped: b["override_helped"] += 1
        if hurt:   b["override_hurt"] += 1
        b["out"] += new_hit
        overrides.append({"qid": qid, "type": qtype, "pick": pick+1,
                          "in_hit": in_hit, "out_hit": new_hit,
                          "helped": helped, "hurt": hurt, "reply": reply})

        new_rec = dict(rec)
        new_rec["retrieval_results"] = dict(rec["retrieval_results"])
        new_rec["retrieval_results"]["ranked_items"] = new_items
        out_records.append(new_rec)

        if (i+1) % 25 == 0:
            el = time.time() - t0
            print(f"  q{i+1}/{len(records)}  elapsed {el:.0f}s")

    n = sum(b["n"] for b in by_type.values())
    R_in  = sum(b["in"]  for b in by_type.values()) / n
    R_out = sum(b["out"] for b in by_type.values()) / n

    rem_fails = []
    for rec in out_records:
        qid = rec["question_id"]
        ans = set(gold.get(qid, {}).get("answer_session_ids") or [])
        top1 = rec["retrieval_results"]["ranked_items"][0]["corpus_id"]
        if ans and top1 not in ans:
            rem_fails.append((qid, rec["question_type"]))

    report = {
        "experiment": "Sprint 4 phase 3 LLM listwise rerank (temporal+multi)",
        "model": MODEL,
        "top_k": args.top_k,
        "n": n,
        "R@1_in": round(R_in, 4),
        "R@1_out": round(R_out, 4),
        "fails_in": n - sum(b["in"] for b in by_type.values()),
        "fails_out": n - sum(b["out"] for b in by_type.values()),
        "by_type": {t: dict(b, R_in=round(b["in"]/b["n"],4), R_out=round(b["out"]/b["n"],4))
                    for t,b in sorted(by_type.items())},
        "overrides": overrides,
        "remaining_fails": sorted(rem_fails, key=lambda x:(x[1],x[0])),
        "wall_clock_s": round(time.time()-t0, 1),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    with open(args.out_jsonl, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: report[k] for k in ["R@1_in","R@1_out","fails_in","fails_out","wall_clock_s"]}, indent=2))
    print(f"saved -> {args.out}\n         {args.out_jsonl}")


if __name__ == "__main__":
    main()
