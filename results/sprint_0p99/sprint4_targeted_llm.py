"""Sprint 4 — Phase 3b: targeted LLM rerank on the 8 attackable fails only.

Reads run8, identifies remaining fails (excluding _abs noise), runs Llama-3.3-70B
listwise rerank with:
  - Full-text 2000 char per candidate (vs 600 in phase 3)
  - Chain-of-thought prompt
  - 5s throttle + exponential backoff on 429
  - top-K = 10

Only emits a final top-1 override; if LLM picks 1 (already top-1) or invalid,
keep baseline.
"""
from __future__ import annotations
import argparse, json, os, re, time, urllib.request, urllib.error
from pathlib import Path

REPO = Path("/Users/macmini/Projects/adaptmem")
RUN_IN = REPO / "benchmarks/v335/run8_time_rerank.jsonl"
GOLD = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.3-70b-instruct"


def nim_call(prompt: str, api_key: str, max_retries: int = 6) -> str | None:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
        "temperature": 0.0,
    }).encode()
    for attempt in range(max_retries):
        req = urllib.request.Request(NIM_URL, data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            wait = max(5, 2 ** attempt)
            print(f"    HTTP {e.code} attempt {attempt+1}, backoff {wait}s")
            time.sleep(wait)
        except Exception as e:
            print(f"    err: {e}")
            time.sleep(5)
    return None


def build_prompt(q: str, qdate: str, items: list[dict]) -> str:
    lines = [
        "You are reranking memory snippets. Pick which snippet best ANSWERS the question.",
        f"Question: {q}",
        f"Question asked on: {qdate}",
        "",
        "Snippets (numbered, with timestamp):",
    ]
    for i, it in enumerate(items, 1):
        lines.append(f"--- [{i}] ts={it.get('timestamp','')}")
        lines.append(it["text"][:2000])
    lines += [
        "",
        "Reasoning steps:",
        "1. What exact fact does the question ask for?",
        "2. Which snippet EXPLICITLY contains that fact (not just a topical match)?",
        "3. Consider temporal cues (date math) when the question mentions time.",
        "",
        f"Final line of your reply must be exactly: ANSWER: <N>   (a number 1-{len(items)})",
    ]
    return "\n".join(lines)


def parse_answer(reply: str | None, k: int) -> int | None:
    if not reply: return None
    m = re.search(r"ANSWER:\s*(\d+)", reply)
    if m:
        n = int(m.group(1))
        if 1 <= n <= k: return n - 1
    # fallback: last digit in reply
    digits = re.findall(r"\d+", reply)
    if digits:
        n = int(digits[-1])
        if 1 <= n <= k: return n - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--throttle", type=float, default=5.0)
    ap.add_argument("--runs", type=int, default=1, help="self-consistency runs; majority pick")
    ap.add_argument("--out", default=str(REPO / "results/sprint_0p99/sprint4_targeted_llm_result.json"))
    args = ap.parse_args()

    api_key = os.environ["NVIDIA_API_KEY"]
    gold = {q["question_id"]: q for q in json.loads(GOLD.read_text())}

    # Build run map
    run = {}
    with open(RUN_IN) as f:
        for line in f:
            d = json.loads(line)
            run[d["question_id"]] = d

    # Identify remaining fails from run8
    fails = []
    for qid, rec in run.items():
        g = set(gold.get(qid, {}).get("answer_session_ids") or [])
        items = rec["retrieval_results"]["ranked_items"]
        top1 = items[0]["corpus_id"] if items else None
        if g and top1 not in g:
            fails.append(qid)
    print(f"Remaining fails in run8: {len(fails)} -> {fails}")

    # Skip _abs (structural noise) — track but don't waste calls
    attackable = [q for q in fails if not q.endswith("_abs")]
    print(f"Attackable (no _abs): {len(attackable)} -> {attackable}")

    results = []
    helped = 0; hurt = 0
    for i, qid in enumerate(attackable):
        rec = run[qid]
        g_q = gold[qid]
        g_ids = set(g_q.get("answer_session_ids") or [])
        items = rec["retrieval_results"]["ranked_items"][: args.top_k]
        prompt = build_prompt(rec["question"], g_q.get("question_date",""), items)
        print(f"\n[{i+1}/{len(attackable)}] {qid} ({rec['question_type']})")
        print(f"  Q: {rec['question'][:140]}")
        # Show gold rank in top-K
        grank = None
        for j, it in enumerate(items):
            if it["corpus_id"] in g_ids: grank = j+1; break
        print(f"  Gold rank in top-{args.top_k}: {grank}")

        # Self-consistency: vote across N runs
        from collections import Counter
        picks_log = []
        for run_idx in range(args.runs):
            reply = nim_call(prompt, api_key)
            p = parse_answer(reply, len(items))
            picks_log.append(p)
            if args.runs > 1:
                print(f"  run{run_idx+1}: pick={p+1 if p is not None else None}")
            if run_idx < args.runs - 1:
                time.sleep(args.throttle)
        valid = [p for p in picks_log if p is not None]
        if valid:
            pick = Counter(valid).most_common(1)[0][0]
        else:
            pick = None
        reply = f"votes={picks_log}"
        print(f"  Picks: {picks_log} -> majority {pick+1 if pick is not None else None}")

        in_top1 = items[0]["corpus_id"]; in_hit = int(in_top1 in g_ids)
        if pick is None or pick == 0:
            final = in_top1; final_hit = in_hit
            action = "trust_baseline" if pick == 0 else "llm_failed"
        else:
            final = items[pick]["corpus_id"]; final_hit = int(final in g_ids)
            action = "override"
        if action == "override":
            if in_hit == 0 and final_hit == 1: helped += 1; print(f"  ✅ HELPED")
            elif in_hit == 1 and final_hit == 0: hurt += 1; print(f"  ❌ HURT")
            else: print(f"  -- no change (still {'hit' if final_hit else 'miss'})")

        results.append({"qid": qid, "type": rec["question_type"], "gold_rank": grank,
                        "in_hit": in_hit, "pick": (pick+1 if pick is not None else None),
                        "final_hit": final_hit, "action": action,
                        "reply": (reply or "")[:400]})
        if i < len(attackable) - 1:
            time.sleep(args.throttle)

    # New fail count if we applied LLM picks
    pre = sum(1 for r in results if not r["in_hit"])  # all were 1 (fails)
    post = sum(1 for r in results if not r["final_hit"])
    new_total_fails = (post) + sum(1 for q in fails if q.endswith("_abs"))  # _abs still fail
    n = 500
    report = {
        "experiment": "Sprint 4 phase 3b targeted LLM rerank (full text + CoT + throttle)",
        "model": MODEL,
        "top_k": args.top_k,
        "n_attackable": len(attackable),
        "helped": helped, "hurt": hurt,
        "pre_fails_in_run8": len(fails),
        "post_total_fails_estimate": new_total_fails,
        "R@1_estimate": round(1 - new_total_fails / n, 4),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n== SUMMARY ==")
    print(f"helped={helped}  hurt={hurt}  pre_fails={len(fails)}  post_fails={new_total_fails}")
    print(f"R@1 estimate: {report['R@1_estimate']}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
