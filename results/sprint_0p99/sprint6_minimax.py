"""Sprint 6 — MiniMax M2 targeted rerank on 8 attackable fails.

Same pattern as sprint6_deepseek.py but with MiniMax M2 (reasoning model).
Endpoint: https://api.minimaxi.chat/v1/text/chatcompletion_v2
"""
from __future__ import annotations
import argparse, json, os, re, time, urllib.request, urllib.error
from collections import Counter
from pathlib import Path

REPO = Path("/Users/macmini/Projects/adaptmem")
RUN_IN = REPO / "benchmarks/v335/run8_time_rerank.jsonl"
GOLD = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")
URL = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"


def call(prompt: str, model: str, api_key: str, max_tokens: int = 1500, max_retries: int = 5) -> str | None:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    for attempt in range(max_retries):
        req = urllib.request.Request(URL, data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode())
                if "base_resp" in d and d["base_resp"].get("status_code", 0) != 0:
                    print(f"    minimax base_resp err: {d['base_resp']}")
                    return None
                msg = d["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                # Strip <think> ... </think> blocks if present (M2 may embed thinking in content)
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                # Fallback to reasoning_content if content empty
                if not content:
                    rc = (msg.get("reasoning_content") or "").strip()
                    if rc:
                        content = rc[-500:]
                return content
        except urllib.error.HTTPError as e:
            wait = min(60, 5 + 5 * attempt)
            print(f"    HTTP {e.code} attempt {attempt+1}, backoff {wait}s")
            time.sleep(wait)
        except Exception as e:
            print(f"    err: {e}")
            time.sleep(5)
    return None


def build_prompt(q: str, qdate: str, items: list[dict]) -> str:
    lines = [
        "You are reranking memory snippets. Pick the snippet that BEST answers the question.",
        f"Question: {q}",
        f"Question asked on: {qdate}",
        "",
        "Snippets:",
    ]
    for i, it in enumerate(items, 1):
        lines.append(f"--- [{i}] ts={it.get('timestamp','')}")
        lines.append(it["text"][:2000])
    lines += [
        "",
        "Steps:",
        "1. What exact fact does the question ask for?",
        "2. Which snippet EXPLICITLY contains that fact?",
        "3. Consider temporal cues (date math from QUESTION DATE) when relevant.",
        "",
        f"Final line MUST be exactly: ANSWER: <N>   where N is 1-{len(items)}.",
    ]
    return "\n".join(lines)


def parse_answer(reply: str | None, k: int) -> int | None:
    if not reply: return None
    m = re.search(r"ANSWER:\s*(\d+)", reply)
    if m:
        n = int(m.group(1))
        if 1 <= n <= k: return n - 1
    digits = re.findall(r"\d+", reply)
    if digits:
        n = int(digits[-1])
        if 1 <= n <= k: return n - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MiniMax-M2")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--throttle", type=float, default=1.0)
    ap.add_argument("--out", default=str(REPO / "results/sprint_0p99/sprint6_minimax_result.json"))
    args = ap.parse_args()

    api_key = os.environ["MINIMAX_API_KEY"]
    gold = {q["question_id"]: q for q in json.loads(GOLD.read_text())}
    run = {}
    with open(RUN_IN) as f:
        for line in f:
            d = json.loads(line); run[d["question_id"]] = d

    fails = []
    for qid, rec in run.items():
        g = set(gold.get(qid, {}).get("answer_session_ids") or [])
        top1 = rec["retrieval_results"]["ranked_items"][0]["corpus_id"]
        if g and top1 not in g: fails.append(qid)
    attackable = [q for q in fails if not q.endswith("_abs")]
    print(f"Attackable: {attackable}")
    print(f"Model: {args.model}, runs={args.runs}, max_tokens={args.max_tokens}")

    results = []; helped = 0; hurt = 0
    t0 = time.time()
    for i, qid in enumerate(attackable):
        rec = run[qid]; g_q = gold[qid]
        g_ids = set(g_q.get("answer_session_ids") or [])
        items = rec["retrieval_results"]["ranked_items"][: args.top_k]
        grank = None
        for j, it in enumerate(items):
            if it["corpus_id"] in g_ids: grank = j+1; break
        prompt = build_prompt(rec["question"], g_q.get("question_date",""), items)
        print(f"\n[{i+1}/{len(attackable)}] {qid} ({rec['question_type']})  gold_rank={grank}")
        picks = []
        for r_i in range(args.runs):
            reply = call(prompt, args.model, api_key, max_tokens=args.max_tokens)
            p = parse_answer(reply, len(items))
            picks.append(p)
            print(f"  run{r_i+1}: pick={p+1 if p is not None else None}  tail={(reply or '')[-80:]!r}")
            if r_i < args.runs - 1: time.sleep(args.throttle)
        valid = [p for p in picks if p is not None]
        pick = Counter(valid).most_common(1)[0][0] if valid else None
        in_top1 = items[0]["corpus_id"]; in_hit = int(in_top1 in g_ids)
        if pick is None or pick == 0:
            final_hit = in_hit
            action = "trust_baseline" if pick == 0 else "llm_failed"
        else:
            final_hit = int(items[pick]["corpus_id"] in g_ids)
            action = "override"
        if action == "override":
            if in_hit == 0 and final_hit == 1: helped += 1; print(f"  ✅ HELPED")
            elif in_hit == 1 and final_hit == 0: hurt += 1; print(f"  ❌ HURT")
            else: print(f"  -- no change")
        results.append({"qid": qid, "type": rec["question_type"], "gold_rank": grank,
                        "in_hit": in_hit, "picks": picks,
                        "majority": (pick+1 if pick is not None else None),
                        "final_hit": final_hit, "action": action})
        if i < len(attackable) - 1: time.sleep(args.throttle)

    post = sum(1 for r in results if not r["final_hit"])
    new_total = post + sum(1 for q in fails if q.endswith("_abs"))
    n = 500
    report = {
        "experiment": f"Sprint 6 MiniMax targeted rerank ({args.model})",
        "model": args.model, "top_k": args.top_k, "runs": args.runs,
        "helped": helped, "hurt": hurt,
        "pre_fails_run8": len(fails),
        "post_total_fails": new_total,
        "R@1_estimate": round(1 - new_total / n, 4),
        "wall_clock_s": round(time.time()-t0, 1),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n== SUMMARY ==")
    print(f"helped={helped} hurt={hurt} post_fails={new_total} R@1={report['R@1_estimate']} wall={report['wall_clock_s']}s")


if __name__ == "__main__":
    main()
