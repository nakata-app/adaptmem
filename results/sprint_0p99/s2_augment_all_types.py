"""Sprint 2 — S2-E: synthetic augmentation for ALL train question types except
single-session-assistant (already at R@1=1.0 with chat-ce-v2).

For each train query (excluding assistant), generate paraphrases via NIM Llama
preserving intent, pair with gold doc (positive) + top-K non-gold (hard-neg).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path

REPO = Path("/Users/macmini/Projects/adaptmem")
RUN5 = REPO / "benchmarks/v335/run5_v335_hybrid_v4_ft300.jsonl"
GOLD = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")
SPLIT = REPO / "benchmarks/data/split_ids_100_400.json"

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.3-70b-instruct"

# train types we augment — assistant is excluded (already perfect after chat-ce-v2)
TARGET_TYPES = {
    "single-session-preference",  # 7 q
    "single-session-user",         # 13 q
    "temporal-reasoning",          # 19 q
    "multi-session",               # 37 q
    "knowledge-update",            # 20 q
}


def call_nim(prompt: str, max_tokens: int = 2000, temperature: float = 0.9,
             timeout: int = 90, retries: int = 2) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                NIM_URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode()
            return json.loads(raw)["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(3 + attempt * 5)
    raise last  # type: ignore[misc]


PROMPTS = {
    "single-session-preference": """You are augmenting a retrieval training set.

Original question: {question}
Underlying preference / context to retrieve: {preference}

Produce {n} DIFFERENT paraphrased questions that would retrieve the same context.
Vary phrasing, length, indirectness. Do NOT mention the answer/preference verbatim.
Return ONLY a JSON array of {n} strings.""",

    "single-session-user": """Augmentation task — paraphrase the user's question while
keeping the same underlying information need.

Original question: {question}
Answer (what should be retrieved): {preference}

Produce {n} paraphrases asking the SAME thing in different ways (varied phrasing,
politeness, length, sentence shape). Do NOT mention the answer verbatim.
Return ONLY a JSON array of {n} strings.""",

    "temporal-reasoning": """Augmentation task — paraphrase a temporal-reasoning question.

Original question: {question}
Answer to recall: {preference}

Produce {n} paraphrases asking about the SAME temporal event/period, varying
phrasing (e.g. "last X", "X ago", "back in", "recently"). Keep the relative time
expression but use different words. Do NOT mention the answer verbatim.
Return ONLY a JSON array of {n} strings.""",

    "multi-session": """Augmentation task — paraphrase a multi-session count/aggregation question.

Original question: {question}
Answer to be reconstructed across multiple past sessions: {preference}

Produce {n} paraphrases asking the SAME aggregation in different ways (count, total,
sum, list). Vary phrasing. Do NOT mention the numeric answer verbatim.
Return ONLY a JSON array of {n} strings.""",

    "knowledge-update": """Augmentation task — paraphrase a knowledge-update question
about something the user previously mentioned (and possibly updated).

Original question: {question}
Latest known state to retrieve: {preference}

Produce {n} paraphrases asking about the user's CURRENT state of the same topic,
varying phrasing. Do NOT mention the answer verbatim.
Return ONLY a JSON array of {n} strings.""",
}


def extract_json_array(text: str) -> list[str]:
    m = re.search(r"\[\s*(?:\"[^\"]*\"|'[^']*')[\s\S]*?\]", text)
    if m:
        try:
            return [str(x) for x in json.loads(m.group(0)) if isinstance(x, str)]
        except Exception:
            try:
                return [str(x) for x in json.loads(m.group(0).replace("'", '"')) if isinstance(x, str)]
            except Exception:
                pass
    return [l.strip().strip("\"',") for l in text.splitlines()
            if l.strip() and "?" in l and 8 < len(l.strip()) < 300]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=30)
    ap.add_argument("--hard-neg-per-syn", type=int, default=6)
    ap.add_argument("--out-syn", default=str(REPO / "results/sprint_0p99/s2_syn_all.jsonl"))
    ap.add_argument("--out-pairs", default=str(REPO / "results/sprint_0p99/s2_train_pairs_all.jsonl"))
    ap.add_argument("--out-val", default=str(REPO / "results/sprint_0p99/s2_val_pairs_all.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.08)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    split = json.loads(SPLIT.read_text())
    train_qids = set(split["train_question_ids"])
    gold_map = {r["question_id"]: r for r in json.loads(GOLD.read_text())}
    run5 = {}
    with RUN5.open() as f:
        for line in f:
            r = json.loads(line)
            run5[r["question_id"]] = r

    target = [qid for qid, g in gold_map.items()
              if qid in train_qids and g.get("question_type") in TARGET_TYPES]
    print(f"target queries: {len(target)} across {TARGET_TYPES}")

    syn = []
    t_start = time.time()
    for idx, qid in enumerate(target, 1):
        g = gold_map[qid]
        qtype = g["question_type"]
        orig_q = g["question"]
        pref_text = g["answer"]
        answer_ids = g.get("answer_session_ids") or []
        if not answer_ids:
            continue
        items_by_id = {it["corpus_id"]: it for it in run5[qid]["retrieval_results"]["ranked_items"]}
        haystack = dict(zip(g.get("haystack_session_ids", []), g.get("haystack_sessions", []) or []))
        gold_texts = []
        for gid in answer_ids:
            if gid in items_by_id:
                gold_texts.append(items_by_id[gid]["text"])
            elif gid in haystack:
                s = haystack[gid]
                gold_texts.append("\n".join((t.get("content", "") if isinstance(t, dict) else str(t)) for t in s) if isinstance(s, list) else str(s))
        if not gold_texts:
            continue

        prompt = PROMPTS[qtype].format(question=orig_q, preference=pref_text, n=args.per_query)
        t0 = time.time()
        try:
            raw = call_nim(prompt, max_tokens=3000, temperature=0.9)
        except Exception as e:
            print(f"[{idx}/{len(target)}] {qid} ({qtype}): NIM error {type(e).__name__}: {e}")
            continue
        paras = extract_json_array(raw)
        seen = {orig_q.lower()}
        keep = []
        for p in paras:
            p = p.strip().strip("\"'")
            if 8 < len(p) < 300 and p.lower() not in seen:
                seen.add(p.lower())
                keep.append(p)
        elapsed = time.time() - t0
        print(f"[{idx}/{len(target)}] {qid} ({qtype}): {len(paras)} -> {len(keep)} ({elapsed:.1f}s)")
        for p in keep:
            for gt in gold_texts:
                syn.append({"orig_qid": qid, "qtype": qtype, "syn_q": p, "gold_text": gt[:1800]})

    print(f"\ntotal syn records: {len(syn)}  total time: {time.time()-t_start:.0f}s")
    Path(args.out_syn).write_text("\n".join(json.dumps(r) for r in syn))

    # Build (q, doc, label) pairs
    pairs = []
    for r in syn:
        qid = r["orig_qid"]; p_q = r["syn_q"]; qtype = r["qtype"]
        pairs.append({"qid": f"syn_{qid}", "qtype": qtype, "q": p_q, "doc": r["gold_text"], "label": 1.0})
        ans = set(gold_map[qid].get("answer_session_ids") or [])
        negs = [it for it in run5[qid]["retrieval_results"]["ranked_items"][:20]
                if it["corpus_id"] not in ans]
        rng.shuffle(negs)
        for it in negs[: args.hard_neg_per_syn]:
            pairs.append({"qid": f"syn_{qid}", "qtype": qtype, "q": p_q, "doc": it["text"][:1800], "label": 0.0})

    # Merge with Sprint 1 base + Sprint 2 preference syn (cumulative)
    s1_train = [json.loads(l) for l in (REPO / "results/sprint_0p99/task2_train_pairs.jsonl").open()]
    s1_val = [json.loads(l) for l in (REPO / "results/sprint_0p99/task2_val_pairs.jsonl").open()]

    # train/val split by orig qid groups
    orig_qids = sorted({r["orig_qid"] for r in syn})
    rng.shuffle(orig_qids)
    n_val = max(1, int(len(orig_qids) * args.val_frac))
    val_orig = set(orig_qids[:n_val])
    syn_train = [p for p in pairs if p["qid"].split("_", 1)[1] not in val_orig]
    syn_val = [p for p in pairs if p["qid"].split("_", 1)[1] in val_orig]

    all_train = s1_train + syn_train
    all_val = s1_val + syn_val
    rng.shuffle(all_train); rng.shuffle(all_val)
    with open(args.out_pairs, "w") as f:
        for p in all_train: f.write(json.dumps(p) + "\n")
    with open(args.out_val, "w") as f:
        for p in all_val: f.write(json.dumps(p) + "\n")
    print(f"\nTRAIN total: {len(all_train)} (s1={len(s1_train)} + syn={len(syn_train)}) -> {args.out_pairs}")
    print(f"VAL   total: {len(all_val)} (s1={len(s1_val)} + syn={len(syn_val)}) -> {args.out_val}")


if __name__ == "__main__":
    main()
