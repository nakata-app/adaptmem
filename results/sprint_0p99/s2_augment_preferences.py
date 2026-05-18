"""Sprint 2 — Task A: synthetic preference paraphrase via DeepSeek V4 Pro (NIM).

For each of the 7 train preference queries:
  - Send Q + gold preference description to the model
  - Ask for N paraphrased queries that reflect the same underlying preference
  - Parse JSON list, dedupe, validate

Each synthetic q is paired with:
  - the same gold doc text (label=1)
  - K hard negatives from the original run5 top-20 non-gold list (label=0)
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
MODEL = "meta/llama-3.3-70b-instruct"  # V4 Pro/Flash timed out repeatedly on NIM; fallback


def call_nim(prompt: str, max_tokens: int = 2000, temperature: float = 0.9,
             timeout: int = 300, retries: int = 2) -> str:
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
            obj = json.loads(raw)
            return obj["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(3 + attempt * 5)
    raise last  # type: ignore[misc]


PROMPT_TMPL = """You are augmenting a training set for a retrieval reranker.

Given:
- An original user question:
  {question}

- The user's underlying preference / context the assistant should retrieve:
  {preference}

Task: produce {n} DIFFERENT paraphrased user questions that, if asked by the
same user later, should retrieve the SAME underlying preference / past-session
context. The paraphrases must:
  - vary in surface form: question phrasing, length, vocabulary, indirectness
  - keep the same underlying request topic (so the preference still applies)
  - sound natural — like real chat turns, not formal queries
  - NOT mention the answer or the preference text verbatim
  - NOT all start with the same words

Return ONLY a JSON array of {n} strings, no commentary. Example:
["new question 1?", "new question 2?", ...]"""


def extract_json_array(text: str) -> list[str]:
    # try to find first [...] block
    m = re.search(r"\[\s*(?:\"[^\"]*\"|'[^']*')[\s\S]*?\]", text)
    if not m:
        # fallback: line-by-line parse
        lines = [l.strip().strip("\"'") for l in text.splitlines() if l.strip() and (l.strip().endswith("?") or "?" in l)]
        return [l for l in lines if 8 < len(l) < 250]
    blob = m.group(0)
    try:
        arr = json.loads(blob)
        return [str(x) for x in arr if isinstance(x, str)]
    except Exception:
        # try with single-quote -> double-quote substitution
        try:
            arr = json.loads(blob.replace("'", '"'))
            return [str(x) for x in arr if isinstance(x, str)]
        except Exception:
            return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=40, help="paraphrases per train preference q")
    ap.add_argument("--hard-neg-per-syn", type=int, default=6)
    ap.add_argument("--out-syn", default=str(REPO / "results/sprint_0p99/s2_syn_preferences.jsonl"))
    ap.add_argument("--out-pairs", default=str(REPO / "results/sprint_0p99/s2_train_pairs_aug.jsonl"))
    ap.add_argument("--out-val", default=str(REPO / "results/sprint_0p99/s2_val_pairs_aug.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    split = json.loads(SPLIT.read_text())
    train_qids = set(split["train_question_ids"])

    gold_map = {r["question_id"]: r for r in json.loads(GOLD.read_text())}

    # Load run5 records for retrieved top-K (for hard-neg)
    run5 = {}
    with RUN5.open() as f:
        for line in f:
            r = json.loads(line)
            run5[r["question_id"]] = r

    # Find 7 train preference queries
    pref_qids = [qid for qid, g in gold_map.items()
                 if qid in train_qids and g.get("question_type") == "single-session-preference"]
    print(f"train preference queries: {len(pref_qids)}")

    # Step 1: generate synthetic queries
    syn_records = []  # one per synthetic q: {orig_qid, syn_q, gold_doc_text}
    for i, qid in enumerate(pref_qids, 1):
        g = gold_map[qid]
        orig_q = g["question"]
        pref_text = g["answer"]
        answer_ids = g.get("answer_session_ids") or []
        if not answer_ids:
            print(f"[{i}/{len(pref_qids)}] {qid}: NO gold session_ids, skip")
            continue
        # gold doc text
        items_by_id = {it["corpus_id"]: it for it in run5[qid]["retrieval_results"]["ranked_items"]}
        haystack = dict(zip(g.get("haystack_session_ids", []), g.get("haystack_sessions", []) or []))
        gold_texts = []
        for gid in answer_ids:
            if gid in items_by_id:
                gold_texts.append(items_by_id[gid]["text"])
            elif gid in haystack:
                s = haystack[gid]
                if isinstance(s, list):
                    gold_texts.append("\n".join((t.get("content", "") if isinstance(t, dict) else str(t)) for t in s))
                else:
                    gold_texts.append(str(s))
        if not gold_texts:
            print(f"[{i}/{len(pref_qids)}] {qid}: cannot resolve gold text, skip")
            continue

        prompt = PROMPT_TMPL.format(question=orig_q, preference=pref_text, n=args.per_query)
        t0 = time.time()
        try:
            raw = call_nim(prompt, max_tokens=4000, temperature=0.9)
        except Exception as e:
            print(f"[{i}/{len(pref_qids)}] {qid}: NIM error {e}")
            continue
        elapsed = time.time() - t0
        paraphrases = extract_json_array(raw)
        # dedupe + filter
        seen = set()
        keep = []
        for p in paraphrases:
            p = p.strip()
            if 8 < len(p) < 300 and p.lower() not in seen and p.lower() != orig_q.lower():
                seen.add(p.lower())
                keep.append(p)
        print(f"[{i}/{len(pref_qids)}] {qid}: got {len(paraphrases)} -> kept {len(keep)} in {elapsed:.1f}s")
        for p in keep:
            for gt in gold_texts:
                syn_records.append({
                    "orig_qid": qid,
                    "syn_q": p,
                    "gold_text": gt[:1800],
                })

    Path(args.out_syn).write_text("\n".join(json.dumps(r) for r in syn_records))
    print(f"\nsynthetic positive records: {len(syn_records)} -> {args.out_syn}")

    # Step 2: build (q, doc, label) pairs
    pairs = []
    for r in syn_records:
        qid = r["orig_qid"]
        p_q = r["syn_q"]
        pairs.append({"qid": f"syn_{qid}", "qtype": "single-session-preference",
                      "q": p_q, "doc": r["gold_text"], "label": 1.0})
        # hard-neg from run5 top-K non-gold of the original qid
        g = gold_map[qid]
        ans = set(g.get("answer_session_ids") or [])
        negs = [it for it in run5[qid]["retrieval_results"]["ranked_items"][:20]
                if it["corpus_id"] not in ans]
        rng.shuffle(negs)
        for it in negs[: args.hard_neg_per_syn]:
            pairs.append({"qid": f"syn_{qid}", "qtype": "single-session-preference",
                          "q": p_q, "doc": it["text"][:1800], "label": 0.0})

    # Load Sprint 1 base pairs + merge
    s1_train = [json.loads(l) for l in (REPO / "results/sprint_0p99/task2_train_pairs.jsonl").open()]
    s1_val = [json.loads(l) for l in (REPO / "results/sprint_0p99/task2_val_pairs.jsonl").open()]

    # Synthetic q's get split into train/val by orig_qid groups
    syn_orig_qids = sorted({r["orig_qid"] for r in syn_records})
    rng.shuffle(syn_orig_qids)
    n_val_orig = max(1, int(len(syn_orig_qids) * args.val_frac))
    val_orig = set(syn_orig_qids[:n_val_orig])
    syn_train = [p for p in pairs if p["qid"].split("_", 1)[1] not in val_orig]
    syn_val = [p for p in pairs if p["qid"].split("_", 1)[1] in val_orig]

    all_train = s1_train + syn_train
    all_val = s1_val + syn_val
    rng.shuffle(all_train)
    rng.shuffle(all_val)

    with open(args.out_pairs, "w") as f:
        for p in all_train:
            f.write(json.dumps(p) + "\n")
    with open(args.out_val, "w") as f:
        for p in all_val:
            f.write(json.dumps(p) + "\n")
    print(f"\nTRAIN pairs total: {len(all_train)} (s1={len(s1_train)} + syn={len(syn_train)}) -> {args.out_pairs}")
    print(f"VAL   pairs total: {len(all_val)} (s1={len(s1_val)} + syn={len(syn_val)}) -> {args.out_val}")


if __name__ == "__main__":
    main()
