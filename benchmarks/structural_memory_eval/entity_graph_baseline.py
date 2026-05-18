"""LongMemEval graph-traversal substrate baseline.

Tests the "structural extraction + entity traversal" framing from
MemPalace #1384 (xg-gh-25's alternative to chunking + vector retrieval).

Substrate:
  - For each question, extract entities from query + all haystack sessions
  - Build inverted index: entity -> {session_ids it appears in}
  - Rank sessions by IDF-weighted entity overlap with query
  - Compare against bi-encoder (ftv4) and full FT-300 + hybrid_v4 + rerank stack

Entity extraction (deliberately simple — mempalace's heuristics over
regex; no LLM, no NER, no fine-tuning):
  - Capitalized multi-word phrases (proper nouns: "Apache Kafka",
    "Susan", "New York")
  - Standalone capitalized words >= 4 chars not in stopword set
  - Dates (YYYY, YYYY-MM-DD, "March 5", "5 days ago")
  - Numbers with units (5km, 10 years, $50)
  - Technical tokens (CamelCase, snake_case, kebab-case identifiers)
  - URL/email tokens
  - Quoted strings of 2-6 words

Scoring: per-query IDF (computed over haystack of THIS question, since
LongMemEval is per-question independent). Session score = sum of
log(N / df(e)) for every shared entity e. Ties broken by entity count.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")
OUT = Path("/tmp/longmemeval_entity_graph_result.json")

STOPWORDS = {
    "The", "This", "That", "These", "Those", "There", "Their", "They",
    "What", "When", "Where", "Why", "How", "Who", "Which",
    "Hello", "Hi", "Hey", "Thanks", "Thank", "Please", "Sure", "Okay",
    "Yes", "No", "Maybe", "Some", "Any", "All", "Each", "Every",
    "User", "Assistant", "Bot", "AI", "Human",
    "Could", "Would", "Should", "Will", "Shall", "May", "Might",
    "Have", "Has", "Had", "Was", "Were", "Been", "Being",
    "Get", "Got", "Make", "Made", "Take", "Took", "Give", "Gave",
    "Know", "Knew", "Think", "Thought", "See", "Saw", "Say", "Said",
    "I", "You", "He", "She", "It", "We", "Me", "Him", "Her", "Us", "Them",
    "Mr", "Ms", "Mrs", "Dr",
    "Like", "Want", "Need", "Use", "Used", "Try", "Tried",
}

PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]+){0,3})\b")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*|"
    r"\d+\s+(?:year|month|week|day|hour|minute)s?\s+ago)\b",
    re.IGNORECASE,
)
NUM_UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?\s*(?:km|mi|kg|lb|cm|mm|ft|in|m|kg|g|mg|"
    r"hour|min|sec|year|month|day|week|"
    r"%|usd|eur|gbp|tl|jpy|inr|cny))\b",
    re.IGNORECASE,
)
DOLLAR_RE = re.compile(r"\$\d+(?:,\d{3})*(?:\.\d+)?")
URL_RE = re.compile(r"https?://\S+|\b\w+@\w+\.\w+")
TECH_TOKEN_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+_[a-z]+(?:_[a-z]+)*|[a-z]+-[a-z]+(?:-[a-z]+)*)\b")
QUOTED_RE = re.compile(r'"([^"]{3,80})"|\'([^\']{3,80})\'')


def extract_entities(text: str) -> set[str]:
    ents: set[str] = set()
    for m in PROPER_NOUN_RE.findall(text):
        m = m.strip()
        if m and m not in STOPWORDS and len(m) >= 3:
            ents.add(m.lower())
    for m in DATE_RE.findall(text):
        ents.add(m.lower())
    for m in NUM_UNIT_RE.findall(text):
        ents.add(m.lower().replace(" ", ""))
    for m in DOLLAR_RE.findall(text):
        ents.add(m.lower())
    for m in URL_RE.findall(text):
        ents.add(m.lower())
    for m in TECH_TOKEN_RE.findall(text):
        if len(m) >= 4:
            ents.add(m.lower())
    for a, b in QUOTED_RE.findall(text):
        s = (a or b).strip().lower()
        if 8 <= len(s) <= 80:
            ents.add(s)
    return ents


def session_text(session_turns: list[dict]) -> str:
    return "\n".join(f"{t.get('role', '')}: {t.get('content', '')}" for t in session_turns)


def score_one_question(q: dict) -> dict:
    query = q["question"]
    gold = set(q.get("answer_session_ids") or [])
    session_ids = q["haystack_session_ids"]
    sessions = q["haystack_sessions"]
    n_sessions = len(sessions)

    # Extract entities per session
    sess_ents = [extract_entities(session_text(s)) for s in sessions]
    # Document frequency per entity
    df: Counter = Counter()
    for ents in sess_ents:
        for e in ents:
            df[e] += 1

    # Query entities
    q_ents = extract_entities(query)

    # Rank
    scores = []
    for sid, ents in zip(session_ids, sess_ents):
        shared = q_ents & ents
        if not shared:
            scores.append((sid, 0.0, 0))
            continue
        idf_sum = sum(math.log((n_sessions + 1) / df[e]) for e in shared)
        scores.append((sid, idf_sum, len(shared)))

    scores.sort(key=lambda x: (-x[1], -x[2], x[0]))
    ranked_ids = [s[0] for s in scores]

    hit_at = {}
    for k in (1, 5, 10):
        topk = set(ranked_ids[:k])
        hit_at[k] = int(bool(gold) and bool(gold & topk))

    return {
        "qid": q["question_id"],
        "qtype": q["question_type"],
        "n_query_entities": len(q_ents),
        "gold": list(gold),
        "top1": ranked_ids[0] if ranked_ids else None,
        "top1_score": scores[0][1] if scores else 0.0,
        "hit@1": hit_at[1],
        "hit@5": hit_at[5],
        "hit@10": hit_at[10],
    }


def main() -> int:
    print(f"loading {DATA}...", flush=True)
    data = json.loads(DATA.read_text())
    print(f"  {len(data)} questions", flush=True)

    t0 = time.time()
    results = []
    by_type = defaultdict(lambda: {"n": 0, "h1": 0, "h5": 0, "h10": 0})
    for i, q in enumerate(data):
        r = score_one_question(q)
        results.append(r)
        b = by_type[r["qtype"]]
        b["n"] += 1
        b["h1"] += r["hit@1"]
        b["h5"] += r["hit@5"]
        b["h10"] += r["hit@10"]
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  q{i+1}/{len(data)}  el {el:.1f}s  eta {el*(len(data)-i-1)/(i+1):.0f}s", flush=True)

    n = len(results)
    R1 = sum(r["hit@1"] for r in results) / n
    R5 = sum(r["hit@5"] for r in results) / n
    R10 = sum(r["hit@10"] for r in results) / n
    report = {
        "experiment": "Graph-traversal substrate (IDF-weighted entity overlap) on LongMemEval-500",
        "method": "regex entity extraction (proper nouns, dates, num+unit, tech tokens, quoted) per-question IDF",
        "n": n,
        "R@1": round(R1, 4),
        "R@5": round(R5, 4),
        "R@10": round(R10, 4),
        "by_type": {t: {
            "n": b["n"],
            "R@1": round(b["h1"]/b["n"], 4),
            "R@5": round(b["h5"]/b["n"], 4),
            "R@10": round(b["h10"]/b["n"], 4),
        } for t, b in sorted(by_type.items())},
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps({"summary": report, "per_q": results}, indent=2))
    print("\n== RESULT ==")
    print(json.dumps(report, indent=2))
    print(f"\nsaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
