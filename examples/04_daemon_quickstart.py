"""Daemon quickstart: start `adaptmem serve`, talk to it over HTTP.

Pattern for cross-language consumers (e.g. metis, a Rust agent CLI) or
multi-process Python deployments where you want one model load shared
across many callers.

Two terminals:

  Terminal 1 — daemon:
    pip install "adaptmem[server]"
    adaptmem serve --port 7800 --base-model all-MiniLM-L6-v2

  Terminal 2 — this script:
    pip install requests
    python examples/04_daemon_quickstart.py
"""
from __future__ import annotations

import sys

import requests


DAEMON = "http://127.0.0.1:7800"


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    print("Is `adaptmem serve` running?", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    # 1. Health-check
    try:
        r = requests.get(f"{DAEMON}/healthz", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        fail(f"daemon /healthz unreachable: {e}")
    print(f"daemon ok — uptime {r.json()['uptime_s']:.1f}s")

    # 2. Reindex a small corpus under the `demo` corpus_id.
    docs = [
        {"id": "a", "text": "PostgreSQL ships native JSON since 9.4."},
        {"id": "b", "text": "Redis stores JSON via the RedisJSON module."},
        {"id": "c", "text": "MongoDB stores documents in BSON."},
        {"id": "d", "text": "SQLite has no native JSON column; use the JSON1 extension."},
    ]
    r = requests.post(
        f"{DAEMON}/reindex",
        json={"corpus_id": "demo", "documents": docs},
        timeout=120,  # first reindex pays the model load cost
    )
    if not r.ok:
        fail(f"/reindex {r.status_code}: {r.text}")
    body = r.json()
    print(f"reindexed {body['n_docs']} docs in {body['elapsed_ms']}ms")

    # 3. Issue a few queries.
    queries = [
        "Which databases have native JSON?",
        "Where is JSON stored as TEXT?",
        "What format does MongoDB use?",
    ]
    for q in queries:
        r = requests.post(
            f"{DAEMON}/search",
            json={"query": q, "top_k": 3, "corpus_id": "demo"},
            timeout=10,
        )
        r.raise_for_status()
        body = r.json()
        print(f"\nQ: {q!r}  ({body['elapsed_ms']:.0f}ms)")
        for hit in body["hits"]:
            print(f"  {hit['score']:.3f}  [{hit['id']}]  {hit['text']}")


if __name__ == "__main__":
    main()
