"""Minimal end-to-end: tune a tiny corpus, save, reload, search.

Runs in ~30 seconds on CPU with the default `all-MiniLM-L6-v2`. Useful as a
"does my install work?" smoke test.

  pip install -e .[dev]   # from repo root
  python examples/01_basic_usage.py
"""
from __future__ import annotations

from pathlib import Path

from adaptmem import AdaptMem


def main():
    # A toy domain — enough to make hard-negative mining meaningful.
    corpus = [
        {"id": "p1", "text": "PostgreSQL is great for transactional workloads with strong JSON support."},
        {"id": "p2", "text": "MongoDB stores documents as BSON and excels at flexible schemas."},
        {"id": "p3", "text": "Redis is an in-memory key-value store often used as a cache."},
        {"id": "p4", "text": "ElasticSearch is a distributed search engine built on top of Lucene."},
        {"id": "p5", "text": "Postgres B-tree indexes are the default; GIN indexes power JSONB queries."},
        {"id": "p6", "text": "Mongo aggregation pipelines compose match, group, sort, and project stages."},
    ]
    # Labelled queries — the supervision signal. Each query points at the
    # corpus ids that should be retrievable for it.
    labelled = [
        {"query": "transactional database with JSON", "relevant_ids": ["p1", "p5"]},
        {"query": "document store with flexible schema", "relevant_ids": ["p2", "p6"]},
        {"query": "in-memory cache layer",              "relevant_ids": ["p3"]},
        {"query": "full-text search engine",             "relevant_ids": ["p4"]},
    ]

    # Force CPU on Apple silicon to dodge the MPS deadlock seen during
    # contrastive fine-tunes. Drop `device="cpu"` on Linux/CUDA.
    am = AdaptMem(base_model="all-MiniLM-L6-v2", device="cpu")
    stats = am.train(corpus=corpus, labelled=labelled)
    print("train stats:", stats)

    out = Path("./tmp-encoder")
    am.save(out)
    print(f"saved to {out}/")

    # Reload from disk to prove persistence works
    am2 = AdaptMem.load(out)
    hits = am2.search("Postgres JSON", top_k=3)
    for h in hits:
        print(f"  {h.score:.3f}  {h.chunk_id}  {h.text[:60]}")


if __name__ == "__main__":
    main()
