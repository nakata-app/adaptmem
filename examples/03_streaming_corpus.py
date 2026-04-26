"""Streaming index updates: add corpus chunks without retraining.

Use when new documents land in the corpus after the initial training run.
`add_corpus()` encodes only the new entries and appends them to the existing
embedding matrix — no re-encoding of the old corpus, no second `.train()`
call. De-duplicates by id, so calling it repeatedly with overlapping
batches is safe.

  python examples/03_streaming_corpus.py
"""
from __future__ import annotations

from adaptmem import AdaptMem


def main():
    initial_corpus = [
        {"id": "doc1", "text": "Rust ownership rules prevent data races at compile time."},
        {"id": "doc2", "text": "Borrow checker enforces lifetimes statically."},
        {"id": "doc3", "text": "Async Rust lets you write futures without runtime overhead."},
    ]
    labelled = [
        {"query": "rust memory safety", "relevant_ids": ["doc1", "doc2"]},
        {"query": "rust async runtime",  "relevant_ids": ["doc3"]},
    ]

    am = AdaptMem(base_model="all-MiniLM-L6-v2", device="cpu")
    am.train(corpus=initial_corpus, labelled=labelled)
    print(f"after train:  corpus={len(am._corpus)}  embeddings={am._embeddings.shape}")

    # New documents arrive over time. add_corpus encodes only the unseen ids.
    new_arrivals = [
        {"id": "doc4", "text": "Tokio is the most-used async runtime for Rust applications."},
        {"id": "doc5", "text": "Channels move data between async tasks safely."},
        {"id": "doc1", "text": "(duplicate id — will be skipped, original preserved)"},
    ]
    n_added = am.add_corpus(new_arrivals)
    print(f"after stream: corpus={len(am._corpus)}  embeddings={am._embeddings.shape}  added={n_added}")

    hits = am.search("async task communication", top_k=3)
    for h in hits:
        print(f"  {h.score:.3f}  {h.chunk_id}  {h.text[:60]}")


if __name__ == "__main__":
    main()
