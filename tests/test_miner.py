"""Hard-negative miner tests with FakeEncoder (no model download)."""
from __future__ import annotations

import re

import numpy as np

from adaptmem.miner import CorpusEntry, HardNegativeMiner
from adaptmem.types import LabelledQuery


class FakeEncoder:
    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(64, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w not in self.vocab and len(self.vocab) < 64:
                self.vocab[w] = len(self.vocab)
            if w in self.vocab:
                v[self.vocab[w]] += 1.0
        return v

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
        arr = np.stack([self._vec(t) for t in texts])
        if normalize_embeddings:
            n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            arr = arr / n
        return arr


def test_miner_picks_hard_negative():
    corpus = [
        CorpusEntry(id="m1", text="postgres index btree gin"),
        CorpusEntry(id="m2", text="apple banana fruit basket"),
        CorpusEntry(id="m3", text="postgres jsonb gin index"),
    ]
    queries = [LabelledQuery(query="postgres index", relevant_ids=["m1"])]
    miner = HardNegativeMiner(base_model=FakeEncoder(), top_k_mine=3)
    pairs = miner.mine(corpus, queries)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.anchor == "postgres index"
    assert p.pos_id == "m1"
    # m3 is the hardest negative (shares words "postgres", "index"), m2 is irrelevant
    assert p.neg_id == "m3"


def test_miner_multi_relevant_emits_one_per_positive():
    corpus = [
        CorpusEntry(id=f"m{i}", text=f"doc {i} postgres index") for i in range(5)
    ]
    queries = [LabelledQuery(query="postgres index", relevant_ids=["m0", "m1"])]
    miner = HardNegativeMiner(base_model=FakeEncoder(), top_k_mine=5)
    pairs = miner.mine(corpus, queries)
    assert len(pairs) == 2
    pos_ids = sorted(p.pos_id for p in pairs)
    assert pos_ids == ["m0", "m1"]


def test_miner_empty_corpus():
    miner = HardNegativeMiner(base_model=FakeEncoder())
    assert miner.mine([], [LabelledQuery(query="x", relevant_ids=[])]) == []


def test_miner_random_fallback_when_all_topk_relevant():
    # All top-K are gold → must fall back to random non-rel
    corpus = [
        CorpusEntry(id="m1", text="postgres index btree"),
        CorpusEntry(id="m2", text="apples fruit basket"),
    ]
    queries = [LabelledQuery(query="postgres index btree", relevant_ids=["m1"])]
    miner = HardNegativeMiner(base_model=FakeEncoder(), top_k_mine=1)
    pairs = miner.mine(corpus, queries)
    # Top-1 is m1 (which IS relevant), so neg falls back to random non-rel = m2
    assert len(pairs) == 1
    assert pairs[0].neg_id == "m2"
