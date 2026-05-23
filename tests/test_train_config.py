"""Integration tests for new TrainConfig options (no GPU, uses FakeEncoder)."""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import numpy as np

from adaptmem.miner import CorpusEntry, HardNegativeMiner
from adaptmem.types import LabelledQuery, TrainConfig


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


def _make_corpus():
    return [
        CorpusEntry(id="m1", text="postgres index btree gin"),
        CorpusEntry(id="m2", text="apple banana fruit basket"),
        CorpusEntry(id="m3", text="postgres jsonb gin index"),
        CorpusEntry(id="m4", text="mysql query planner optimizer"),
        CorpusEntry(id="m5", text="redis cache eviction policy"),
    ]


def _make_queries():
    return [LabelledQuery(query="postgres index optimization", relevant_ids=["m1"])]


def test_n_negatives_config_wired_to_miner():
    cfg = TrainConfig(n_negatives=3)
    miner = HardNegativeMiner(
        base_model=FakeEncoder(),
        top_k_mine=cfg.top_k_mine,
        n_negatives=cfg.n_negatives,
    )
    pairs = miner.mine(_make_corpus(), _make_queries())
    assert len(pairs) == 3
    neg_ids = {p.neg_id for p in pairs}
    assert "m1" not in neg_ids


def test_gradient_accumulation_steps_defaults_to_1():
    cfg = TrainConfig()
    assert cfg.gradient_accumulation_steps == 1


def test_gradient_accumulation_steps_custom():
    cfg = TrainConfig(gradient_accumulation_steps=16)
    assert cfg.gradient_accumulation_steps == 16


def test_loss_type_default_is_mnrl():
    cfg = TrainConfig()
    assert cfg.loss_type == "mnrl"


def test_loss_type_cached_mnrl():
    cfg = TrainConfig(loss_type="cached_mnrl")
    assert cfg.loss_type == "cached_mnrl"


def test_n_negatives_default_backward_compat():
    cfg = TrainConfig()
    assert cfg.n_negatives == 1
    miner = HardNegativeMiner(
        base_model=FakeEncoder(),
        top_k_mine=cfg.top_k_mine,
        n_negatives=cfg.n_negatives,
    )
    pairs = miner.mine(_make_corpus(), _make_queries())
    assert len(pairs) == 1


def test_full_config_round_trip():
    cfg = TrainConfig(
        epochs=5,
        batch_size=2,
        n_negatives=3,
        gradient_accumulation_steps=16,
        loss_type="cached_mnrl",
        learning_rate=3e-5,
    )
    assert cfg.epochs == 5
    assert cfg.batch_size == 2
    assert cfg.n_negatives == 3
    assert cfg.gradient_accumulation_steps == 16
    assert cfg.loss_type == "cached_mnrl"
    assert cfg.learning_rate == 3e-5
