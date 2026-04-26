"""Cross-encoder rerank tests for AdaptMem.search() — model-free.

We stub the bi-encoder embeddings + corpus directly (no SentenceTransformer
download) and stub the CrossEncoder so the test is fast and offline.
"""
from __future__ import annotations

import numpy as np

from adaptmem import AdaptMem
from adaptmem.miner import CorpusEntry


class _StubBiEncoder:
    """Returns a unit vector pointing at the supplied query string."""

    def __init__(self, query_to_vec: dict[str, np.ndarray]):
        self.q2v = query_to_vec

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        out = np.stack([self.q2v[t] for t in texts])
        if normalize_embeddings:
            n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
            out = out / n
        return out


class _StubCrossEncoder:
    """Returns a fixed vector of scores aligned with the input pairs."""

    def __init__(self, scores_for_text: dict[str, float]):
        self.scores_for_text = scores_for_text
        self.last_pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs, show_progress_bar=False):
        self.last_pairs = list(pairs)
        return np.array([self.scores_for_text[p[1]] for p in pairs])


def _build_am(rerank: bool = False) -> AdaptMem:
    am = AdaptMem(rerank=rerank)
    # Inject corpus + embeddings directly (skip train()).
    corpus = [
        CorpusEntry(id="c0", text="alpha"),
        CorpusEntry(id="c1", text="bravo"),
        CorpusEntry(id="c2", text="charlie"),
        CorpusEntry(id="c3", text="delta"),
    ]
    # Bi-encoder dot-product order for query "q":
    #   c0 highest, c1 next, c2 third, c3 last
    embs = np.array(
        [
            [1.00, 0.00],  # c0
            [0.95, 0.05],  # c1
            [0.50, 0.10],  # c2
            [0.10, 0.90],  # c3
        ],
        dtype=np.float32,
    )
    am._corpus = corpus
    am._embeddings = embs
    am._model = _StubBiEncoder({"q": np.array([1.0, 0.0], dtype=np.float32)})
    return am


def test_search_default_returns_bi_encoder_order():
    am = _build_am(rerank=False)
    hits = am.search("q", top_k=2)
    assert [h.chunk_id for h in hits] == ["c0", "c1"]


def test_search_rerank_reorders_by_cross_encoder():
    # Cross-encoder strongly prefers c2 and c1, demotes c0
    am = _build_am(rerank=True)
    am._rerank_model = _StubCrossEncoder(
        {"alpha": 0.10, "bravo": 0.80, "charlie": 0.95, "delta": -0.50}
    )
    hits = am.search("q", top_k=2)
    # CE expected order: c2 (0.95), c1 (0.80), c0 (0.10), c3 (-0.50)
    assert [h.chunk_id for h in hits] == ["c2", "c1"]
    # Score returned is the CE score, not the bi-encoder cosine
    assert abs(hits[0].score - 0.95) < 1e-6


def test_rerank_widens_candidate_set_default():
    # top_k=2 with rerank → fetches top_k * 3 = 6 (clamped to 4 corpus size)
    am = _build_am(rerank=True)
    am._rerank_model = _StubCrossEncoder(
        {"alpha": 0.0, "bravo": 0.0, "charlie": 0.0, "delta": 1.0}
    )
    am.search("q", top_k=2)
    # All 4 corpus items must have been scored by CE (delta only reachable via
    # widened candidate set — bi-encoder ranks it last)
    pairs_text = [p[1] for p in am._rerank_model.last_pairs]
    assert "delta" in pairs_text


def test_rerank_explicit_rerank_top_k():
    am = _build_am(rerank=True)
    am._rerank_model = _StubCrossEncoder(
        {"alpha": 1.0, "bravo": 0.5, "charlie": 0.0, "delta": -1.0}
    )
    # rerank_top_k=2 → CE only sees the top 2 bi-encoder candidates (c0, c1)
    am.search("q", top_k=1, rerank_top_k=2)
    pairs_text = [p[1] for p in am._rerank_model.last_pairs]
    assert set(pairs_text) == {"alpha", "bravo"}
