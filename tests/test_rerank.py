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


# ---- add_corpus streaming -------------------------------------------------


def _build_am_with_encoder(rerank: bool = False) -> AdaptMem:
    """Same as _build_am but with a stub encoder that grows its vocab on demand."""
    am = AdaptMem(rerank=rerank)
    am._corpus = [CorpusEntry(id="c0", text="alpha")]
    am._embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    # A stub encoder that returns a deterministic 2-d vector per text.
    class _Enc:
        vecs = {
            "alpha": np.array([1.0, 0.0]),
            "bravo": np.array([0.0, 1.0]),
            "charlie": np.array([0.7, 0.7]),
        }

        def encode(self, texts, batch_size=64, show_progress_bar=False,
                   normalize_embeddings=True, convert_to_numpy=True):
            arr = np.stack([self.vecs[t] for t in texts]).astype(np.float32)
            if normalize_embeddings:
                arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
            return arr

    am._model = _Enc()
    return am


def test_add_corpus_appends_new_entries_only():
    am = _build_am_with_encoder()
    n = am.add_corpus([{"id": "c1", "text": "bravo"}, {"id": "c2", "text": "charlie"}])
    assert n == 2
    assert [c.id for c in am._corpus] == ["c0", "c1", "c2"]
    assert am._embeddings.shape == (3, 2)


def test_add_corpus_skips_duplicates():
    am = _build_am_with_encoder()
    n1 = am.add_corpus([{"id": "c1", "text": "bravo"}])
    n2 = am.add_corpus([{"id": "c1", "text": "bravo"}, {"id": "c2", "text": "charlie"}])
    assert n1 == 1
    assert n2 == 1  # c1 skipped, c2 added
    assert [c.id for c in am._corpus] == ["c0", "c1", "c2"]


def test_add_corpus_empty_input_is_zero():
    am = _build_am_with_encoder()
    assert am.add_corpus([]) == 0
    assert len(am._corpus) == 1


def test_add_corpus_requires_init():
    am = AdaptMem()
    import pytest
    with pytest.raises(RuntimeError, match="Not initialised"):
        am.add_corpus([{"id": "c0", "text": "alpha"}])


# ---- public rerank() helper (no bi-encoder index required) ----------------


def test_rerank_method_with_strings():
    am = AdaptMem()
    am._rerank_model = _StubCrossEncoder(
        {"alpha": 0.10, "bravo": 0.80, "charlie": 0.95, "delta": -0.50}
    )
    ranked = am.rerank("q", ["alpha", "bravo", "charlie", "delta"])
    # Expected order by CE score: charlie(2)=0.95, bravo(1)=0.80, alpha(0)=0.10, delta(3)=-0.50
    assert [i for i, _ in ranked] == [2, 1, 0, 3]
    assert abs(ranked[0][1] - 0.95) < 1e-6


def test_rerank_method_with_tuples():
    am = AdaptMem()
    am._rerank_model = _StubCrossEncoder({"alpha": 0.5, "bravo": 0.9})
    ranked = am.rerank("q", [("id-a", "alpha"), ("id-b", "bravo")], top_k=1)
    assert len(ranked) == 1
    assert ranked[0][0] == 1  # bravo
    assert abs(ranked[0][1] - 0.9) < 1e-6


def test_rerank_method_with_dicts():
    am = AdaptMem()
    am._rerank_model = _StubCrossEncoder({"alpha": 0.1, "bravo": 0.9})
    ranked = am.rerank("q", [{"text": "alpha"}, {"text": "bravo"}])
    assert [i for i, _ in ranked] == [1, 0]


def test_rerank_empty_candidates_returns_empty():
    am = AdaptMem()
    assert am.rerank("q", []) == []


def test_rerank_top_k_truncates():
    am = AdaptMem()
    am._rerank_model = _StubCrossEncoder(
        {"a": 0.9, "b": 0.5, "c": 0.1}
    )
    ranked = am.rerank("q", ["a", "b", "c"], top_k=2)
    assert len(ranked) == 2
    assert [i for i, _ in ranked] == [0, 1]


def test_rerank_works_without_bi_encoder_index():
    """The whole point: mnemonics has its own index, only needs CE here."""
    am = AdaptMem()
    # No .train(), no .load() — _model/_embeddings stay None.
    am._rerank_model = _StubCrossEncoder({"x": 1.0, "y": 0.0})
    ranked = am.rerank("q", ["x", "y"])
    assert ranked[0][0] == 0
    assert am._model is None  # confirm no bi-encoder needed
