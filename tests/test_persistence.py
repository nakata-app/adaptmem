"""CorpusStore round-trip tests — save / load / list."""
from __future__ import annotations

import numpy as np
import pytest

from adaptmem.persistence import CorpusStore


def test_save_load_round_trip(tmp_path):
    store = CorpusStore(tmp_path / "corpora.db")
    docs = [("a", "PostgreSQL has native JSON"), ("b", "Redis JSON via module")]
    embeddings = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
    )

    store.save_corpus("test", "all-MiniLM-L6-v2", docs, embeddings)

    loaded = store.load_corpus("test")
    assert loaded is not None
    docs_back, embs_back, model = loaded
    assert docs_back == docs
    np.testing.assert_array_almost_equal(embs_back, embeddings)
    assert model == "all-MiniLM-L6-v2"


def test_load_corpus_returns_none_for_missing(tmp_path):
    store = CorpusStore(tmp_path / "corpora.db")
    assert store.load_corpus("nope") is None


def test_list_corpora_returns_ids_in_order(tmp_path):
    store = CorpusStore(tmp_path / "corpora.db")
    embeddings = np.zeros((1, 3), dtype=np.float32)
    store.save_corpus("zeta", "m", [("d", "x")], embeddings)
    store.save_corpus("alpha", "m", [("d", "x")], embeddings)
    store.save_corpus("mu", "m", [("d", "x")], embeddings)

    assert store.list_corpora() == ["alpha", "mu", "zeta"]


def test_save_corpus_replaces_existing(tmp_path):
    """Re-saving the same corpus_id replaces the old rows."""
    store = CorpusStore(tmp_path / "corpora.db")
    embs1 = np.array([[1.0, 2.0]], dtype=np.float32)
    embs2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    store.save_corpus("c", "m", [("d1", "old text")], embs1)
    store.save_corpus("c", "m", [("d1", "new text 1"), ("d2", "new text 2")], embs2)

    loaded = store.load_corpus("c")
    assert loaded is not None
    docs, embs, _ = loaded
    assert len(docs) == 2
    assert docs[0] == ("d1", "new text 1")
    np.testing.assert_array_almost_equal(embs, embs2)


def test_save_rejects_shape_mismatch(tmp_path):
    store = CorpusStore(tmp_path / "corpora.db")
    embs = np.zeros((3, 4), dtype=np.float32)  # 3 rows
    with pytest.raises(ValueError, match="row counts must match"):
        store.save_corpus("c", "m", [("d", "x")], embs)  # 1 doc


def test_save_rejects_non_2d_embeddings(tmp_path):
    store = CorpusStore(tmp_path / "corpora.db")
    embs = np.zeros((4,), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2-D"):
        store.save_corpus("c", "m", [("d", "x")], embs)


def test_save_casts_non_float32(tmp_path):
    """float64 embeddings should be silently downcast."""
    store = CorpusStore(tmp_path / "corpora.db")
    embs64 = np.array([[1.0, 2.0]], dtype=np.float64)
    store.save_corpus("c", "m", [("d", "x")], embs64)

    loaded = store.load_corpus("c")
    assert loaded is not None
    _, embs_back, _ = loaded
    assert embs_back.dtype == np.float32
    np.testing.assert_array_almost_equal(embs_back, embs64.astype(np.float32))


def test_delete_corpus_removes_rows_and_returns_true(tmp_path):
    store = CorpusStore(tmp_path / "corpora.db")
    store.save_corpus(
        "victim", "m", [("d1", "hi"), ("d2", "ho")],
        np.zeros((2, 4), dtype=np.float32),
    )
    assert "victim" in store.list_corpora()
    assert store.delete_corpus("victim") is True
    assert "victim" not in store.list_corpora()
    # Cascaded: documents rows are gone too.
    assert store.load_corpus("victim") is None


def test_delete_corpus_returns_false_for_missing(tmp_path):
    store = CorpusStore(tmp_path / "corpora.db")
    assert store.delete_corpus("never-saved") is False


def test_persistence_survives_close_reopen(tmp_path):
    """The whole point: stop the process, reopen the DB, corpora are still there."""
    db = tmp_path / "corpora.db"
    embs = np.array([[0.1, 0.2]], dtype=np.float32)

    store1 = CorpusStore(db)
    store1.save_corpus("survives", "m", [("d", "hello")], embs)
    store1.close()

    store2 = CorpusStore(db)
    assert store2.list_corpora() == ["survives"]
    loaded = store2.load_corpus("survives")
    assert loaded is not None
    assert loaded[0] == [("d", "hello")]
