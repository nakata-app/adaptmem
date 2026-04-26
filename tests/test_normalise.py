"""Normalisation helper tests (no model download)."""
from __future__ import annotations

from adaptmem.core import _normalise_corpus, _normalise_queries
from adaptmem.miner import CorpusEntry
from adaptmem.types import LabelledQuery


def test_corpus_strings_get_auto_ids():
    out = _normalise_corpus(["a", "b", "c"])
    assert [c.id for c in out] == ["c0", "c1", "c2"]
    assert [c.text for c in out] == ["a", "b", "c"]


def test_corpus_dicts_preserve_ids():
    out = _normalise_corpus([{"id": "x1", "text": "a"}, {"id": "x2", "text": "b"}])
    assert [c.id for c in out] == ["x1", "x2"]


def test_corpus_entries_passthrough():
    e = [CorpusEntry(id="m1", text="text")]
    out = _normalise_corpus(e)
    assert out == e


def test_queries_dict_to_dataclass():
    out = _normalise_queries([{"query": "q", "relevant_ids": ["a", "b"]}])
    assert len(out) == 1
    assert isinstance(out[0], LabelledQuery)
    assert out[0].query == "q"
    assert out[0].relevant_ids == ["a", "b"]


def test_queries_dataclass_passthrough():
    q = [LabelledQuery(query="x", relevant_ids=["y"])]
    assert _normalise_queries(q) == q
