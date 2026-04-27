"""CLI smoke tests for adaptmem.

Subprocess-based; only argparse plumbing is exercised, so no
sentence-transformers download is triggered.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CLI = [sys.executable, "-m", "adaptmem.cli"]


def _run(args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        CLI + args,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        **kwargs,
    )


def test_top_help_lists_subcommands():
    out = _run(["--help"])
    assert out.returncode == 0
    for sub in ["train", "search", "evaluate"]:
        assert sub in out.stdout, f"missing subcommand in help: {sub}"


def test_train_help_lists_rerank_flags():
    out = _run(["train", "--help"])
    assert out.returncode == 0
    for flag in ["--corpus", "--queries", "--out", "--rerank", "--rerank-model"]:
        assert flag in out.stdout, f"missing train flag: {flag}"


def test_search_help_lists_rerank_top_k():
    out = _run(["search", "--help"])
    assert out.returncode == 0
    assert "--rerank-top-k" in out.stdout
    assert "--rerank-model" in out.stdout


def test_evaluate_help_lists_required_args():
    out = _run(["evaluate", "--help"])
    assert out.returncode == 0
    assert "--model" in out.stdout
    assert "--queries" in out.stdout
    assert "--top-k" in out.stdout


def test_no_subcommand_errors():
    out = _run([])
    assert out.returncode != 0
    # argparse prints "the following arguments are required: cmd" or similar
    assert "required" in out.stderr.lower() or "argument" in out.stderr.lower()


def test_search_missing_required_args_errors():
    out = _run(["search"])
    assert out.returncode != 0
    assert "model" in out.stderr.lower() or "query" in out.stderr.lower()


# ---- evaluate command logic (no model fetch) ----------------------------


def test_evaluate_perfect_recall(tmp_path, monkeypatch):
    """When the stubbed model returns the relevant id at rank 0, R@1/5/10 = 1.0."""
    import argparse
    import io
    import json as _json
    from contextlib import redirect_stdout

    queries = [
        {"query": "q1", "relevant_ids": ["a"]},
        {"query": "q2", "relevant_ids": ["b"]},
    ]
    qf = tmp_path / "queries.json"
    qf.write_text(_json.dumps(queries))

    from adaptmem.types import RetrievalHit

    class _StubAM:
        rerank_enabled = False
        rerank_model_name = None
        _rerank_model = None

        def search(self, query, top_k):
            # Always return the "right" id at rank 0
            target = "a" if query == "q1" else "b"
            return [RetrievalHit(chunk_id=target, text="match", score=1.0)]

    monkeypatch.setattr("adaptmem.AdaptMem.load", classmethod(lambda cls, p: _StubAM()))

    from adaptmem.cli import _cmd_evaluate
    args = argparse.Namespace(
        model="x", queries=str(qf), top_k=10, rerank=False, rerank_model=None
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        _cmd_evaluate(args)
    out = _json.loads(buf.getvalue())
    assert out["n"] == 2
    assert out["recall"]["@1"] == 1.0
    assert out["recall"]["@5"] == 1.0
    assert out["recall"]["@10"] == 1.0


def test_evaluate_zero_recall(tmp_path, monkeypatch):
    """When the stubbed model never returns the relevant id, R@k = 0."""
    import argparse
    import io
    import json as _json
    from contextlib import redirect_stdout

    queries = [{"query": "q", "relevant_ids": ["target"]}]
    qf = tmp_path / "queries.json"
    qf.write_text(_json.dumps(queries))

    from adaptmem.types import RetrievalHit

    class _StubAM:
        rerank_enabled = False
        rerank_model_name = None
        _rerank_model = None

        def search(self, query, top_k):
            return [
                RetrievalHit(chunk_id=f"wrong{i}", text="x", score=0.5) for i in range(top_k)
            ]

    monkeypatch.setattr("adaptmem.AdaptMem.load", classmethod(lambda cls, p: _StubAM()))

    from adaptmem.cli import _cmd_evaluate
    args = argparse.Namespace(
        model="x", queries=str(qf), top_k=10, rerank=False, rerank_model=None
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        _cmd_evaluate(args)
    out = _json.loads(buf.getvalue())
    assert out["n"] == 1
    assert out["recall"]["@1"] == 0.0
    assert out["recall"]["@5"] == 0.0


def test_evaluate_skips_queries_with_empty_relevant_ids(tmp_path, monkeypatch):
    """Queries with no relevant_ids must be skipped (cannot score recall)."""
    import argparse
    import io
    import json as _json
    from contextlib import redirect_stdout

    queries = [
        {"query": "q1", "relevant_ids": []},  # skipped
        {"query": "q2", "relevant_ids": ["a"]},
    ]
    qf = tmp_path / "queries.json"
    qf.write_text(_json.dumps(queries))

    from adaptmem.types import RetrievalHit

    class _StubAM:
        rerank_enabled = False
        rerank_model_name = None
        _rerank_model = None

        def search(self, query, top_k):
            return [RetrievalHit(chunk_id="a", text="match", score=1.0)]

    monkeypatch.setattr("adaptmem.AdaptMem.load", classmethod(lambda cls, p: _StubAM()))

    from adaptmem.cli import _cmd_evaluate
    args = argparse.Namespace(
        model="x", queries=str(qf), top_k=5, rerank=False, rerank_model=None
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        _cmd_evaluate(args)
    out = _json.loads(buf.getvalue())
    # Only 1 of the 2 queries had relevant_ids
    assert out["n"] == 1
    assert out["recall"]["@1"] == 1.0
