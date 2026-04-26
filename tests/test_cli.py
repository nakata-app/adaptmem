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
