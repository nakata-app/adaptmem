#!/usr/bin/env python3
"""Run mempal's longmemeval_bench.py with a sentence-transformers fine-tuned
encoder, by monkey-patching mempal's `_bench_embed_fn` global. NO modification
to mempal's bench script required — we use the encoder-substitution hook they
already designed in.

This is the "matched-protocol" path: same retrieval pipeline (chromadb, palace
build, ranking, scoring) — only the encoder is swapped.

Usage:
    python mempal_bench_with_ft.py \\
        --bench-script /path/to/mempalace_repo/longmemeval_bench.py \\
        --data-file    /path/to/longmemeval_s_cleaned.json \\
        --ft-model     /path/to/minilm-lme-ft-300/model \\
        --mode         raw \\
        --out          results_mempal_raw_ft300.jsonl

If --ft-model is omitted, falls through to mempal's default ChromaDB MiniLM
(useful for sanity reproducing the published raw 0.966).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def load_bench_module(bench_script_path: Path):
    """Load mempal's bench script as a module so we can patch its globals."""
    spec = importlib.util.spec_from_file_location("longmemeval_bench", bench_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load bench script at {bench_script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["longmemeval_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_st_embed_fn(model_path: str):
    """ChromaDB-compatible EmbeddingFunction backed by sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class _STEmbedFn(EmbeddingFunction):
        def __init__(self, path: str):
            print(f"  Loading sentence-transformers model: {path}", flush=True)
            self._model = SentenceTransformer(path)
            print("  Model ready.", flush=True)

        def __call__(self, input: Documents) -> Embeddings:
            vecs = self._model.encode(list(input), show_progress_bar=False, convert_to_numpy=True)
            return [v.tolist() for v in vecs]

        def name(self) -> str:
            return f"st:{Path(model_path).name}"

    return _STEmbedFn(model_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-script", required=True, help="Path to mempal's longmemeval_bench.py")
    p.add_argument("--data-file", required=True, help="Path to longmemeval_s_cleaned.json")
    p.add_argument("--ft-model", default=None, help="Path to FT sentence-transformers model dir (omit = mempal default)")
    p.add_argument("--mode", default="raw", choices=["raw", "aaak", "rooms", "hybrid", "hybrid_v2", "hybrid_v3", "hybrid_v4", "palace", "diary", "full"])
    p.add_argument("--granularity", default="session", choices=["session", "turn"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--hybrid-weight", type=float, default=0.30)
    p.add_argument("--out", required=True)
    p.add_argument("--split-file", default=None)
    p.add_argument("--split-subset", default=None, choices=[None, "dev", "held_out"])
    args = p.parse_args()

    bench = load_bench_module(Path(args.bench_script).resolve())

    if args.ft_model:
        embed_fn = make_st_embed_fn(args.ft_model)
        bench._bench_embed_fn = embed_fn  # monkey-patch the global
        print(f"  [matched-protocol] Substituted bench encoder → {args.ft_model}", flush=True)
    else:
        print("  [sanity] Using mempal's ChromaDB default encoder.", flush=True)

    bench.run_benchmark(
        data_file=args.data_file,
        granularity=args.granularity,
        limit=args.limit,
        out_file=args.out,
        mode=args.mode,
        skip=args.skip,
        hybrid_weight=args.hybrid_weight,
        split_file=args.split_file,
        split_subset=args.split_subset,
    )


if __name__ == "__main__":
    main()
