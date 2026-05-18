#!/usr/bin/env python3
"""chunk_strategy_ablation_ft300.py — A/B/C ablation, FT-300 encoder swap.

Derivative of jphein-mempalace/scripts/chunk_strategy_ablation.py.
Only difference: monkey-patches mempalace.embedding.get_embedding_function
to return a SentenceTransformer-backed EF wrapping FT-300 weights
(adaptmem nakata-app/minilm-lme-ft-300), instead of the default
ONNX MiniLM. EF.name() == "default" so chromadb's persisted EF identity
matches the baseline path on read.

Tests jpheinden's #1384 hypothesis: chunking-axis sensitivity is downstream
of encoder calibration. If the hypothesis holds:
  - B-vs-A markdown lift (+0.027 MRR on baseline) should compress.
  - C-vs-A code disadvantage may stay (FT-300 conversational, not code).

Run:
    .venv/bin/python /tmp/chunk_strategy_ablation_ft300.py \\
        --corpus /tmp/chunk_ablation_corpus_curated \\
        --chunk-sizes 800 \\
        --out /tmp/chunk_strategy_ablation_ft300_result.json
"""
from __future__ import annotations

import sys
from pathlib import Path

_FORK = Path("/Users/macmini/Projects/jphein-mempalace")
sys.path.insert(0, str(_FORK / "scripts"))
sys.path.insert(0, str(_FORK))

FT300_PATH = "/Users/macmini/Projects/metis-pair/benchmarks/models/minilm-lme-ft-300"


def _install_ft300_ef() -> None:
    """Monkey-patch mempalace.embedding.get_embedding_function before any
    chromadb collection is opened. Must run before chunk_strategy_ablation
    imports mempalace.miner / mempalace.palace.
    """
    from chromadb.api.types import EmbeddingFunction
    from sentence_transformers import SentenceTransformer

    class FT300EmbeddingFunction(EmbeddingFunction):
        def __init__(self, model_path: str) -> None:
            print(f"[ft300] loading SentenceTransformer from {model_path}", flush=True)
            self._model = SentenceTransformer(model_path, device="cpu")
            print("[ft300] model loaded", flush=True)

        def __call__(self, input):
            if isinstance(input, str):
                input = [input]
            embs = self._model.encode(
                list(input),
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embs.tolist()

        @staticmethod
        def name() -> str:
            return "default"

    ef = FT300EmbeddingFunction(FT300_PATH)

    import mempalace.embedding as _emb
    _emb.get_embedding_function = lambda device=None: ef
    _emb._EF_CACHE.clear()

    import mempalace.backends.chroma as _chroma
    _orig = _chroma.ChromaBackend._resolve_embedding_function
    _chroma.ChromaBackend._resolve_embedding_function = staticmethod(lambda: ef)
    print("[ft300] monkey-patch installed: get_embedding_function and ChromaBackend._resolve_embedding_function", flush=True)


_install_ft300_ef()

import chunk_strategy_ablation as _base

if __name__ == "__main__":
    sys.exit(_base.main())
