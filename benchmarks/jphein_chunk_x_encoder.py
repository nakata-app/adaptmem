"""C-alpha: Run jphein's chunk_strategy_ablation.py once with mempal's default
encoder and once with adaptmem FT-300 swapped in. Each run is its own
Python process to avoid cache / import-state contamination between runs.

The FT-300 swap monkey-patches mempalace.embedding.get_embedding_function
inside the child process before importing the ablation script.

Hypothesis: encoder lift (FT-300, trained on LongMemEval QA pairs) is
LongMemEval-domain-specific and won't compose with chunking-axis changes
on mempalace's own .py corpus. Whatever shows up — flat / negative / lift —
goes into the report verbatim. Domain mismatch IS a finding."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

JPHEIN_REPO = Path("/Users/macmini/Projects/jphein-mempalace")
ABLATION = JPHEIN_REPO / "scripts" / "chunk_strategy_ablation.py"


CHILD_TEMPLATE = textwrap.dedent("""
    import sys, importlib.util
    from pathlib import Path

    sys.path.insert(0, {jphein_repo!r})
    sys.path.insert(0, str(Path({jphein_repo!r}) / "scripts"))

    if {ft_model_path!r}:
        import mempalace.embedding as _emb
        from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
        from sentence_transformers import SentenceTransformer

        class _FTEmbedFn(EmbeddingFunction):
            def __init__(self, path):
                self._model = SentenceTransformer(path, device="cpu")

            def __call__(self, input):
                vecs = self._model.encode(
                    list(input),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return vecs.tolist()

            @staticmethod
            def name():
                return "default"

        _ft = _FTEmbedFn({ft_model_path!r})
        _emb.get_embedding_function = lambda device=None: _ft
        try:
            _emb._EF_CACHE.clear()
        except Exception:
            pass
        print("[patch] encoder swapped to FT-300", flush=True)

    spec = importlib.util.spec_from_file_location("chunk_strategy_ablation", {ablation!r})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sys.argv = ["chunk_strategy_ablation.py", "--out", {out_file!r}]
    sys.exit(mod.main())
""").strip()


def run_one(label: str, out_dir: Path, ft_model_path: str | None, venv_python: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ablation_{label}.json"
    log_file = out_dir / f"ablation_{label}.log"
    enc_tag = "FT-300" if ft_model_path else "default"
    banner = f"{'=' * 60}\n  RUN: {label}  (encoder={enc_tag})\n{'=' * 60}"
    print(banner, flush=True)

    code = CHILD_TEMPLATE.format(
        jphein_repo=str(JPHEIN_REPO),
        ablation=str(ABLATION),
        ft_model_path=ft_model_path or "",
        out_file=str(out_file),
    )

    with open(log_file, "w") as logf:
        logf.write(banner + "\n")
        proc = subprocess.run(
            [venv_python, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logf.write(proc.stdout)
        logf.write(f"\n[exit] {proc.returncode}\n")
    if proc.returncode != 0:
        print(proc.stdout[-2000:], flush=True)
        raise SystemExit(f"child failed: {label}")
    print(f"  -> {out_file}  log={log_file}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-model", default="/Users/macmini/Projects/metis-pair/benchmarks/models/minilm-lme-ft-300")
    ap.add_argument("--out-dir", default="/Users/macmini/Projects/adaptmem/benchmarks/v335/chunk_x_encoder")
    ap.add_argument("--venv", default="/Users/macmini/Projects/metis-pair/benchmarks/.venv/bin/python")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    run_one("default_encoder", out_dir, ft_model_path=None, venv_python=args.venv)
    run_one("ft300_encoder", out_dir, ft_model_path=args.ft_model, venv_python=args.venv)
    print(f"\n[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
