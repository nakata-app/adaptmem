"""Colab training script for FT-Code checkpoints on CodeSearchNet/python.

Drop this into a Colab cell (T4 runtime), set MOUNT_DRIVE=True to persist
checkpoints, run. Produces FT-Code-300, FT-Code-1000, FT-Code-5000 by default.

Each FT-Code-N is one AdaptMem.train() pass with N labelled (docstring, body)
pairs from CodeSearchNet python train split, dedup'd on body. Same MNR loss
recipe as FT-300 — only the adaptation signal (code vs conversation) changes.

Local-run note: works on CPU too, slower. Smoke test on Mac (n=1000) took
~3 minutes on CPU end-to-end. T4 should be ~10x faster.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from datasets import load_dataset

from adaptmem.core import AdaptMem
from adaptmem.miner import CorpusEntry
from adaptmem.types import LabelledQuery, TrainConfig


MOUNT_DRIVE = False  # set True in Colab to persist to /content/drive/MyDrive/adaptmem-bench/ft-code/
DRIVE_OUT = "/content/drive/MyDrive/adaptmem-bench/ft-code"
LOCAL_OUT = "./checkpoints/ft-code"
TRAIN_SIZES = [300, 1000, 5000]
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def mount_drive_if_requested() -> str:
    if MOUNT_DRIVE:
        from google.colab import drive  # noqa: I001 — only available in Colab
        drive.mount("/content/drive")
        Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)
        return DRIVE_OUT
    Path(LOCAL_OUT).mkdir(parents=True, exist_ok=True)
    return LOCAL_OUT


def load_train_pairs(n: int) -> tuple[list[CorpusEntry], list[LabelledQuery]]:
    """Pull N (docstring → body) pairs from CodeSearchNet python train split.

    Dedup on first 200 chars of body to suppress near-duplicates from copy-paste
    code (helpers, common patterns). Drop rows with docstring < 10 chars or body
    < 40 chars (too noisy to be a useful retrieval signal).
    """
    ds = load_dataset("code_search_net", "python", split="train")
    corpus: list[CorpusEntry] = []
    labelled: list[LabelledQuery] = []
    seen: set[str] = set()
    for i, row in enumerate(ds):
        if len(corpus) >= n:
            break
        body = row.get("func_code_string") or ""
        doc = (row.get("func_documentation_string") or "").strip()
        if len(body) < 40 or len(doc) < 10:
            continue
        key = body[:200]
        if key in seen:
            continue
        seen.add(key)
        cid = f"cs{i}"
        corpus.append(CorpusEntry(id=cid, text=body))
        labelled.append(LabelledQuery(query=doc.splitlines()[0][:200], relevant_ids=[cid]))
    return corpus, labelled


def train_one(n: int, out_dir: Path, device: str | None) -> dict:
    print(f"\n=== FT-Code-{n} ===")
    t0 = time.time()
    corpus, labelled = load_train_pairs(n)
    t_data = time.time() - t0
    print(f"  data: {len(corpus)} pairs in {t_data:.1f}s")

    am = AdaptMem(base_model=BASE_MODEL, device=device)
    cfg = TrainConfig(epochs=1, batch_size=8, learning_rate=2e-5, warmup_ratio=0.1, top_k_mine=10)
    t1 = time.time()
    stats = am.train(corpus, labelled, config=cfg)
    t_train = time.time() - t1
    print(f"  train: {stats}, {t_train:.1f}s")

    ckpt_dir = out_dir / f"ft-code-{n}"
    am.save(ckpt_dir)
    print(f"  saved → {ckpt_dir}")

    return {
        "n": n,
        "n_pairs": len(corpus),
        "data_load_s": t_data,
        "train_runtime_s": t_train,
        "train_stats": stats,
        "checkpoint": str(ckpt_dir),
    }


def main() -> None:
    device = os.environ.get("ADAPTMEM_DEVICE")  # "cuda" on T4, leave unset locally for auto
    out_dir = Path(mount_drive_if_requested())
    results = [train_one(n, out_dir, device) for n in TRAIN_SIZES]
    summary_path = out_dir / "training_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nALL DONE → {summary_path}")


if __name__ == "__main__":
    main()
