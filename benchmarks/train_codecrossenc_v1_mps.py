"""CodeCrossEnc-v1 training on Mac M2 MPS (no Colab, no Drive, no FUSE trap).

Base: cross-encoder/ms-marco-MiniLM-L-6-v2 (90MB)
Data: CodeSearchNet python train 100K positive + 2 random negative each = 300K
1 epoch, batch=32, lr=2e-5, warmup=500, max_length=384
Tahmini süre: ~2.5-3.5 saat M2 MPS, 8GB RAM (Cursor/browser kapalı önerilir)

Output: /Users/macmini/Projects/adaptmem/checkpoints/code-crossenc/v1/

Koşturma (background, Mac uyumasın):
  cd ~/Projects/adaptmem
  caffeinate -i nohup .venv/bin/python benchmarks/train_codecrossenc_v1_mps.py \
    > /tmp/codecrossenc_train.log 2>&1 &
  echo $!  # PID

İzleme:
  tail -f /tmp/codecrossenc_train.log
"""
from __future__ import annotations

import gc
import os
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

POS_LIMIT = 100_000
OUT_DIR = Path("/Users/macmini/Projects/adaptmem/checkpoints/code-crossenc/v1")
SEED = 42


def main() -> None:
    print("=" * 70, flush=True)
    print(f"CodeCrossEnc-v1 training | device=mps | started {time.strftime('%H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)

    assert torch.backends.mps.is_available(), "MPS unavailable, abort"
    random.seed(SEED)
    gc.collect()

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1] Loading CodeSearchNet python train split...", flush=True)
    t0 = time.time()
    ds = load_dataset("code_search_net", "python", split="train")
    print(f"  {len(ds)} raw rows in {time.time()-t0:.1f}s", flush=True)

    print(f"[2] Building pairs (limit={POS_LIMIT}, 1 pos + 2 neg each)...", flush=True)
    clean: list[tuple[str, str]] = []
    for r in ds:
        if len(clean) >= POS_LIMIT:
            break
        doc = (r.get("func_documentation_string") or "").strip()
        code = r.get("func_code_string") or ""
        if len(doc) < 10 or len(code) < 40:
            continue
        query = doc.splitlines()[0][:200]
        clean.append((query, code))
    print(f"  {len(clean)} clean positive pairs", flush=True)

    all_codes = [c for _, c in clean]
    train_examples: list[InputExample] = []
    for query, code in clean:
        train_examples.append(InputExample(texts=[query, code], label=1.0))
        for _ in range(2):
            neg = random.choice(all_codes)
            if neg != code:
                train_examples.append(InputExample(texts=[query, neg], label=0.0))
    random.shuffle(train_examples)
    print(f"  {len(train_examples)} total examples", flush=True)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

    print(f"[3] Loading base ms-marco-MiniLM-L-6-v2 on MPS...", flush=True)
    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        num_labels=1,
        max_length=384,
        device="mps",
    )

    print(f"[4] Training 1 epoch -> {OUT_DIR} (local Mac SSD, no FUSE)...", flush=True)
    t_train = time.time()
    model.fit(
        train_dataloader=train_dataloader,
        epochs=1,
        warmup_steps=500,
        optimizer_params={"lr": 2e-5},
        output_path=str(OUT_DIR),
        save_best_model=False,
        show_progress_bar=True,
    )
    elapsed_min = (time.time() - t_train) / 60
    print(f"[5] Training done in {elapsed_min:.1f}min", flush=True)

    # Verify save (Mac local, no FUSE; sync still good practice)
    os.sync()
    print(f"\n[6] Saved files in {OUT_DIR}:", flush=True)
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.relative_to(OUT_DIR)}  {size_mb:.2f}MB", flush=True)

    print(f"\n[7] DONE. {time.strftime('%H:%M:%S')}", flush=True)
    print(f"    Total elapsed: {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
