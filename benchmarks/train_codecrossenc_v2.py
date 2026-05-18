"""
CodeCrossEnc-v2: Cross-encoder trained on hard negatives (CPU, Mac mini safe mode).

Reads hard_negatives_30k.jsonl produced by mine_hard_negatives.py.
Each record: {query, pos, neg1, neg2} → 3 training examples per record.

Key difference vs v1: negatives are top-50 bi-encoder retrievals (not random).
These are the candidates a cross-encoder actually needs to re-rank correctly.

Run:
    .venv/bin/python benchmarks/train_codecrossenc_v2.py
"""
import os, sys, time, json, random
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
torch.backends.mps.is_available = lambda: False

DEVICE   = "cpu"
BATCH    = 8
LR       = 2e-5
WARMUP   = 300
IN_FILE  = "benchmarks/hard_negatives_30k.jsonl"
OUT_DIR  = os.path.expanduser("~/Projects/adaptmem/checkpoints/codecrossenc-v2-local")
LOG_FILE = os.path.expanduser("~/Projects/adaptmem/benchmarks/codecrossenc_v2_train.log")

os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


log("=== CodeCrossEnc-v2 hard-negative training ===")
log(f"device: {DEVICE}  batch: {BATCH}  lr: {LR}  warmup: {WARMUP}")
log(f"input: {IN_FILE}")

log("loading hard-negative pairs...")
records = []
with open(IN_FILE) as f:
    for line in f:
        records.append(json.loads(line))
log(f"loaded {len(records)} records")

from sentence_transformers import InputExample, CrossEncoder
from torch.utils.data import DataLoader

train_examples = []
for r in records:
    train_examples.append(InputExample(texts=[r["query"], r["pos"]],  label=1.0))
    train_examples.append(InputExample(texts=[r["query"], r["neg1"]], label=0.0))
    train_examples.append(InputExample(texts=[r["query"], r["neg2"]], label=0.0))

random.shuffle(train_examples)
log(f"total training examples: {len(train_examples)}")

log("loading cross-encoder base model...")
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    num_labels=1, max_length=384, device=DEVICE,
)
log("model loaded")

loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH)
steps  = len(loader)
log(f"steps per epoch: {steps}  (~{steps*0.4/60:.0f}-{steps*1.5/60:.0f} min on CPU)")

t0 = time.time()
model.fit(
    train_dataloader=loader,
    epochs=1,
    warmup_steps=WARMUP,
    optimizer_params={"lr": LR},
    show_progress_bar=True,
)
elapsed = time.time() - t0
log(f"training done in {elapsed/60:.1f} min")

log(f"saving to {OUT_DIR} ...")
model.save(OUT_DIR)
files = os.listdir(OUT_DIR)
log(f"saved files: {files}")
assert len(files) >= 2, f"save may have failed: {files}"

summary = {
    "n_records": len(records),
    "n_examples": len(train_examples),
    "negative_type": "hard (bi-encoder top-50)",
    "batch": BATCH, "lr": LR,
    "runtime_min": round(elapsed / 60, 1),
    "out_dir": OUT_DIR,
}
log(json.dumps(summary))
log("=== TRAINING COMPLETE ===")
print(f"\nDONE. Checkpoint: {OUT_DIR}")
