"""Distillation: 3-encoder RRF ensemble → single bi-encoder student.

Teacher: RRF-3way (default ONNX + FT-Code-1000 + FT-Code-5000) soft scores.
Student: fresh MiniLM bi-encoder fine-tuned with KL loss on teacher soft labels.

Addresses Issue #82: single-model that matches ensemble without 3x inference cost.

Run:
    .venv/bin/python benchmarks/train_distill_ensemble.py
    .venv/bin/python benchmarks/train_distill_ensemble.py --n 50000 --steps 3000
"""
import argparse, json, time, random, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--bi1",   default="checkpoints/ft-code/ft-code-1000/ft-code-1000/model", help="FT-Code-1000 path")
ap.add_argument("--bi2",   default="checkpoints/ft-code/ft-code-5000/ft-code-5000/model", help="FT-Code-5000 path")
ap.add_argument("--default-bi", default="sentence-transformers/all-MiniLM-L6-v2",          help="Default (teacher 3)")
ap.add_argument("--student-base", default="sentence-transformers/all-MiniLM-L6-v2",        help="Student base model")
ap.add_argument("--n",     type=int, default=20000, help="Training query count")
ap.add_argument("--neg",   type=int, default=7,     help="Negatives per query (in-batch)")
ap.add_argument("--steps", type=int, default=2000,  help="Gradient steps")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr",    type=float, default=2e-5)
ap.add_argument("--warmup",type=int, default=200)
ap.add_argument("--temp",  type=float, default=0.05, help="Temperature for teacher soft labels")
ap.add_argument("--out",   default="checkpoints/distilled-ensemble-v1")
args = ap.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
log_f = open(OUT / "train.log", "w")

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    log_f.write(line + "\n"); log_f.flush()

device = "mps" if torch.backends.mps.is_available() else "cpu"
log(f"device={device}  steps={args.steps}  batch={args.batch}  lr={args.lr}")

# ── 1. Load CodeSearchNet train split ───────────────────────────────────────
log("loading CodeSearchNet python train split...")
ds = load_dataset("code_search_net", "python", split="train")
pairs = []
for row in ds:
    doc = (row.get("func_documentation_string") or "").strip()
    code = (row.get("func_code_string") or "").strip()
    if len(doc) < 10 or len(code) < 40:
        continue
    pairs.append((doc.splitlines()[0][:200], code))
    if len(pairs) >= args.n:
        break
random.shuffle(pairs)
log(f"loaded {len(pairs)} training pairs")

# ── 2. Pre-compute teacher soft labels ──────────────────────────────────────
log("loading teacher encoders...")
t_default = SentenceTransformer(args.default_bi, device=device)
t_ft1k    = SentenceTransformer(args.bi1,        device=device)
t_ft5k    = SentenceTransformer(args.bi2,        device=device)

for m in [t_default, t_ft1k, t_ft5k]:
    m.max_seq_length = 256

log("encoding all training docs with 3 teachers (this takes a while)...")
docs = [p[1] for p in pairs]
t0 = time.time()
emb_def  = t_default.encode(docs, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
emb_ft1k = t_ft1k.encode(   docs, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
emb_ft5k = t_ft5k.encode(   docs, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
log(f"doc encoding done in {time.time()-t0:.0f}s")

log("encoding all queries with 3 teachers...")
queries = [p[0] for p in pairs]
qemb_def  = t_default.encode(queries, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
qemb_ft1k = t_ft1k.encode(   queries, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
qemb_ft5k = t_ft5k.encode(   queries, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
log(f"query encoding done in {time.time()-t0:.0f}s total")

# Free teacher models from memory
del t_default, t_ft1k, t_ft5k
if device == "mps":
    torch.mps.empty_cache()

# Stack embeddings: (N, 3, dim)
doc_embs   = np.stack([emb_def,  emb_ft1k,  emb_ft5k],  axis=1)  # (N, 3, 384)
query_embs = np.stack([qemb_def, qemb_ft1k, qemb_ft5k], axis=1)  # (N, 3, 384)

def teacher_soft_label(q_idx: int, candidate_idxs: list[int]) -> torch.Tensor:
    """RRF-3way soft label: average cosine similarity across 3 teachers."""
    scores = np.zeros(len(candidate_idxs), dtype=np.float32)
    for t in range(3):
        qv = query_embs[q_idx, t]       # (384,)
        dv = doc_embs[candidate_idxs, t]  # (K, 384)
        scores += dv @ qv
    scores /= 3.0  # average across teachers → approx RRF signal
    return torch.tensor(scores / args.temp, dtype=torch.float32)

# ── 3. Student model ─────────────────────────────────────────────────────────
log("loading student model...")
tokenizer = AutoTokenizer.from_pretrained(args.student_base)
model     = AutoModel.from_pretrained(args.student_base).to(device)

def mean_pool(model_out, attention_mask):
    token_emb = model_out.last_hidden_state
    mask_exp  = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

def encode_texts(texts: list[str]) -> torch.Tensor:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    emb = mean_pool(out, enc["attention_mask"])
    return F.normalize(emb, dim=-1)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup, args.steps)

# ── 4. Training loop ─────────────────────────────────────────────────────────
log(f"starting distillation training for {args.steps} steps, batch={args.batch}, neg={args.neg}...")
N = len(pairs)
step = 0
total_loss = 0.0

while step < args.steps:
    # Sample batch of queries
    q_idxs = random.sample(range(N), args.batch)

    # For each query, sample neg negatives (different from positive)
    all_idxs = []
    for qi in q_idxs:
        negs = random.sample([j for j in range(N) if j != qi], args.neg)
        all_idxs.append([qi] + negs)  # positive first

    # Encode queries + all candidates with student
    q_texts   = [queries[qi] for qi in q_idxs]
    doc_texts = [docs[all_idxs[i][j]] for i in range(args.batch) for j in range(args.neg + 1)]

    model.train()
    q_emb   = encode_texts(q_texts)                            # (B, D)
    doc_emb = encode_texts(doc_texts).view(args.batch, args.neg + 1, -1)  # (B, K+1, D)

    # Student scores: (B, K+1)
    student_scores = (q_emb.unsqueeze(1) * doc_emb).sum(-1) / args.temp

    # Teacher soft labels: (B, K+1)
    teacher_scores = torch.stack([
        teacher_soft_label(q_idxs[i], all_idxs[i])
        for i in range(args.batch)
    ]).to(device)

    # KL divergence loss (teacher distribution → student)
    teacher_probs = F.softmax(teacher_scores, dim=-1)
    student_log_probs = F.log_softmax(student_scores, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    total_loss += loss.item()
    step += 1

    if step % 100 == 0:
        avg = total_loss / 100
        log(f"step {step}/{args.steps}  loss={avg:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")
        total_loss = 0.0

# ── 5. Save ──────────────────────────────────────────────────────────────────
log(f"saving to {OUT}...")
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)

summary = {
    "type": "distilled-ensemble",
    "teachers": ["all-MiniLM-L6-v2", args.bi1, args.bi2],
    "student_base": args.student_base,
    "n_train": len(pairs),
    "steps": args.steps,
    "batch": args.batch,
    "neg": args.neg,
    "lr": args.lr,
    "temp": args.temp,
    "out": str(OUT),
}
(OUT / "train_summary.json").write_text(json.dumps(summary, indent=2))
log(json.dumps(summary, indent=2))
log("=== DISTILLATION COMPLETE ===")
log_f.close()
print(f"\nDONE: {OUT}")
