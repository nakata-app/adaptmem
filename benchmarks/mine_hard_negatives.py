"""
CodeCrossEnc-v2: Hard negative mining with FT-Code-5000 bi-encoder.

For each training (query, pos_code) pair, finds top-50 most similar codes
from the training corpus using the bi-encoder, then picks 2 negatives from
ranks 2-50 (rank 1 is the true positive). These "hard negatives" are what
the bi-encoder almost-but-not-quite ranked correctly — exactly the
reranking distribution a cross-encoder needs to learn.

Output: benchmarks/hard_negatives_30k.jsonl
  {"query": ..., "pos": ..., "neg1": ..., "neg2": ...}

Run:
    .venv/bin/python benchmarks/mine_hard_negatives.py
"""
import json, time, random
import numpy as np
from pathlib import Path
from datasets import load_dataset

# sentence-transformers 5.x crashes on code strings with IPv6-like patterns
# (e.g. "[::1]") because urlparse raises ValueError. Patch urlparse at the
# stdlib level so all callers in sentence-transformers see the safe version.
import urllib.parse as _up
_orig_urlparse = _up.urlparse
def _safe_urlparse(url, *a, **kw):
    try:
        return _orig_urlparse(url, *a, **kw)
    except ValueError:
        return _up.ParseResult("", "", url, "", "", "")
_up.urlparse = _safe_urlparse

from sentence_transformers import SentenceTransformer

BI_PATH   = "checkpoints/ft-code/ft-code-5000/ft-code-5000/model"
N_POS     = 30_000
TOP_K     = 50
N_NEG     = 2
OUT       = Path("benchmarks/hard_negatives_30k.jsonl")
LOG_FILE  = Path("benchmarks/mine_hard_negatives.log")

random.seed(42)


def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


log("=== Hard negative mining for CodeCrossEnc-v2 ===")
log(f"bi={BI_PATH}  n_pos={N_POS}  top_k={TOP_K}  n_neg={N_NEG}")

log("loading CodeSearchNet python train...")
ds = load_dataset("code_search_net", "python", split="train")
log(f"dataset: {len(ds)} rows")

samples = []
for r in ds:
    doc  = (r.get("func_documentation_string") or "").strip()
    code = (r.get("func_code_string") or "").strip()
    if len(doc) < 10 or len(code) < 40:
        continue
    samples.append((doc.splitlines()[0][:200], code))
    if len(samples) >= N_POS:
        break

log(f"positive pairs: {len(samples)}")

queries   = [q for q, _ in samples]
pos_codes = [c for _, c in samples]

log("loading FT-Code-5000 bi-encoder...")
bi = SentenceTransformer(BI_PATH, device="cpu")
bi.max_seq_length = 256

log("encoding training corpus (codes)...")
t0 = time.time()
corpus_emb = bi.encode(
    pos_codes, batch_size=128, normalize_embeddings=True,
    convert_to_numpy=True, show_progress_bar=True,
)
log(f"encoded {len(pos_codes)} codes in {time.time()-t0:.1f}s")

log("encoding queries...")
t1 = time.time()
query_emb = bi.encode(
    queries, batch_size=128, normalize_embeddings=True,
    convert_to_numpy=True, show_progress_bar=True,
)
log(f"encoded {len(queries)} queries in {time.time()-t1:.1f}s")

log(f"computing similarities and mining top-{TOP_K} per query...")
t2 = time.time()

# Process in chunks to avoid huge matrix (30K x 30K = ~3.4GB float32)
CHUNK = 1000
records = []
skipped = 0

for start in range(0, len(queries), CHUNK):
    end = min(start + CHUNK, len(queries))
    q_chunk = query_emb[start:end]        # (chunk, 384)
    sims    = q_chunk @ corpus_emb.T      # (chunk, 30K)

    for i, qi in enumerate(range(start, end)):
        row = sims[i]
        # top-K indices, sorted descending
        top_idx = np.argpartition(-row, min(TOP_K, len(row)-1))[:TOP_K]
        top_idx = top_idx[np.argsort(-row[top_idx])]

        # exclude the true positive (qi itself)
        candidates = [int(idx) for idx in top_idx if idx != qi]

        if len(candidates) < N_NEG:
            skipped += 1
            continue

        negs = random.sample(candidates[:TOP_K], N_NEG)
        records.append({
            "query": queries[qi],
            "pos":   pos_codes[qi],
            "neg1":  pos_codes[negs[0]],
            "neg2":  pos_codes[negs[1]],
        })

    if (start // CHUNK) % 5 == 0:
        elapsed = time.time() - t2
        pct = 100 * end / len(queries)
        log(f"  mined {end}/{len(queries)} ({pct:.0f}%) in {elapsed:.0f}s")

log(f"mining done in {time.time()-t2:.1f}s — {len(records)} pairs, {skipped} skipped")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

log(f"saved {len(records)} hard-negative pairs to {OUT}")
log("=== MINING COMPLETE ===")
print(f"\nDONE: {OUT}")
