"""CodeCrossEnc-v1-local: bi-encoder top-20 -> cross-encoder rerank eval.

Uses FT-Code-5000 as bi-encoder, CodeCrossEnc-v1-local as reranker.
Runs on CPU. Default n=5000 for fast result; pass --n -1 for full ~22K.

Run:
    .venv/bin/python benchmarks/eval_codecrossenc_local.py
    .venv/bin/python benchmarks/eval_codecrossenc_local.py --n -1
"""
import argparse, time, json, numpy as np
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5000, help="Queries to eval; -1 = full test set")
ap.add_argument("--bi",    default="checkpoints/ft-code/ft-code-5000/ft-code-5000/model")
ap.add_argument("--cross", default="checkpoints/codecrossenc-v1-local")
args = ap.parse_args()

BI_PATH    = args.bi
CROSS_PATH = args.cross
N_QUERIES  = args.n
OUT_DIR    = Path("benchmarks/results/codecrossenc-v1-local")
TOP_K      = 20

OUT_DIR.mkdir(parents=True, exist_ok=True)
log_f = open(OUT_DIR / "eval.log", "w")


def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    log_f.write(line + "\n")
    log_f.flush()


log("=== CodeCrossEnc-v1-local eval ===")
log(f"bi={BI_PATH}  cross={CROSS_PATH}  top_k={TOP_K}  n={N_QUERIES}")

# Load test split
log(f"loading CodeSearchNet python test split (n={'all' if N_QUERIES < 0 else N_QUERIES})...")
t0 = time.time()
ds = load_dataset("code_search_net", "python", split="test")
queries, truths, corpus_ids, corpus_texts = [], [], [], []
seen: set[str] = set()
for i, row in enumerate(ds):
    if N_QUERIES > 0 and len(queries) >= N_QUERIES:
        break
    body = row.get("func_code_string") or ""
    doc  = (row.get("func_documentation_string") or "").strip()
    if len(body) < 40 or len(doc) < 10:
        continue
    key = body[:200]
    if key in seen:
        continue
    seen.add(key)
    cid = f"ts{i}"
    queries.append(doc.splitlines()[0][:200])
    truths.append(cid)
    corpus_ids.append(cid)
    corpus_texts.append(body)

n_q = len(queries)
log(f"loaded {n_q} queries, {len(corpus_ids)} corpus in {time.time()-t0:.1f}s")

# Bi-encoder encode
log("loading FT-Code-5000 bi-encoder...")
bi = SentenceTransformer(BI_PATH, device="cpu")
bi.max_seq_length = 256

t1 = time.time()
log("encoding corpus...")
corpus_emb = bi.encode(corpus_texts, batch_size=128, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
log("encoding queries...")
query_emb = bi.encode(queries, batch_size=128, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=True)
log(f"encoded in {time.time()-t1:.1f}s")

# Top-K retrieval
sims   = query_emb @ corpus_emb.T
topk_u = np.argpartition(-sims, TOP_K, axis=1)[:, :TOP_K]
rows   = np.arange(n_q)[:, None]
ts     = sims[rows, topk_u]
order  = np.argsort(-ts, axis=1)
topk   = topk_u[rows, order]   # (n_q, TOP_K) sorted

cid2idx = {c: i for i, c in enumerate(corpus_ids)}
tarr    = np.array([cid2idx[t] for t in truths])

# Bi-alone metrics
bi_r1 = bi_r5 = bi_r10 = 0; bi_mrr = 0.0
for qi in range(n_q):
    pos_arr = np.where(topk[qi] == tarr[qi])[0]
    if not len(pos_arr): continue
    p = int(pos_arr[0])
    if p == 0:  bi_r1  += 1
    if p < 5:   bi_r5  += 1
    if p < 10:  bi_r10 += 1
    bi_mrr += 1.0 / (p + 1)
log(f"[bi-alone] R@1={bi_r1/n_q:.4f} R@5={bi_r5/n_q:.4f} R@10={bi_r10/n_q:.4f} MRR={bi_mrr/n_q:.4f}")

# Cross-encoder rerank
log("loading CodeCrossEnc-v1-local...")
cross = CrossEncoder(CROSS_PATH, max_length=384, device="cpu")

pairs  = [(queries[qi], corpus_texts[int(c)]) for qi in range(n_q) for c in topk[qi]]
log(f"scoring {len(pairs)} pairs with cross-encoder...")
CHUNK  = 2048
scores = np.zeros(len(pairs), dtype=np.float32)
t2 = time.time()
for s in range(0, len(pairs), CHUNK):
    e = min(s + CHUNK, len(pairs))
    scores[s:e] = cross.predict(pairs[s:e], batch_size=32, show_progress_bar=False)
    if s % 20000 == 0:
        pct = 100 * e / len(pairs)
        log(f"  scored {e}/{len(pairs)} ({pct:.0f}%) in {time.time()-t2:.0f}s")

scores  = scores.reshape(n_q, TOP_K)
ro      = np.argsort(-scores, axis=1)
rerank  = topk[np.arange(n_q)[:, None], ro]

r1 = r5 = r10 = 0; mrr = 0.0
for qi in range(n_q):
    pos_arr = np.where(rerank[qi] == tarr[qi])[0]
    if not len(pos_arr): continue
    p = int(pos_arr[0]) + 1
    if p == 1:  r1  += 1
    if p <= 5:  r5  += 1
    if p <= 10: r10 += 1
    mrr += 1.0 / p

log(f"[reranked] R@1={r1/n_q:.4f} R@5={r5/n_q:.4f} R@10={r10/n_q:.4f} MRR={mrr/n_q:.4f}")

summary = {
    "bi_alone":  {"R@1": bi_r1/n_q, "R@5": bi_r5/n_q, "R@10": bi_r10/n_q, "MRR": bi_mrr/n_q},
    "reranked":  {"R@1": r1/n_q,    "R@5": r5/n_q,    "R@10": r10/n_q,    "MRR": mrr/n_q},
    "delta_R@1": (r1 - bi_r1) / n_q,
    "delta_MRR": (mrr - bi_mrr) / n_q,
    "n_queries": n_q, "top_k": TOP_K,
    "bi_model":    BI_PATH,
    "cross_model": CROSS_PATH,
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
log(json.dumps(summary, indent=2))
log_f.close()
print("\nDONE:", OUT_DIR / "summary.json")
