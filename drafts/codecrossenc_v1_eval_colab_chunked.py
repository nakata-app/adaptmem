# Single-cell Colab eval (CHUNKED matmul, OOM-safe): CodeCrossEnc-v1 rerank
# Bir önceki versiyonda 22k x 22k matmul Colab T4 RAM'i yedi. Bu sürüm chunked.
# T4, ~12-15dk. Çıktı: /content/drive/MyDrive/adaptmem-bench/eval/codecrossenc-v1/summary.json

import os, time, json
from pathlib import Path

try:
    from google.colab import drive
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
except ImportError:
    pass

try:
    import sentence_transformers, datasets  # noqa: F401
except ImportError:
    os.system('pip install -q sentence-transformers datasets')

BI_PATH    = '/content/drive/MyDrive/adaptmem-bench/ft-code/ft-code-5000/model'
CROSS_PATH = '/content/drive/MyDrive/adaptmem-bench/code-crossenc/v1'
OUT_DIR    = '/content/drive/MyDrive/adaptmem-bench/eval/codecrossenc-v1'
TOP_K      = 20

os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load test set ----
from datasets import load_dataset
print('[load] downloading code_search_net python test split...', flush=True)
ds = load_dataset('code_search_net', 'python', split='test')
queries, truths, corpus_ids = [], [], []
body_by_id = {f'ts{i}': (row.get('func_code_string') or '') for i, row in enumerate(ds)}
seen = set()
for i, row in enumerate(ds):
    body = row.get('func_code_string') or ''
    doc = (row.get('func_documentation_string') or '').strip()
    if len(body) < 40 or len(doc) < 10:
        continue
    key = body[:200]
    if key in seen:
        continue
    seen.add(key)
    cid = f'ts{i}'
    queries.append(doc.splitlines()[0][:200])
    truths.append(cid)
    corpus_ids.append(cid)
corpus_texts = [body_by_id[cid] for cid in corpus_ids]
n_q = len(queries)
print(f'[load] {n_q} queries, {len(corpus_ids)} corpus', flush=True)

# ---- Bi-encoder encode ----
import numpy as np
from sentence_transformers import SentenceTransformer

print(f'[bi] loading FT-Code-5000', flush=True)
bi = SentenceTransformer(BI_PATH)
bi.max_seq_length = 256

t0 = time.time()
corpus_emb = bi.encode(corpus_texts, batch_size=128, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
query_emb  = bi.encode(queries,      batch_size=128, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
print(f'[bi] encoded in {time.time()-t0:.1f}s', flush=True)

# ---- CHUNKED top-K retrieve (OOM-safe) ----
t1 = time.time()
topk_idx = np.zeros((n_q, TOP_K), dtype=np.int64)
CHUNK_Q = 500
for s in range(0, n_q, CHUNK_Q):
    e = min(s + CHUNK_Q, n_q)
    sims_chunk = query_emb[s:e] @ corpus_emb.T  # 500 x ~22k = ~44MB
    topk_un = np.argpartition(-sims_chunk, TOP_K, axis=1)[:, :TOP_K]
    scores_chunk = sims_chunk[np.arange(e-s)[:, None], topk_un]
    order = np.argsort(-scores_chunk, axis=1)
    topk_idx[s:e] = topk_un[np.arange(e-s)[:, None], order]
    if s % 5000 == 0:
        print(f'  [bi-chunk] {e}/{n_q} done', flush=True)
print(f'[bi] top-{TOP_K} chunked in {time.time()-t1:.1f}s', flush=True)

# ---- Bi-alone sanity ----
cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
truth_idx_arr = np.array([cid_to_idx[t] for t in truths])
bi_r1 = bi_r5 = bi_r10 = 0
bi_mrr = 0.0
for qi in range(n_q):
    pos_arr = np.where(topk_idx[qi] == truth_idx_arr[qi])[0]
    if len(pos_arr) == 0:
        continue
    pos = int(pos_arr[0])
    if pos == 0:  bi_r1  += 1
    if pos < 5:   bi_r5  += 1
    if pos < 10:  bi_r10 += 1
    bi_mrr += 1.0 / (pos + 1)
print(f'[bi-alone] R@1={bi_r1/n_q:.4f} R@5={bi_r5/n_q:.4f} R@10={bi_r10/n_q:.4f} MRR={bi_mrr/n_q:.4f}',
      flush=True)

# ---- Free bi memory before cross ----
del bi, corpus_emb, query_emb
import gc; gc.collect()
try:
    import torch; torch.cuda.empty_cache()
except ImportError:
    pass

# ---- Cross-encoder rerank ----
from sentence_transformers import CrossEncoder

print(f'[cross] loading CodeCrossEnc-v1', flush=True)
cross = CrossEncoder(CROSS_PATH, max_length=384)

t2 = time.time()
pairs = []
for qi in range(n_q):
    q = queries[qi]
    for cidx in topk_idx[qi]:
        pairs.append((q, corpus_texts[int(cidx)]))
print(f'[cross] {len(pairs)} pairs to score', flush=True)

CHUNK = 4096
scores = np.zeros(len(pairs), dtype=np.float32)
for s in range(0, len(pairs), CHUNK):
    e = min(s + CHUNK, len(pairs))
    scores[s:e] = cross.predict(pairs[s:e], batch_size=64,
                                 show_progress_bar=(s == 0))
print(f'[cross] scored in {time.time()-t2:.1f}s', flush=True)

scores = scores.reshape(n_q, TOP_K)
rerank_order = np.argsort(-scores, axis=1)
reranked_idx = topk_idx[np.arange(n_q)[:, None], rerank_order]

# ---- Reranked metrics ----
r1 = r5 = r10 = 0
mrr = 0.0
out_path = Path(OUT_DIR) / 'per_query.jsonl'
with out_path.open('w') as f:
    for qi in range(n_q):
        ranked = reranked_idx[qi]
        pos_arr = np.where(ranked == truth_idx_arr[qi])[0]
        if len(pos_arr) == 0:
            rank = TOP_K + 1
        else:
            rank = int(pos_arr[0]) + 1
            if rank == 1:  r1  += 1
            if rank <= 5:  r5  += 1
            if rank <= 10: r10 += 1
            mrr += 1.0 / rank
        f.write(json.dumps({'qi': qi, 'truth': truths[qi], 'rank': rank,
                            'top1': corpus_ids[int(ranked[0])]}) + '\n')

summary = {
    'bi_checkpoint':    BI_PATH,
    'cross_checkpoint': CROSS_PATH,
    'top_k':            TOP_K,
    'n_queries':        n_q,
    'n_corpus':         len(corpus_ids),
    'bi_alone':  {'R@1': bi_r1/n_q, 'R@5': bi_r5/n_q,
                  'R@10': bi_r10/n_q, 'MRR': bi_mrr/n_q},
    'reranked':  {'R@1': r1/n_q,    'R@5': r5/n_q,
                  'R@10': r10/n_q,  'MRR': mrr/n_q},
    'delta_R@1': (r1 - bi_r1) / n_q,
    'delta_MRR': (mrr - bi_mrr) / n_q,
}
(Path(OUT_DIR) / 'summary.json').write_text(json.dumps(summary, indent=2))
print()
print('='*60)
print(json.dumps(summary, indent=2))
print('='*60)

r1_final = summary['reranked']['R@1']
if r1_final >= 0.95:
    print(f"\nGATE: TIER-1 (>=0.95). R@1={r1_final:.4f}. Compose ediyor, cross-harness sirada.")
elif r1_final >= 0.93:
    print(f"\nGATE: TIER-2 (0.93-0.95). R@1={r1_final:.4f}. Marjinal pozitif, kabul.")
else:
    print(f"\nGATE: TIER-3 (<0.93). R@1={r1_final:.4f}. Trivial overfit, hard negative mining v2.")
