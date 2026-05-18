# Colab cell: CodeCrossEnc-v2 rerank eval on CodeSearchNet python full test (22k)
#
# Pipeline:
#   1. FT-Code-5000 bi-encoder retrieve top-20 candidates per query
#   2. CodeCrossEnc-v2 cross-encoder rerank those 20
#   3. Compute R@1, R@5, R@10, MRR on full test set
#
# Compare against:
#   - FT-Code-5000 alone:            R@1 = 0.926, MRR = 0.952
#   - + CodeCrossEnc-v1 (random-neg): R@1 = 0.9148 (negative transfer)
#   - + CodeCrossEnc-v2 (hard-neg):   <-- this run
#
# Estimated time on T4: ~15-20 min

# ---- Cell 1: Drive mount + paths ----
from google.colab import drive
drive.mount('/content/drive')

BI_PATH    = '/content/drive/MyDrive/adaptmem-bench/ft-code/ft-code-5000/model'
CROSS_PATH = '/content/drive/MyDrive/adaptmem-bench/code-crossenc/v2'
OUT_DIR    = '/content/drive/MyDrive/adaptmem-bench/eval/codecrossenc-v2'

import os
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Cell 2: Install ----
# !pip install -q sentence-transformers datasets

# ---- Cell 3: Load test set ----
import time, json
from datasets import load_dataset
from pathlib import Path

def load_test(n: int = -1):
    ds = load_dataset('code_search_net', 'python', split='test')
    queries, truths, corpus_ids = [], [], []
    seen = set()
    for i, row in enumerate(ds):
        if n > 0 and len(queries) >= n:
            break
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
    body_by_id = {f'ts{i}': (row.get('func_code_string') or '')
                  for i, row in enumerate(ds)}
    corpus_texts = [body_by_id[cid] for cid in corpus_ids]
    return queries, truths, corpus_ids, corpus_texts

t0 = time.time()
queries, truths, corpus_ids, corpus_texts = load_test(n=-1)
print(f'[load] {len(queries)} queries, {len(corpus_ids)} corpus in {time.time()-t0:.1f}s')

# ---- Cell 4: Bi-encoder retrieve top-K ----
import numpy as np
from sentence_transformers import SentenceTransformer

TOP_K = 20

print('[bi] loading FT-Code-5000')
bi = SentenceTransformer(BI_PATH)
bi.max_seq_length = 256

t0 = time.time()
corpus_emb = bi.encode(corpus_texts, batch_size=128, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
query_emb  = bi.encode(queries,      batch_size=128, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
print(f'[bi] encoded in {time.time()-t0:.1f}s')

t1 = time.time()
sims = query_emb @ corpus_emb.T
topk_unsorted = np.argpartition(-sims, TOP_K, axis=1)[:, :TOP_K]
rows = np.arange(sims.shape[0])[:, None]
topk_scores  = sims[rows, topk_unsorted]
order        = np.argsort(-topk_scores, axis=1)
topk_idx     = topk_unsorted[rows, order]
topk_scores  = topk_scores[rows, order]
print(f'[bi] top-{TOP_K} retrieved in {time.time()-t1:.1f}s')

cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
truth_idx_arr = np.array([cid_to_idx[t] for t in truths])
bi_r1 = bi_r5 = bi_r10 = 0
bi_mrr = 0.0
for qi in range(len(queries)):
    ranked = topk_idx[qi]
    pos_arr = np.where(ranked == truth_idx_arr[qi])[0]
    if len(pos_arr) == 0:
        continue
    pos = int(pos_arr[0])
    if pos == 0:  bi_r1  += 1
    if pos < 5:   bi_r5  += 1
    if pos < 10:  bi_r10 += 1
    bi_mrr += 1.0 / (pos + 1)
n_q = len(queries)
print(f'[bi-alone] R@1={bi_r1/n_q:.4f} R@5={bi_r5/n_q:.4f} R@10={bi_r10/n_q:.4f} MRR={bi_mrr/n_q:.4f}')

# ---- Cell 5: Cross-encoder rerank ----
from sentence_transformers import CrossEncoder

print('[cross] loading CodeCrossEnc-v2')
cross = CrossEncoder(CROSS_PATH, max_length=384)

t2 = time.time()
pairs = []
for qi in range(len(queries)):
    q = queries[qi]
    for cidx in topk_idx[qi]:
        pairs.append((q, corpus_texts[int(cidx)]))
print(f'[cross] {len(pairs)} pairs to score')

CHUNK = 4096
scores = np.zeros(len(pairs), dtype=np.float32)
for s in range(0, len(pairs), CHUNK):
    e = min(s + CHUNK, len(pairs))
    scores[s:e] = cross.predict(pairs[s:e], batch_size=64,
                                show_progress_bar=(s == 0))
print(f'[cross] scored in {time.time()-t2:.1f}s')

scores = scores.reshape(len(queries), TOP_K)
rerank_order = np.argsort(-scores, axis=1)
reranked_idx = topk_idx[np.arange(len(queries))[:, None], rerank_order]

# ---- Cell 6: Reranked metrics ----
r1 = r5 = r10 = 0
mrr = 0.0
out_path = Path(OUT_DIR) / 'per_query.jsonl'
with out_path.open('w') as f:
    for qi in range(len(queries)):
        ranked = reranked_idx[qi]
        pos_arr = np.where(ranked == truth_idx_arr[qi])[0]
        if len(pos_arr) == 0:
            rank = TOP_K + 1
        else:
            rank = int(pos_arr[0]) + 1
            if rank == 1:   r1  += 1
            if rank <= 5:   r5  += 1
            if rank <= 10:  r10 += 1
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
print(json.dumps(summary, indent=2))

# ---- Decision gate ----
dv = summary['reranked']
bv = summary['bi_alone']
print('\n=== DECISION GATE ===')
print(f"bi-alone : R@1={bv['R@1']:.4f} R@5={bv['R@5']:.4f} MRR={bv['MRR']:.4f}")
print(f"v2 rerank: R@1={dv['R@1']:.4f} R@5={dv['R@5']:.4f} MRR={dv['MRR']:.4f}")
print(f"delta    : R@1={summary['delta_R@1']:+.4f}  MRR={summary['delta_MRR']:+.4f}")
if dv['R@1'] >= 0.95:
    print('PASS (strong): R@1 >= 0.95. 3-axis probe (chunk x bi x cross) sirada.')
elif dv['R@1'] >= 0.93:
    print('PASS (marginal): R@1 in [0.93, 0.95). Kabul, jphein followup postalabilir.')
else:
    print(f"FAIL: R@1 = {dv['R@1']:.4f} < 0.93. Hard-neg budget arttirilmali.")
