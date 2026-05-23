import json, os, sys, time, random
import numpy as np
import torch

DATA = '/kaggle/working/longmemeval_s'
with open(DATA) as f:
    all_questions = json.load(f)

# 400 train / 100 test (daha fazla egitim verisi)
SPLIT_PATH = '/kaggle/working/split_ids_400_100.json'
if os.path.exists(SPLIT_PATH):
    with open(SPLIT_PATH) as f:
        split_ids = json.load(f)
else:
    qids = [q['question_id'] for q in all_questions]
    rng = random.Random(42)
    rng.shuffle(qids)
    split_ids = {'train_question_ids': qids[:400], 'test_question_ids': qids[400:]}
    with open(SPLIT_PATH, 'w') as f:
        json.dump(split_ids, f)

qid_to_q = {q['question_id']: q for q in all_questions}
train_qs = [qid_to_q[qid] for qid in split_ids['train_question_ids'] if qid in qid_to_q]
test_qs = [qid_to_q[qid] for qid in split_ids['test_question_ids'] if qid in qid_to_q]
print(f'Train: {len(train_qs)}, Test: {len(test_qs)}')

def session_to_text(turns):
    return '\n'.join(t.get('content', '') for t in turns if t.get('role') == 'user')

def make_per_question_corpus(q):
    sids = list(q['haystack_session_ids'])
    sessions = list(q['haystack_sessions'])
    docs = [session_to_text(s) for s in sessions]
    active = [(sid, d) for sid, d in zip(sids, docs) if d]
    if not active:
        return [], []
    a_sids, a_docs = zip(*active)
    return list(a_sids), list(a_docs)

def recall_at_k(ret, gt, k):
    return 1.0 if any(g in ret[:k] for g in gt) else 0.0

def evaluate_model(model, questions, label='', reranker=None, rerank_top_k=50):
    n = len(questions)
    r1 = r5 = r10 = 0.0
    skipped = 0
    t0 = time.time()
    kw = dict(batch_size=32, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    for i, q in enumerate(questions):
        sids, docs = make_per_question_corpus(q)
        if not sids:
            skipped += 1
            continue
        embs = model.encode(docs, **kw)
        qv = model.encode([q['question']], **kw)[0]
        scores = embs @ qv
        order = np.argsort(-scores)
        ranked = [sids[j] for j in order]

        if reranker is not None:
            top_n = min(rerank_top_k, len(ranked))
            top_indices = order[:top_n]
            pairs = [(q['question'], docs[j]) for j in top_indices]
            ce_scores = reranker.predict(pairs, show_progress_bar=False)
            ce_order = np.argsort(-np.array(ce_scores))
            reranked_sids = [sids[top_indices[j]] for j in ce_order]
            remaining = [sids[j] for j in order[top_n:]]
            ranked = reranked_sids + remaining

        gt = set(q['answer_session_ids'])
        r1 += recall_at_k(ranked, gt, 1)
        r5 += recall_at_k(ranked, gt, 5)
        r10 += recall_at_k(ranked, gt, 10)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f'  {label} q{i+1}/{n}  eta {el*(n-i-1)/(i+1):.0f}s')
    ne = n - skipped
    w = round(time.time() - t0, 2)
    res = {'label': label, 'r1': round(r1/ne, 4), 'r5': round(r5/ne, 4), 'r10': round(r10/ne, 4), 'wall_clock_s': w}
    misses = int(ne - r1)
    print(f'\n  {label}: R@1={res["r1"]}  R@5={res["r5"]}  R@10={res["r10"]}  ({w:.0f}s)  misses={misses}/{ne}')
    return res

print('Helpers OK')

print('\n=== INSTALLING ADAPTMEM ===')
os.system('git clone --depth 1 https://github.com/nakata-app/adaptmem.git /kaggle/working/adaptmem 2>/dev/null || true')
os.chdir('/kaggle/working/adaptmem')
os.system('pip install -q -e .')

sys.path.insert(0, '/kaggle/working/adaptmem/benchmarks')
from longmemeval_eval import build_labelled_queries

corpus, labelled = build_labelled_queries(train_qs)
print(f'Corpus: {len(corpus)}, Queries: {len(labelled)}')

from adaptmem import AdaptMem
from adaptmem.types import TrainConfig

MODEL_NAME = 'BAAI/bge-large-en-v1.5'
cfg = TrainConfig(epochs=5, batch_size=2, learning_rate=2e-5, warmup_ratio=0.1, top_k_mine=10)
am = AdaptMem(base_model=MODEL_NAME, device='cuda')

print(f'\n=== FINE-TUNING {MODEL_NAME} (5 epoch, 400 train) ===')
stats = am.train(corpus=corpus, labelled=labelled, config=cfg)
print(f'Done! {stats["n_pairs"]} pairs, {stats["runtime_s"]}s')

ft_model = am.encoder

print('\n=== EVAL 1: bi-encoder only (rerank yok) ===')
ft_only = evaluate_model(ft_model, test_qs, label='bge-FT-bienc')

print('\n=== LOADING CROSS-ENCODER ===')
from sentence_transformers import CrossEncoder
ce = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cuda')
print('Cross-encoder loaded')

print('\n=== EVAL 2: bi-encoder + cross-encoder rerank (top-30) ===')
ft_rerank30 = evaluate_model(ft_model, test_qs, label='bge-FT+CE-30', reranker=ce, rerank_top_k=30)

print('\n=== EVAL 3: bi-encoder + cross-encoder rerank (top-50) ===')
ft_rerank50 = evaluate_model(ft_model, test_qs, label='bge-FT+CE-50', reranker=ce, rerank_top_k=50)

print('\n' + '=' * 65)
print('RESULTS (100 test questions)')
print('=' * 65)
print(f'{"Model":<30} {"R@1":>6} {"R@5":>6} {"R@10":>6} {"Miss":>5}')
print('-' * 65)
print(f'{"MiniLM baseline (ref)":<30} {0.795:>6.3f} {0.965:>6.3f} {0.980:>6.3f}')
print(f'{"MiniLM FT-300 (ref)":<30} {0.915:>6.3f} {0.995:>6.3f} {0.995:>6.3f}')
print(f'{"MemPalace (ref)":<30} {0.920:>6.3f} {"--":>6} {1.000:>6.3f}')
print(f'{"BGE-large FT v1 (ref)":<30} {0.950:>6.3f} {0.995:>6.3f} {1.000:>6.3f}')
print('-' * 65)
for r in [ft_only, ft_rerank30, ft_rerank50]:
    misses = int(100 * (1 - r['r1']))
    print(f'{r["label"]:<30} {r["r1"]:>6.3f} {r["r5"]:>6.3f} {r["r10"]:>6.3f} {misses:>5}')
print('=' * 65)

best = max([ft_only, ft_rerank30, ft_rerank50], key=lambda x: x['r1'])
print(f'\nBest: {best["label"]}  R@1={best["r1"]}  misses={int(100*(1-best["r1"]))}')

from pathlib import Path
OUT_DIR = '/kaggle/working/output'
os.makedirs(OUT_DIR, exist_ok=True)
am.save(Path(OUT_DIR) / 'bge-large-ft-400-v2')
all_results = {'ft_only': ft_only, 'ft_rerank30': ft_rerank30, 'ft_rerank50': ft_rerank50, 'stats': stats}
Path(OUT_DIR, 'results_v2.json').write_text(json.dumps(all_results, indent=2))
print(f'\nModel: {OUT_DIR}/bge-large-ft-400-v2/')
print(f'Results: {OUT_DIR}/results_v2.json')
