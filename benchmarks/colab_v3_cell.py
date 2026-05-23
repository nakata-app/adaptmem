"""AdaptMem v3 Colab cell - multi-negative + gradient accumulation + CachedMNRL.

Usage on Colab:
1. Runtime > Change runtime type > T4 GPU
2. Upload longmemeval_s dataset to /content/ (or mount Drive)
3. Run this cell

Improvements over v2:
- n_negatives=3: 3x training pairs from same labelled data
- gradient_accumulation_steps=16: effective batch 32 without extra VRAM
- loss_type="cached_mnrl": gradient-cached large-batch loss
- Miss analysis: logs which questions R@1 failed on
"""
# !pip install -q sentence-transformers torch numpy

import json, os, sys, time, random
import numpy as np
import torch

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")

# --- DATA ---
# Drive'dan oku (bir kere yukle, hep orda kalir)
# Dosya adi: longmemeval_s_cleaned.json
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except:
    pass

DATA = None
for p in [
    '/content/longmemeval_s_cleaned.json',
    '/content/drive/MyDrive/longmemeval_s_cleaned.json',
    '/content/drive/MyDrive/datasets/longmemeval_s_cleaned.json',
    '/content/longmemeval_s',
]:
    if os.path.exists(p):
        DATA = p
        break
if DATA is None:
    raise FileNotFoundError(
        "longmemeval_s_cleaned.json bulunamadi.\n"
        "Drive'a yukle (drive.google.com > surukle birak) veya Colab'da files.upload() kullan."
    )

with open(DATA) as f:
    all_questions = json.load(f)
print(f"Dataset: {len(all_questions)} questions")

# --- SPLIT ---
SPLIT_PATH = '/content/split_ids_400_100.json'
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

# --- HELPERS ---
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
    misses = []
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
        h1 = recall_at_k(ranked, gt, 1)
        h5 = recall_at_k(ranked, gt, 5)
        h10 = recall_at_k(ranked, gt, 10)
        r1 += h1
        r5 += h5
        r10 += h10
        if h1 == 0.0:
            misses.append({
                'qid': q.get('question_id', f'q{i}'),
                'type': q.get('question_type', '?'),
                'q': q['question'][:100],
                'expected': sorted(gt),
                'got_rank1': ranked[0] if ranked else None,
                'found_at_5': h5 == 1.0,
                'found_at_10': h10 == 1.0,
            })
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f'  {label} q{i+1}/{n}  eta {el*(n-i-1)/(i+1):.0f}s')
    ne = n - skipped
    w = round(time.time() - t0, 2)
    res = {
        'label': label,
        'r1': round(r1/ne, 4), 'r5': round(r5/ne, 4), 'r10': round(r10/ne, 4),
        'wall_clock_s': w, 'n_misses': len(misses), 'misses': misses,
    }
    print(f'\n  {label}: R@1={res["r1"]}  R@5={res["r5"]}  R@10={res["r10"]}  ({w:.0f}s)  misses={len(misses)}/{ne}')
    if misses:
        by_type = {}
        for m in misses:
            by_type[m['type']] = by_type.get(m['type'], 0) + 1
        for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f'    miss {t}: {c}')
    return res

print('Helpers OK')

# --- INSTALL ADAPTMEM ---
print('\n=== INSTALLING ADAPTMEM ===')
os.system('pip install -q git+https://github.com/nakata-app/adaptmem.git')

# build_labelled_queries needs the benchmarks module
os.system('git clone --depth 1 https://github.com/nakata-app/adaptmem.git /content/adaptmem_src 2>/dev/null || true')
sys.path.insert(0, '/content/adaptmem_src/benchmarks')
from longmemeval_eval import build_labelled_queries

corpus, labelled = build_labelled_queries(train_qs)
print(f'Corpus: {len(corpus)}, Queries: {len(labelled)}')

# --- TRAIN ---
from adaptmem import AdaptMem, TrainConfig

MODEL_NAME = 'BAAI/bge-large-en-v1.5'

cfg = TrainConfig(
    epochs=5,
    batch_size=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    top_k_mine=15,
    n_negatives=3,
    gradient_accumulation_steps=16,
    loss_type="cached_mnrl",
)

am = AdaptMem(base_model=MODEL_NAME, device='cuda')

print(f'\n=== FINE-TUNING {MODEL_NAME} ===')
print(f'Config: epochs={cfg.epochs}, batch={cfg.batch_size}, n_neg={cfg.n_negatives}, '
      f'accum={cfg.gradient_accumulation_steps}, loss={cfg.loss_type}, top_k_mine={cfg.top_k_mine}')
stats = am.train(corpus=corpus, labelled=labelled, config=cfg)
print(f'Done! {stats["n_pairs"]} pairs, {stats["runtime_s"]}s')

ft_model = am.encoder

# VRAM temizligi: egitim sonrasi gradient/optimizer buffer'larini sil
import gc
del am
gc.collect()
torch.cuda.empty_cache()
vram_free = torch.cuda.mem_get_info()[0] / 1e9
print(f'VRAM temizlendi, {vram_free:.1f} GB bos')

# --- EVAL ---
print('\n=== EVAL 1: bi-encoder only ===')
ft_only = evaluate_model(ft_model, test_qs, label='v3-bienc')

# Bi-encoder'i CPU'ya tasi, cross-encoder'a yer ac
ft_model.to('cpu')
torch.cuda.empty_cache()
print('\n=== LOADING CROSS-ENCODER ===')
from sentence_transformers import CrossEncoder
ce = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cuda')
print('Cross-encoder loaded')

# Bi-encoder'i GPU'ya geri al (ikisi de inference, T4'e sigar)
ft_model.to('cuda')
torch.cuda.empty_cache()
vram_used = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9
print(f'Her iki model GPU\'da, VRAM kullanimi: {vram_used:.1f} GB')

print('\n=== EVAL 2: bi-encoder + CE rerank (top-30) ===')
ft_rerank30 = evaluate_model(ft_model, test_qs, label='v3+CE-30', reranker=ce, rerank_top_k=30)

print('\n=== EVAL 3: bi-encoder + CE rerank (top-50) ===')
ft_rerank50 = evaluate_model(ft_model, test_qs, label='v3+CE-50', reranker=ce, rerank_top_k=50)

# --- RESULTS TABLE ---
print('\n' + '=' * 70)
print('RESULTS COMPARISON')
print('=' * 70)
print(f'{"Model":<35} {"R@1":>6} {"R@5":>6} {"R@10":>6} {"Miss":>5}')
print('-' * 70)
print(f'{"MiniLM baseline (ref)":<35} {0.795:>6.3f} {0.965:>6.3f} {0.980:>6.3f}')
print(f'{"MiniLM FT-300 (ref)":<35} {0.915:>6.3f} {0.995:>6.3f} {0.995:>6.3f}')
print(f'{"MemPalace (ref)":<35} {0.920:>6.3f} {"--":>6} {1.000:>6.3f}')
print(f'{"BGE-large FT v1 (ref)":<35} {0.950:>6.3f} {0.995:>6.3f} {1.000:>6.3f}')
print('-' * 70)
for r in [ft_only, ft_rerank30, ft_rerank50]:
    misses = r['n_misses']
    print(f'{r["label"]:<35} {r["r1"]:>6.3f} {r["r5"]:>6.3f} {r["r10"]:>6.3f} {misses:>5}')
print('=' * 70)

best = max([ft_only, ft_rerank30, ft_rerank50], key=lambda x: x['r1'])
print(f'\nBest: {best["label"]}  R@1={best["r1"]}  misses={best["n_misses"]}')

# --- SAVE ---
from pathlib import Path
OUT_DIR = '/content/output'
os.makedirs(OUT_DIR, exist_ok=True)
am.save(Path(OUT_DIR) / 'bge-large-ft-v3')
all_results = {
    'v3_config': {
        'epochs': cfg.epochs, 'batch_size': cfg.batch_size,
        'n_negatives': cfg.n_negatives, 'gradient_accumulation_steps': cfg.gradient_accumulation_steps,
        'loss_type': cfg.loss_type, 'top_k_mine': cfg.top_k_mine,
    },
    'ft_only': ft_only, 'ft_rerank30': ft_rerank30, 'ft_rerank50': ft_rerank50,
    'train_stats': stats,
}
Path(OUT_DIR, 'results_v3.json').write_text(json.dumps(all_results, indent=2))
print(f'\nModel saved: {OUT_DIR}/bge-large-ft-v3/')
print(f'Results saved: {OUT_DIR}/results_v3.json')
print('\nBitti! Sonuclari buraya yapistir.')
