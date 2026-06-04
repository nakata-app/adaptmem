"""Mnemonics-native FT-CE: mine -> train -> gate eval, tek Kaggle kerneli.

Hedef (Atakan: 0.98): 16 fail'in 7'sinde chat-ce-v3 dogru cevabi seciyor ama
margin'i 0.0003-0.018 (esik alti). Cozum: chat-ce-v3'u mnemonics pipeline'inin
KENDI hard-negative'leriyle keskinlestir (sifirdan degil, ustunden fine-tune).

Protokol disiplini:
  - Egitim ciftleri SADECE 100 train sorusundan (split_ids_100_400).
  - Yeni CE'nin margin'i SADECE train-100 per-q kayitlarindan secilir.
  - Rapor: full-500 (vitrin) + temiz-400 (CE'nin hic gormedigi sorular).

Asamalar:
  1. Mining: champion config'le train-100 koşusu, --dump-candidates
  2. Pair build: pozitif = gold session turn-chunk'lari; negatif = dump'taki
     gold-olmayan top adaylar (hard) + gold-olmayan sessionlardan ornek (easy)
  3. Train: CrossEncoder(chat-ce-v3) uzerinden BCE, 2 epoch, lr 1e-5
  4. Eval A: champion 0.968 (gate=chat-ce-v3 m=0.2 pin=0.5)
     Eval B: ayni config, gate=mn-ce-v1 (ayni m/pin, gecici)
  5. Analiz: A/B flip + temiz-400 + train-100-secimli margin onerisi
"""
import os, sys, subprocess, json, time, glob, random
import torch

TRAIN_QIDS = set(["cc06de0d", "f9e8c073", "b320f3f8", "a89d7624", "311778f1", "gpt4_59c863d7", "bbf86515", "099778bb", "e831120c", "dcfa8644", "8fb83627", "e66b632c", "gpt4_7fce9456", "55241a1f", "352ab8bd", "f4f1d8a4", "830ce83f", "2311e44b", "09ba9854", "gpt4_a1b77f9c", "07741c45", "gpt4_70e84552", "b46e15ee", "6071bd76", "6f9b354f", "1d4da289", "gpt4_8279ba02", "6456829e_abs", "0db4c65d", "d6062bb9", "60bf93ed_abs", "d3ab962e", "87f22b4a", "e01b8e2f", "gpt4_7ddcf75f", "8ebdbe50", "26bdc477", "29f2956b_abs", "2311e44b_abs", "75f70248", "852ce960", "f0e564bc", "fca70973", "3c1045c8", "18bc8abd", "afdc33df", "54026fce", "b9cfe692", "6456829e", "e6041065", "gpt4_15e38248", "gpt4_2ba83207", "2133c1b5_abs", "gpt4_8279ba03", "76d63226", "1192316e", "gpt4_fa19884d", "gpt4_372c3eed_abs", "1a8a66a6", "gpt4_fe651585", "e25c3b8d", "945e3d21", "86b68151", "1c0ddc50", "1e043500", "d682f1a2", "gpt4_b5700ca0", "91b15a6e", "ce6d2d27", "f523d9fe", "7024f17c", "8752c811", "gpt4_f420262d", "d01c6aa8", "4b24c848", "7e974930", "3fdac837", "gpt4_b4a80587", "c18a7dc8", "80ec1f4f_abs", "7527f7e2", "6ade9755", "89941a94", "gpt4_1d80365e", "2133c1b5", "06db6396", "gpt4_88806d6e", "88432d0a", "3ba21379", "0862e8bf", "aae3761f", "5025383b", "gpt4_e061b84f", "73d42213", "4bc144e2", "gpt4_5501fe77", "00ca467f", "dfde3500", "01493427", "b6025781"])

BASE = '/kaggle/working'
DATA_DIR = '/tmp/lme_data'   # 277MB dataset output'a karismasin
os.makedirs(DATA_DIR, exist_ok=True)
print(f'workspace: {BASE}')

print(f'cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'device: {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}')
    if cap[0] < 7:
        print('!!! sm<70 abort'); sys.exit(2)
else:
    print('!!! no cuda'); sys.exit(1)

# chat-ce-v3 (Kaggle dataset)
GATE_CE = None
for cand in glob.glob('/kaggle/input/*/'):
    if (os.path.exists(os.path.join(cand, 'model.safetensors'))
            and os.path.exists(os.path.join(cand, 'config.json'))):
        GATE_CE = cand.rstrip('/')
        break
print(f'chat-ce-v3: {GATE_CE}')
if GATE_CE is None:
    print('!!! chat-ce-v3 dataset yok'); sys.exit(3)

# Data
DATA = f'{DATA_DIR}/longmemeval_s_cleaned.json'
if not os.path.exists(DATA) or os.path.getsize(DATA) < 100_000_000:
    print('=== HF download ===')
    os.system('pip install -q huggingface_hub')
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id='xiaowu0162/longmemeval-cleaned',
                        filename='longmemeval_s_cleaned.json',
                        repo_type='dataset', local_dir=DATA_DIR)
    import shutil
    if p != DATA: shutil.move(p, DATA)
print(f'dataset: {os.path.getsize(DATA)/1e6:.0f} MB')

# mnemonics
WD = f'{BASE}/mnemonics'
if not os.path.isdir(WD):
    subprocess.run(['git','clone','--depth','1',
                    'https://github.com/nakata-app/mnemonics.git', WD], check=True)
subprocess.run(['git','-C',WD,'log','--oneline','-1'], check=True)
src = open(f'{WD}/benchmarks/longmemeval_eval.py').read()
if '--dump-candidates' not in src or '--trust-gate-pin-margin' not in src:
    print('!!! clone eski (dump/pin flag yok)'); sys.exit(4)
subprocess.run(f'pip install -q -e {WD} sentence-transformers numpy adaptmem 2>&1 | tail -2',
               shell=True, check=True)

R = f'{BASE}/results'; os.makedirs(R, exist_ok=True)
env = {**os.environ, 'LME_DATA': DATA, 'PYTHONUNBUFFERED': '1'}
champ = ['python', 'benchmarks/longmemeval_eval.py',
         '--mode', 'rerank', '--chunk-mode', 'turn', '--temporal-aware',
         '--candidate-k', '50', '--seed', '42']

# --- 1. Mining (train-100) ---
split_path = f'{DATA_DIR}/train100_split.json'
json.dump({'dev': sorted(TRAIN_QIDS)}, open(split_path, 'w'))
print('\n=== 1. MINING (train-100, champion config, gate yok) ===')
t0 = time.time()
subprocess.run(champ + ['--split-file', split_path,
                        '--dump-candidates', f'{R}/train100_cands.json',
                        '--out', f'{R}/mining_run.json'],
               cwd=WD, env=env, check=True)
print(f'mining: {time.time()-t0:.0f}s')

# --- 2. Pair build ---
print('\n=== 2. PAIR BUILD ===')
sys.path.insert(0, f'{WD}/benchmarks')
from longmemeval_eval import _session_turn_chunks

def clean(t):
    t = t or ''
    return t.split('|', 1)[1] if t.startswith('SID=') and '|' in t else t

all_q = {q['question_id']: q for q in json.load(open(DATA))}
cands = json.load(open(f'{R}/train100_cands.json'))
rng = random.Random(42)
pairs = []  # (query, text, label)
for rec in cands:
    q = all_q.get(rec['qid'])
    if q is None: continue
    gold = set(rec['answer_sids'])
    query = rec['question']
    # pozitifler: gold sessionlarin turn-chunk'lari (pipeline'la ayni chunker)
    pos = []
    for sid, sess in zip(q['haystack_session_ids'], q['haystack_sessions']):
        if sid in gold:
            pos += [clean(c) for c in _session_turn_chunks(sid, sess)]
    # hard negatifler: dump'taki gold-olmayan top adaylar
    hard = [clean(r['text']) for r in rec['rows'] if r['sid'] not in gold]
    # easy negatifler: gold-olmayan sessionlardan rastgele chunk
    easy_pool = []
    for sid, sess in zip(q['haystack_session_ids'], q['haystack_sessions']):
        if sid not in gold:
            easy_pool += [clean(c) for c in _session_turn_chunks(sid, sess)]
    easy = rng.sample(easy_pool, min(len(easy_pool), 30))
    # dengeleme: pozitifleri ~negatif sayisina yaklastir (max 3x kopya)
    neg = hard + easy
    reps = min(3, max(1, len(neg) // max(len(pos), 1)))
    for t in pos * reps: pairs.append((query, t, 1))
    for t in neg:        pairs.append((query, t, 0))
rng.shuffle(pairs)
n_pos = sum(1 for p in pairs if p[2] == 1)
print(f'pairs: {len(pairs)} (pos={n_pos}, neg={len(pairs)-n_pos})')

# --- 3. Train (chat-ce-v3 uzerinden) ---
print('\n=== 3. TRAIN mn-ce-v1 (init=chat-ce-v3, BCE, 2ep, lr1e-5) ===')
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
ce = CrossEncoder(GATE_CE, num_labels=1, device='cuda')
train_samples = [InputExample(texts=[a, b], label=float(l)) for a, b, l in pairs]
dl = DataLoader(train_samples, shuffle=True, batch_size=16)
t0 = time.time()
ce.fit(train_dataloader=dl, epochs=2, warmup_steps=int(0.1*len(dl)*2),
       optimizer_params={'lr': 1e-5}, show_progress_bar=False)
MN_CE = f'{BASE}/output/mn-ce-v1'
os.makedirs(MN_CE, exist_ok=True)
ce.save(MN_CE)
del ce; torch.cuda.empty_cache()
print(f'train: {time.time()-t0:.0f}s -> {MN_CE}')

# --- 4. Eval A (champion 0.968) + B (mn-ce-v1) ---
full = champ + ['--n', '500']
def run_eval(tag, gate_path):
    print(f'\n=== EVAL {tag} (gate={os.path.basename(gate_path)}) ===')
    t0 = time.time()
    subprocess.run(full + ['--trust-gate-ce', gate_path,
                           '--trust-gate-margin', '0.2',
                           '--trust-gate-pin-margin', '0.5',
                           '--out', f'{R}/{tag}.json',
                           '--per-q-out', f'{R}/{tag}_perq.json'],
                   cwd=WD, env=env, check=True)
    d = json.load(open(f'{R}/{tag}.json'))['mnemonics_rerank']
    print(f"{tag}  R@1={d['R@1']:.3f} R@5={d['R@5']:.3f} R@10={d['R@10']:.3f} ({time.time()-t0:.0f}s)")
    return d

A = run_eval('A', GATE_CE)
B = run_eval('B', MN_CE)

# --- 5. Analiz ---
ap = {q['qid']: q for q in json.load(open(f'{R}/A_perq.json'))}
bp = {q['qid']: q for q in json.load(open(f'{R}/B_perq.json'))}
gain = sum(1 for q in ap if not ap[q]['hit@1'] and bp[q]['hit@1'])
loss = sum(1 for q in ap if ap[q]['hit@1'] and not bp[q]['hit@1'])

def r1(d, qids):
    s = [d[q]['hit@1'] for q in d if (q in qids) == True]
    return sum(s)/len(s)
test_ids = set(bp) - TRAIN_QIDS
print('\n' + '='*70)
print('SUMMARY — mn-ce-v1 vs chat-ce-v3 (ikisi de m=0.2 pin=0.5)')
print('='*70)
print(f"A (chat-ce-v3)  full={A['R@1']:.4f}  temiz400={r1(ap, test_ids):.4f}")
print(f"B (mn-ce-v1)    full={B['R@1']:.4f}  temiz400={r1(bp, test_ids):.4f}")
print(f"flip: helped={gain} hurt={loss}")

# margin onerisi SADECE train-100'den
def est(d, qids, m):
    hit = n = 0
    for qid in d:
        if qid not in qids: continue
        q = d[qid]; g = q.get('gate') or {}
        gold = set(q.get('answer_sids') or [])
        base, pick, marg = g.get('base_top1_sid'), g.get('ftce_best_sid'), g.get('ftce_margin')
        top1 = pick if (marg is not None and pick and pick != base and marg >= m) else base
        hit += 1 if top1 in gold else 0; n += 1
    return hit/max(n,1)
print('\nmargin onerisi (SADECE train-100 est):')
best = max(((m, est(bp, TRAIN_QIDS, m)) for m in
            (0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0)), key=lambda x: x[1])
for m in (0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0):
    print(f'  m={m:<5} train100_est={est(bp, TRAIN_QIDS, m):.4f}')
print(f'secilen margin (train-100): {best[0]}')
print(f'bu margin\'le temiz-400 EST (dogrulama kosusu gerekir): {est(bp, test_ids, best[0]):.4f}')

json.dump({'A_chatce': A, 'B_mnce': B,
           'A_clean400': r1(ap, test_ids), 'B_clean400': r1(bp, test_ids),
           'helped': gain, 'hurt': loss,
           'recommended_margin_train100': best[0]},
          open(f'{R}/ftce_summary.json', 'w'), indent=2)
print(f'\nbundle: {R}/ftce_summary.json')
