"""mn-ce-v2: mined + sentetik(pref/temporal) pairs -> train -> A/B.

A: sampiyon 0.972 (mn-ce-v1 gate m=0.2 pin=0.5 + temporal-v2)
B: ayni config, gate = mn-ce-v2 (chat-ce-v3 uzerinden, mined+synth)

Veri: mined pairs kernel icinde yeniden madenlenir (train-100, v1 tarifi);
sentetik pairs Kaggle dataset'inden (atakanakbaba/mn-ce-v2-data, DeepSeek
paraphrase x8 pref / x4 temporal + gold-pos + hard-neg).
"""
import os, sys, subprocess, json, time, glob, random
import torch

print(f'cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'device: {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}')
    if cap[0] < 7:
        print('!!! sm<70 abort'); sys.exit(2)
else:
    print('!!! no cuda'); sys.exit(1)

BASE = '/kaggle/working'
DATA_DIR = '/tmp/lme_data'; os.makedirs(DATA_DIR, exist_ok=True)

# Üç input: chat-ce-v3 (init), mn-ce-v1 (A gate), mn-ce-v2-data (synth jsonl)
CHAT_CE = MN_V1 = SYNTH = None
for cand in glob.glob('/kaggle/input/*/'):
    base = os.path.basename(cand.rstrip('/'))
    if os.path.exists(os.path.join(cand, 'model.safetensors')):
        if 'chat-ce' in base: CHAT_CE = cand.rstrip('/')
        elif 'mn-ce-v1' in base: MN_V1 = cand.rstrip('/')
    js = glob.glob(os.path.join(cand, '*.jsonl'))
    if js: SYNTH = js[0]
print(f'chat-ce-v3: {CHAT_CE}\nmn-ce-v1: {MN_V1}\nsynth: {SYNTH}')
if not (CHAT_CE and MN_V1 and SYNTH):
    print('!!! eksik input dataset'); sys.exit(3)

DATA = f'{DATA_DIR}/longmemeval_s_cleaned.json'
if not os.path.exists(DATA) or os.path.getsize(DATA) < 100_000_000:
    os.system('pip install -q huggingface_hub')
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id='xiaowu0162/longmemeval-cleaned',
                        filename='longmemeval_s_cleaned.json',
                        repo_type='dataset', local_dir=DATA_DIR)
    import shutil
    if p != DATA: shutil.move(p, DATA)
print(f'dataset: {os.path.getsize(DATA)/1e6:.0f} MB')

WD = f'{BASE}/mnemonics'
if not os.path.isdir(WD):
    subprocess.run(['git','clone','--depth','1',
                    'https://github.com/nakata-app/mnemonics.git', WD], check=True)
subprocess.run(['git','-C',WD,'log','--oneline','-1'], check=True)
if '--temporal-v2' not in open(f'{WD}/benchmarks/longmemeval_eval.py').read():
    print('!!! clone eski'); sys.exit(4)
subprocess.run(f'pip install -q -e {WD} sentence-transformers numpy adaptmem 2>&1 | tail -2',
               shell=True, check=True)

TRAIN_QIDS_PATH = f'{DATA_DIR}/train100_split.json'
R = f'{BASE}/results'; os.makedirs(R, exist_ok=True)
env = {**os.environ, 'LME_DATA': DATA, 'PYTHONUNBUFFERED': '1'}
champ = ['python', 'benchmarks/longmemeval_eval.py',
         '--n', '500', '--mode', 'rerank',
         '--chunk-mode', 'turn', '--temporal-aware', '--temporal-v2',
         '--candidate-k', '50', '--seed', '42']

# --- 1. Mining (v1 tarifiyle, train-100) ---
all_q_list = json.load(open(DATA))
qids = [q['question_id'] for q in all_q_list]
rng0 = random.Random(42); rng0.shuffle(qids)
# v1'deki split dosyasiyla ayni kaynak: adaptmem split_ids_100_400 -> pip'ten gelmiyor;
# train qid'leri synth jsonl'in YANINDA degil; mining icin split'i sabit gomak yerine
# adaptmem reposundaki dosyayi klonla:
subprocess.run(['git','clone','--depth','1',
                'https://github.com/nakata-app/adaptmem.git', f'{BASE}/adaptmem'], check=True)
split = json.load(open(f'{BASE}/adaptmem/benchmarks/data/split_ids_100_400.json'))
TRAIN_QIDS = set(split['train_question_ids'])
json.dump({'dev': sorted(TRAIN_QIDS)}, open(TRAIN_QIDS_PATH, 'w'))
print(f'train qids: {len(TRAIN_QIDS)}')

print('\n=== 1. MINING ===')
t0 = time.time()
subprocess.run(champ + ['--split-file', TRAIN_QIDS_PATH,
                        '--dump-candidates', f'{R}/train100_cands.json',
                        '--out', f'{R}/mining_run.json'],
               cwd=WD, env=env, check=True)
print(f'mining: {time.time()-t0:.0f}s')

# --- 2. Pair build: mined (v1 tarifi) + synth ---
print('\n=== 2. PAIRS ===')
sys.path.insert(0, f'{WD}/benchmarks')
from longmemeval_eval import _session_turn_chunks

def clean(t):
    t = t or ''
    return t.split('|', 1)[1] if t.startswith('SID=') and '|' in t else t

all_q = {q['question_id']: q for q in all_q_list}
cands = json.load(open(f'{R}/train100_cands.json'))
rng = random.Random(42)
pairs = []
for rec in cands:
    q = all_q.get(rec['qid'])
    if q is None: continue
    gold = set(rec['answer_sids'])
    query = rec['question']
    pos = []
    for sid, sess in zip(q['haystack_session_ids'], q['haystack_sessions']):
        if sid in gold:
            pos += [clean(c) for c in _session_turn_chunks(sid, sess)]
    hard = [clean(r['text']) for r in rec['rows'] if r['sid'] not in gold]
    easy_pool = []
    for sid, sess in zip(q['haystack_session_ids'], q['haystack_sessions']):
        if sid not in gold:
            easy_pool += [clean(c) for c in _session_turn_chunks(sid, sess)]
    easy = rng.sample(easy_pool, min(len(easy_pool), 30))
    neg = hard + easy
    reps = min(3, max(1, len(neg) // max(len(pos), 1)))
    for t in pos * reps: pairs.append((query, t, 1))
    for t in neg:        pairs.append((query, t, 0))
n_mined = len(pairs)
for line in open(SYNTH):
    d = json.loads(line)
    pairs.append((d['query'], d['text'], d['label']))
rng.shuffle(pairs)
n1 = sum(1 for p in pairs if p[2] == 1)
print(f'pairs: {len(pairs)} (mined={n_mined}, synth={len(pairs)-n_mined}; pos={n1})')

# --- 3. Train (chat-ce-v3 uzerinden) ---
print('\n=== 3. TRAIN mn-ce-v2 ===')
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
ce = CrossEncoder(CHAT_CE, num_labels=1, device='cuda')
samples = [InputExample(texts=[a, b], label=float(l)) for a, b, l in pairs]
dl = DataLoader(samples, shuffle=True, batch_size=16)
t0 = time.time()
ce.fit(train_dataloader=dl, epochs=2, warmup_steps=int(0.1*len(dl)*2),
       optimizer_params={'lr': 1e-5}, show_progress_bar=False)
MN_V2 = f'{BASE}/output/mn-ce-v2'
os.makedirs(MN_V2, exist_ok=True)
ce.save(MN_V2)
del ce; torch.cuda.empty_cache()
print(f'train: {time.time()-t0:.0f}s -> {MN_V2}')

# --- 4. A/B ---
def run_eval(tag, gate):
    print(f'\n=== EVAL {tag} (gate={os.path.basename(gate)}) ===')
    t0 = time.time()
    subprocess.run(champ + ['--trust-gate-ce', gate,
                            '--trust-gate-margin', '0.2',
                            '--trust-gate-pin-margin', '0.5',
                            '--out', f'{R}/{tag}.json',
                            '--per-q-out', f'{R}/{tag}_perq.json'],
                   cwd=WD, env=env, check=True)
    d = json.load(open(f'{R}/{tag}.json'))['mnemonics_rerank']
    print(f"{tag}  R@1={d['R@1']:.4f} R@5={d['R@5']:.4f} ({time.time()-t0:.0f}s)")
    return d

A = run_eval('A', MN_V1)
B = run_eval('B', MN_V2)

ap = {q['qid']: q for q in json.load(open(f'{R}/A_perq.json'))}
bp = {q['qid']: q for q in json.load(open(f'{R}/B_perq.json'))}
gain = [q for q in ap if not ap[q]['hit@1'] and bp[q]['hit@1']]
loss = [q for q in ap if ap[q]['hit@1'] and not bp[q]['hit@1']]
test_ids = set(bp) - TRAIN_QIDS
def r1(d, ids):
    s = [d[q]['hit@1'] for q in d if q in ids]
    return sum(s)/len(s)

print('\n' + '='*70)
print('SUMMARY — mn-ce-v2 vs mn-ce-v1 (sampiyon config)')
print('='*70)
print(f"A (mn-ce-v1)  full={A['R@1']:.4f}  temiz400={r1(ap, test_ids):.4f}")
print(f"B (mn-ce-v2)  full={B['R@1']:.4f}  temiz400={r1(bp, test_ids):.4f}")
print(f"flip: helped={len(gain)} hurt={len(loss)}")
for q in gain: print(f"  HELP {q[:24]} {ap[q]['qtype']} {'TRAIN' if q in TRAIN_QIDS else 'temiz'}")
for q in loss: print(f"  HURT {q[:24]} {ap[q]['qtype']} {'TRAIN' if q in TRAIN_QIDS else 'temiz'}")
json.dump({'A_mnce_v1': A, 'B_mnce_v2': B,
           'A_clean400': r1(ap, test_ids), 'B_clean400': r1(bp, test_ids),
           'helped': gain, 'hurt': loss},
          open(f'{R}/ftce_v2_summary.json', 'w'), indent=2)
print(f'\nbundle: {R}/ftce_v2_summary.json')
