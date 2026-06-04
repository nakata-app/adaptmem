"""Mnemonics × trust-gated FT-CE bridge eval on Kaggle.

THE proven-path test (adaptmem Sprint 4 Stage 1 → mnemonics port): the
champion 0.954 pipeline is a RANKING problem, not recall (R@10≈0.994,
answer in top-5 ~0.99 but not #1). Pure FT-CE rerank regressed -3pp.
Sprint 4's trust gate (FT-CE overrides #1 only when its logit margin is
high) was helped>0 / hurt=0 on the mempal track — this run measures the
same gate inside mnemonics's own champion pipeline.

Two runs, same champion config, only the gate differs:
  A: champion 0.954 config (ms-marco CE, turn chunks, temporal-aware, k50)
  B: A + --trust-gate-ce chat-ce-v3 --trust-gate-margin 1.0

chat-ce-v3 arrives as a Kaggle dataset (atakanakbaba/chat-ce-v3, mounted
under /kaggle/input/). Requires mnemonics HEAD >= 965c7f6 (the
--trust-gate-ce flag); the script verifies and aborts otherwise.
"""
import os, sys, subprocess, json, time, glob
import torch

# Portable workspace
if os.path.exists('/kaggle/working') and os.access('/kaggle/working', os.W_OK):
    BASE = '/kaggle/working'
elif os.path.exists('/home/gridai') or os.path.expanduser('~') != '/root':
    BASE = os.path.expanduser('~/workspace')
    os.makedirs(BASE, exist_ok=True)
else:
    BASE = '/tmp/krun_workspace'
    os.makedirs(BASE, exist_ok=True)
print(f'workspace: {BASE}')

# Hardware + sm_70 guard (P100 draw aborts cleanly to preserve quota)
print(f'cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'device: {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}')
    if cap[0] < 7:
        print(f'!!! sm_{cap[0]}{cap[1]} < sm_70 — abort. Retry for T4.'); sys.exit(2)
else:
    print('!!! no cuda'); sys.exit(1)

# Locate the chat-ce-v3 checkpoint mounted as a Kaggle dataset
GATE_CE = None
for cand in glob.glob('/kaggle/input/*/'):
    if (os.path.exists(os.path.join(cand, 'model.safetensors'))
            and os.path.exists(os.path.join(cand, 'config.json'))):
        GATE_CE = cand.rstrip('/')
        break
print(f'\n=== gate CE checkpoint: {GATE_CE} ===')
if GATE_CE is None:
    print('!!! chat-ce-v3 dataset bulunamadi (krun --dataset atakanakbaba/chat-ce-v3 ?)')
    sys.exit(3)
print(subprocess.run(['ls', '-la', GATE_CE], capture_output=True, text=True).stdout[:800])

# Data: cleaned longmemeval_s (matches the local 0.954 baseline)
DATA = f'{BASE}/longmemeval_s_cleaned.json'
if not os.path.exists(DATA) or os.path.getsize(DATA) < 100_000_000:
    print('\n=== HF download ===')
    os.system('pip install -q huggingface_hub')
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id='xiaowu0162/longmemeval-cleaned',
                        filename='longmemeval_s_cleaned.json',
                        repo_type='dataset', local_dir=BASE)
    import shutil
    if p != DATA: shutil.move(p, DATA)
print(f'Dataset: {os.path.getsize(DATA)/1e6:.0f} MB')

# Install mnemonics (fresh clone must carry the --trust-gate-ce flag)
print('\n=== mnemonics kuruluyor ===')
WD = f'{BASE}/mnemonics'
if not os.path.isdir(WD):
    subprocess.run(['git','clone','--depth','1',
                    'https://github.com/nakata-app/mnemonics.git', WD], check=True)
subprocess.run(['git','-C',WD,'log','--oneline','-1'], check=True)
with open(f'{WD}/benchmarks/longmemeval_eval.py') as f:
    if '--trust-gate-ce' not in f.read():
        print('!!! clone eski: --trust-gate-ce yok (push HEAD bekleniyor)'); sys.exit(4)
subprocess.run(f'pip install -q -e {WD} sentence-transformers numpy adaptmem 2>&1 | tail -2',
               shell=True, check=True)

# Common config (locked 0.954 champion)
R = f'{BASE}/results'
os.makedirs(R, exist_ok=True)
base_env = {**os.environ, 'LME_DATA': DATA, 'PYTHONUNBUFFERED': '1'}
base_cmd = ['python', 'benchmarks/longmemeval_eval.py',
            '--n', '500', '--mode', 'rerank',
            '--chunk-mode', 'turn', '--temporal-aware',
            '--candidate-k', '50', '--seed', '42']

# --- A: champion baseline ---
print('\n' + '='*70)
print('A: champion 0.954 config (ms-marco CE, no gate)')
print('='*70)
t0 = time.time()
subprocess.run(base_cmd + ['--out', f'{R}/A.json',
                           '--per-q-out', f'{R}/A_perq.json'],
               cwd=WD, env=base_env, check=True)
A = json.load(open(f'{R}/A.json'))['mnemonics_rerank']
print(f"\nA  R@1={A['R@1']:.3f}  R@5={A['R@5']:.3f}  R@10={A['R@10']:.3f}  ({time.time()-t0:.0f}s)")

# --- B: champion + trust-gated chat-ce-v3 ---
# m=1.0 (Sprint 4 degeri) 0/500 ateşledi: chat-ce-v3'ün düzelttiği soruların
# margin'i 0.97-0.999 bandında, eşiğin kıl payı altında kalıyordu. Per-q sweep
# 0.05-0.30 platosunda 13 fire / 8 help / 1 hurt (est 0.966) gösterdi; 0.2 =
# platonun ortası. Yarı/yarı split testinde de tuttu (held-out est 0.968).
MARGIN = 0.2
print('\n' + '='*70)
print(f'B: champion + trust-gate(chat-ce-v3, margin={MARGIN})')
print('='*70)
t0 = time.time()
subprocess.run(base_cmd + ['--trust-gate-ce', GATE_CE,
                           '--trust-gate-margin', str(MARGIN),
                           '--out', f'{R}/B.json',
                           '--per-q-out', f'{R}/B_perq.json'],
               cwd=WD, env=base_env, check=True)
B = json.load(open(f'{R}/B.json'))['mnemonics_rerank']
print(f"\nB  R@1={B['R@1']:.3f}  R@5={B['R@5']:.3f}  R@10={B['R@10']:.3f}  ({time.time()-t0:.0f}s)")

# --- Flip analysis ---
ap = {q['qid']: q for q in json.load(open(f'{R}/A_perq.json'))}
bp = {q['qid']: q for q in json.load(open(f'{R}/B_perq.json'))}
gain = sum(1 for q in ap if not ap[q]['hit@1'] and bp[q]['hit@1'])
loss = sum(1 for q in ap if ap[q]['hit@1'] and not bp[q]['hit@1'])

print('\n' + '='*70)
print('SUMMARY — trust-gated chat-ce-v3 vs champion')
print('='*70)
print(f'{"Run":<42} {"R@1":>7} {"R@5":>7} {"R@10":>7}')
print(f'{"A (champion, ms-marco)":<42} {A["R@1"]:>7.3f} {A["R@5"]:>7.3f} {A["R@10"]:>7.3f}')
print(f'{f"B (champion + trust gate m={MARGIN})":<42} {B["R@1"]:>7.3f} {B["R@5"]:>7.3f} {B["R@10"]:>7.3f}')
print(f'\ndelta R@1: {B["R@1"]-A["R@1"]:+.4f}   helped={gain}  hurt={loss}')
print(f'gate fired: {B.get("trust_gate", {}).get("fired", "?")}/500')

print('\nper bucket:')
print(f'  {"bucket":<26} {"A_R@1":>6} {"B_R@1":>6}    {"delta":>7}')
for t in sorted(set(x['qtype'] for x in ap.values())):
    qa = [x for x in ap.values() if x['qtype']==t]
    ra = sum(x['hit@1'] for x in qa)/len(qa)
    rb = sum(bp[x['qid']]['hit@1'] for x in qa)/len(qa)
    print(f'  {t:<26} {ra:.3f}  {rb:.3f}    {rb-ra:+.4f}')

# --- Offline margin sweep ESTIMATE (gate fields only; ignores the
# temporal-aware stage that runs after the gate, so treat as diagnostic,
# not a headline number — re-run at the best margin to confirm) ---
print('\nmargin sweep (ESTIMATE from B per-q gate fields):')
print(f'  {"margin":>7} {"est_R@1":>8} {"would_fire":>10}')
for m in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
    hit = fire = 0
    for q in bp.values():
        g = q.get('gate') or {}
        gold = set(q.get('answer_sids') or [])
        base = g.get('base_top1_sid')
        pick = g.get('ftce_best_sid')
        marg = g.get('ftce_margin')
        top1 = base
        if marg is not None and pick and pick != base and marg >= m:
            top1 = pick; fire += 1
        hit += 1 if top1 in gold else 0
    print(f'  {m:>7.2f} {hit/len(bp):>8.3f} {fire:>10}')

out_bundle = {
    'A_champion': A,
    f'B_trust_gate_m{MARGIN}': B,
    'delta_R@1': B['R@1'] - A['R@1'],
    'helped': gain, 'hurt': loss,
    'gate_ce': GATE_CE,
}
with open(f'{R}/trustgate_bridge_summary.json', 'w') as f:
    json.dump(out_bundle, f, indent=2)
print(f'\nbundle: {R}/trustgate_bridge_summary.json')
