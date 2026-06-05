"""Mnemonics temporal-v2 A/B — Kaggle kerneli.

A: mevcut sampiyon 0.970 (mn-ce-v1 gate m=0.2 pin=0.5)
B: A + --temporal-v2  (3 duzeltme: sahte 1-gun penceresi, asc/desc yon
   bug'i, ordinal sortu top-5 alakali adayla sinirlama)

Hedef: 4 ordinal/sayim fail'i (+2-4 beklenir); flip analizi top-5
kapsam daraltmasinin eski kazanimlari bozup bozmadigini gosterir.
"""
import os, sys, subprocess, json, time, glob
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

GATE_CE = None
for cand in glob.glob('/kaggle/input/*/'):
    if (os.path.exists(os.path.join(cand, 'model.safetensors'))
            and os.path.exists(os.path.join(cand, 'config.json'))):
        GATE_CE = cand.rstrip('/')
        break
print(f'gate CE (mn-ce-v1): {GATE_CE}')
if GATE_CE is None:
    print('!!! mn-ce-v1 dataset yok'); sys.exit(3)

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

WD = f'{BASE}/mnemonics'
if not os.path.isdir(WD):
    subprocess.run(['git','clone','--depth','1',
                    'https://github.com/nakata-app/mnemonics.git', WD], check=True)
subprocess.run(['git','-C',WD,'log','--oneline','-1'], check=True)
if '--temporal-v2' not in open(f'{WD}/benchmarks/longmemeval_eval.py').read():
    print('!!! clone eski (--temporal-v2 yok)'); sys.exit(4)
subprocess.run(f'pip install -q -e {WD} sentence-transformers numpy adaptmem 2>&1 | tail -2',
               shell=True, check=True)

R = f'{BASE}/results'; os.makedirs(R, exist_ok=True)
env = {**os.environ, 'LME_DATA': DATA, 'PYTHONUNBUFFERED': '1'}
champ = ['python', 'benchmarks/longmemeval_eval.py',
         '--n', '500', '--mode', 'rerank',
         '--chunk-mode', 'turn', '--temporal-aware',
         '--candidate-k', '50', '--seed', '42',
         '--trust-gate-ce', GATE_CE,
         '--trust-gate-margin', '0.2',
         '--trust-gate-pin-margin', '0.5']

def run_eval(tag, extra):
    print(f'\n=== EVAL {tag} ===')
    t0 = time.time()
    subprocess.run(champ + extra + ['--out', f'{R}/{tag}.json',
                                    '--per-q-out', f'{R}/{tag}_perq.json'],
                   cwd=WD, env=env, check=True)
    d = json.load(open(f'{R}/{tag}.json'))['mnemonics_rerank']
    print(f"{tag}  R@1={d['R@1']:.3f} R@5={d['R@5']:.3f} R@10={d['R@10']:.3f} ({time.time()-t0:.0f}s)")
    return d

A = run_eval('A', [])
B = run_eval('B', ['--temporal-v2'])

ap = {q['qid']: q for q in json.load(open(f'{R}/A_perq.json'))}
bp = {q['qid']: q for q in json.load(open(f'{R}/B_perq.json'))}
gain = [q for q in ap if not ap[q]['hit@1'] and bp[q]['hit@1']]
loss = [q for q in ap if ap[q]['hit@1'] and not bp[q]['hit@1']]

print('\n' + '='*70)
print('SUMMARY — temporal-v2 vs champion (ikisi de mn-ce-v1 gate)')
print('='*70)
print(f"A (champion 0.970 config)  R@1={A['R@1']:.4f} R@5={A['R@5']:.4f} R@10={A['R@10']:.4f}")
print(f"B (A + temporal-v2)        R@1={B['R@1']:.4f} R@5={B['R@5']:.4f} R@10={B['R@10']:.4f}")
print(f"delta R@1: {B['R@1']-A['R@1']:+.4f}   helped={len(gain)} hurt={len(loss)}")
for q in gain: print(f"  HELP {q[:24]} {ap[q]['qtype']}")
for q in loss: print(f"  HURT {q[:24]} {ap[q]['qtype']}")
print('\nper bucket R@1:')
from collections import defaultdict
bt = defaultdict(list)
for q in ap: bt[ap[q]['qtype']].append(q)
for t, qs in sorted(bt.items()):
    ra = sum(ap[q]['hit@1'] for q in qs)/len(qs)
    rb = sum(bp[q]['hit@1'] for q in qs)/len(qs)
    print(f'  {t:<26} {ra:.3f} -> {rb:.3f}  ({rb-ra:+.4f})')

json.dump({'A': A, 'B_temporal_v2': B, 'helped': gain, 'hurt': loss},
          open(f'{R}/temporalv2_summary.json', 'w'), indent=2)
print(f'\nbundle: {R}/temporalv2_summary.json')
