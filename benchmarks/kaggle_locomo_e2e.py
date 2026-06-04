"""LoCoMo uctan-uca + retrieval R@k — Kaggle kerneli (mnemonics).

Mem0'in eval protokolune uyumlu (skorlama icin onlarin evals.py sekli):
her QA icin iki speaker namespace'inden retrieve -> DeepSeek chat cevaplar.
Ayni geciste BEDAVA retrieval metrigi: recall_any@k (kanit dia_id'si
merged top-k'da mi). Kategori 5 (cevapsiz/celdirici) standart geregi haric.

Dayaniklilik: her 200 soruda partial JSON yazilir (sonuc-kaybi yok).
Key: __DEEPSEEK_KEY__ gonderim aninda enjekte edilir, git'e girmez.
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
R = f'{BASE}/results'; os.makedirs(R, exist_ok=True)

# mnemonics
WD = f'{BASE}/mnemonics'
if not os.path.isdir(WD):
    subprocess.run(['git','clone','--depth','1',
                    'https://github.com/nakata-app/mnemonics.git', WD], check=True)
subprocess.run(['git','-C',WD,'log','--oneline','-1'], check=True)
subprocess.run(f'pip install -q -e {WD} sentence-transformers openai 2>&1 | tail -2',
               shell=True, check=True)
sys.path.insert(0, WD)
from mnemonics.store import Store
from mnemonics.ingest import ingest
from mnemonics.retrieve import retrieve
print('mnemonics import OK')

# LoCoMo verisi (Kaggle dataset: atakanakbaba/locomo)
cands = glob.glob('/kaggle/input/*/locomo10.json') + glob.glob('/kaggle/input/*/*/locomo10.json')
if not cands:
    print('!!! locomo10.json yok (krun --dataset atakanakbaba/locomo ?)'); sys.exit(3)
DATA = cands[0]
data = json.load(open(DATA))
print(f'locomo: {DATA}, {len(data)} conversation')

from openai import OpenAI
client = OpenAI(api_key="__DEEPSEEK_KEY__", base_url="https://api.deepseek.com/v1")

TOP_K, CAND_K = 30, 50
results = {"mnemonics": []}
ks = (1, 5, 10)
rhits = {k: 0 for k in ks}
rby = {}
n_r = 0
t_start = time.time()

def save_partial():
    json.dump(results, open(f'{R}/locomo_answers.json', 'w'), indent=1)
    retr = {
        'n': n_r,
        **{f'R@{k}': round(rhits[k]/max(n_r,1), 4) for k in ks},
        'by_category': {c: {'n': v['n'],
                            **{f'R@{k}': round(v[k]/max(v['n'],1), 4) for k in ks}}
                        for c, v in sorted(rby.items())},
    }
    json.dump(retr, open(f'{R}/locomo_retrieval.json', 'w'), indent=1)
    return retr

for conv in data:
    sid = conv['sample_id']
    spk_a = conv['conversation']['speaker_a']
    spk_b = conv['conversation']['speaker_b']
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        store = Store(path=f'{td}/m.db')
        texts_a, meta_a, texts_b, meta_b = [], [], [], []
        text2dia = {}
        for key in conv['conversation']:
            if not key.startswith('session_') or key.endswith('_date_time'):
                continue
            dt = conv['conversation'].get(f'{key}_date_time', '')
            turns = conv['conversation'][key]
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                spk, txt = turn.get('speaker',''), turn.get('text','')
                dia = turn.get('dia_id','')
                if not txt:
                    continue
                stamped = f'[{dt}] {spk}: {txt}'
                text2dia.setdefault(stamped, set()).add(dia)
                m = {'ts': dt, 'dia_id': dia}
                if spk == spk_a:   texts_a.append(stamped); meta_a.append(m)
                elif spk == spk_b: texts_b.append(stamped); meta_b.append(m)
        if texts_a: ingest(texts=texts_a, store=store, ns=f'loc_{sid}_a', meta=meta_a)
        if texts_b: ingest(texts=texts_b, store=store, ns=f'loc_{sid}_b', meta=meta_b)

        def dias_of(row):
            m = row.get('meta') or {}
            if isinstance(m, str):
                try: m = json.loads(m)
                except Exception: m = {}
            d = m.get('dia_id')
            if d: return {d}
            return text2dia.get(row.get('text',''), set())

        for qa in conv['qa']:
            if str(qa.get('category')) == '5':
                continue
            q = qa['question']; gt = qa['answer']
            hits_a = retrieve(query=q, store=store, ns=f'loc_{sid}_a',
                              top_k=TOP_K, candidate_k=CAND_K, rerank=True)
            hits_b = retrieve(query=q, store=store, ns=f'loc_{sid}_b',
                              top_k=TOP_K, candidate_k=CAND_K, rerank=True)
            ra, rb = hits_a.get('results', []), hits_b.get('results', [])

            # --- retrieval R@k: iki listeyi skora gore birlestir ---
            merged = sorted(ra + rb, key=lambda r: r.get('score', r.get('ce_score', 0.0)),
                            reverse=True)
            ev = set(qa.get('evidence') or [])
            if ev:
                seen = []
                for r in merged:
                    for d in dias_of(r):
                        if d not in seen: seen.append(d)
                n_r += 1
                cat = str(qa.get('category'))
                b = rby.setdefault(cat, {'n': 0, 1: 0, 5: 0, 10: 0})
                b['n'] += 1
                for k in ks:
                    if any(d in ev for d in seen[:k]):
                        rhits[k] += 1; b[k] += 1

            # --- DeepSeek cevap ---
            mem_a = '\n'.join(f"- {r['text']}" for r in ra)
            mem_b = '\n'.join(f"- {r['text']}" for r in rb)
            prompt = (
                'You answer questions from conversation memories. Timestamps are in\n'
                'brackets. Convert relative time references to actual dates using the\n'
                'memory timestamps. Answer in 5-6 words max.\n\n'
                f'Memories {spk_a}:\n{mem_a}\n\nMemories {spk_b}:\n{mem_b}\n\n'
                f'Question: {q}\nAnswer:')
            answer = 'ERROR'
            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model='deepseek-chat',
                        messages=[{'role': 'user', 'content': prompt}],
                        max_tokens=60, temperature=0)
                    answer = resp.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if attempt < 4:
                        time.sleep(8 * (attempt + 1))
                    else:
                        answer = f'ERROR: {e}'
            results['mnemonics'].append({
                'sample_id': sid, 'question': q, 'answer': gt,
                'response': answer, 'category': str(qa.get('category'))})
            if len(results['mnemonics']) % 200 == 0:
                retr = save_partial()
                el = time.time() - t_start
                print(f"  [{len(results['mnemonics'])}] R@1={retr['R@1']} R@5={retr['R@5']} "
                      f"R@10={retr['R@10']}  {el:.0f}s", flush=True)
    print(f'{sid} bitti  (cum n={len(results["mnemonics"])})', flush=True)

retr = save_partial()
errs = sum(1 for r in results['mnemonics'] if str(r['response']).startswith('ERROR'))
print('\n' + '='*66)
print(f"LOCOMO RETRIEVAL (recall_any@k, kanit dia_id, kategori 1-4)")
print('='*66)
print(json.dumps(retr, indent=1))
print(f"\nanswers: {len(results['mnemonics'])} (ERROR={errs}) -> locomo_answers.json")
print('skorlama: Mem0 evals.py sekline uygun; judge ayri adimda.')
