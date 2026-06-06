"""LoCoMo uctan-uca + retrieval R@k — Kaggle kerneli (mnemonics).

Mem0'in eval protokolune uyumlu (skorlama icin onlarin evals.py sekli):
her QA icin iki speaker namespace'inden retrieve -> DeepSeek chat cevaplar.
Ayni geciste BEDAVA retrieval metrigi: recall_any@k (kanit dia_id'si
merged top-k'da mi). Kategori 5 (cevapsiz/celdirici) standart geregi haric.

Dayaniklilik: her 200 soruda partial JSON yazilir (sonuc-kaybi yok).
Key: __DEEPSEEK_KEY__ gonderim aninda enjekte edilir, git'e girmez.

run5 kolu (DEDUP_ANSWER_CTX): fact + ham turn ayni kaniti tasiyinca cevap
baglaminda tekrar oluyor (run4'te multi-hop -1.4pp dustu, fact sayisi artinca).
Cozum: skor sirasinda yuru, dia kapsamasi tamamen onceden-kapsanmis satiri
atla. SADECE cevap baglamina uygulanir; retrieval R@k ham listeden olculur
(run4 ile birebir kiyas icin).
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
if not os.path.exists(f'{WD}/mnemonics/extract.py'):
    print('!!! clone eski (extract.py yok)'); sys.exit(4)
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

# --- S2: fact-extraction katmani (opsiyonel; provenance'li atomik fact'ler) ---
EXTRACT_FACTS = True
from mnemonics.extract import FactExtractor
# S3: 250 butcesi conv-50'nin 22 session'ini fissiz birakmisti; ~190 session
# + retry payi icin 400 yeterli ve tasma riski dusuk.
extractor = FactExtractor(client=client, model='deepseek-chat',
                          max_calls=400, max_facts=25)

TOP_K, CAND_K = 30, 50
# run5 OLCULDU: dedup NEGATIF (0.8214 -> 0.8078; temporal -3.4pp, single-hop
# -1.8pp, open-domain +4.2pp). Otopsi: fact, ham turn'u listeden dusurunce
# ham turn'un '[tarih] konusmaci:' damgasi da gidiyor; cevaplayici tarih
# hesabini o damgadan yapiyordu. Fact'lerde tarih damgasi yok (prozda bazen
# var, tutarsiz). Ders: dedup ancak fact'ler tarih-damgali olursa denenebilir.
DEDUP_ANSWER_CTX = False  # kapali; sampiyon davranis = run4
dedup_kept = dedup_skipped = 0
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
                # Resim icerigi: kanitlarin %23.7'si resimli turn'lerde (run1
                # bulgusu). blip_caption'i metne ekle ki "neyin fotografi"
                # sorulari kor nokta olmasin; resim-only turn'ler de girsin.
                cap = turn.get('blip_caption') or ''
                if cap:
                    txt = (txt + ' ' if txt else '') + f'[shares photo: {cap}]'
                if not txt:
                    continue
                stamped = f'[{dt}] {spk}: {txt}'
                text2dia.setdefault(stamped, set()).add(dia)
                m = {'ts': dt, 'dia_id': dia}
                if spk == spk_a:   texts_a.append(stamped); meta_a.append(m)
                elif spk == spk_b: texts_b.append(stamped); meta_b.append(m)
        # Fact-extraction: session basina LLM damitmasi; fact, kaynak turn'un
        # konusmacisinin namespace'ine girer (karisik kaynak -> iki ns birden).
        # Fact metni DIA=<id+id>| prefix'iyle saklanir ki retrieval R@k
        # skorlamasi fact uzerinden kanita ulasmayi da saysin.
        if EXTRACT_FACTS:
            spk_of = {}
            sess_groups = {}
            for key in conv['conversation']:
                if not key.startswith('session_') or key.endswith('_date_time'):
                    continue
                turns = conv['conversation'][key]
                if not isinstance(turns, list):
                    continue
                dt = conv['conversation'].get(f'{key}_date_time', '')
                tl = []
                for t in turns:
                    if isinstance(t, dict) and t.get('text') and t.get('dia_id'):
                        txt = t['text']
                        cap = t.get('blip_caption') or ''
                        if cap:
                            txt = f'{txt} [shares photo: {cap}]'
                        tl.append({'id': t['dia_id'], 'speaker': t.get('speaker','?'),
                                   'text': txt})
                        spk_of[t['dia_id']] = t.get('speaker','')
                if tl:
                    sess_groups[key] = (tl, dt)
            f_texts_a, f_meta_a, f_texts_b, f_meta_b = [], [], [], []
            for key, (tl, dt) in sess_groups.items():
                try:
                    facts = extractor.extract_session(tl, session_date=dt)
                except RuntimeError as e:
                    print(f'  !!! extraction budget: {e}'); facts = []
                for f in facts:
                    stamped = f"DIA={'+'.join(f['source_ids'])}|[fact] {f['text']}"
                    m = {'kind': 'fact', 'dia_id': f['source_ids'][0]}
                    spks = {spk_of.get(s,'') for s in f['source_ids']}
                    if spk_a in spks or not (spks & {spk_a, spk_b}):
                        f_texts_a.append(stamped); f_meta_a.append(m)
                    if spk_b in spks:
                        f_texts_b.append(stamped); f_meta_b.append(m)
            texts_a += f_texts_a; meta_a += f_meta_a
            texts_b += f_texts_b; meta_b += f_meta_b
            print(f'{sid}: fact_a={len(f_texts_a)} fact_b={len(f_texts_b)} '
                  f'(calls={extractor.stats["calls"]})', flush=True)

        if texts_a: ingest(texts=texts_a, store=store, ns=f'loc_{sid}_a', meta=meta_a)
        if texts_b: ingest(texts=texts_b, store=store, ns=f'loc_{sid}_b', meta=meta_b)

        def dias_of(row):
            t = row.get('text', '')
            if t.startswith('DIA=') and '|' in t:
                return set(t.split('DIA=', 1)[1].split('|', 1)[0].split('+'))
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

            # --- DeepSeek cevap (Mem0-tarzi tam template; run1'deki kompakt
            # prompt temporal'i 0.455'e dusurmus olabilirdi) ---
            def _dedup_rows(rows):
                """Skor sirasinda yuru; dia kapsamasi onceden tamamen
                kapsanmis satirlari atla (fact/ham tekrar temizligi).
                Dia'si bilinmeyen satir her zaman kalir."""
                global dedup_kept, dedup_skipped
                kept, covered = [], set()
                for r in rows:
                    ds = dias_of(r)
                    if ds and ds <= covered:
                        dedup_skipped += 1
                        continue
                    kept.append(r); covered |= ds
                    dedup_kept += 1
                return kept

            ctx_a = _dedup_rows(ra) if DEDUP_ANSWER_CTX else ra
            ctx_b = _dedup_rows(rb) if DEDUP_ANSWER_CTX else rb
            mem_a = '\n'.join(f"- {r['text']}" for r in ctx_a)
            mem_b = '\n'.join(f"- {r['text']}" for r in ctx_b)
            prompt = (
                'You are an intelligent memory assistant tasked with retrieving\n'
                'accurate information from conversation memories.\n\n'
                '# INSTRUCTIONS:\n'
                '1. Carefully analyze all provided memories from both speakers\n'
                '2. Pay special attention to the timestamps to determine the answer\n'
                '3. If the question asks about a specific event or fact, look for direct\n'
                '   evidence in the memories\n'
                '4. If the memories contain contradictory information, prioritize the\n'
                '   most recent memory\n'
                '5. For questions about time references (like "last year", "two months\n'
                '   ago"), calculate the actual date based on the memory timestamp\n'
                '6. Always convert relative time references to specific dates, months,\n'
                '   or years\n'
                '7. Focus only on the content of the memories. Do not confuse character\n'
                '   names mentioned in memories with the actual speakers\n'
                '8. The answer should be less than 5-6 words.\n\n'
                f'Memories for user {spk_a}:\n\n{mem_a}\n\n'
                f'Memories for user {spk_b}:\n\n{mem_b}\n\n'
                f'Question: {q}\n\nAnswer:')
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
if EXTRACT_FACTS:
    print(f"extraction stats: {extractor.stats}")
    json.dump(extractor.stats, open(f'{R}/extraction_stats.json', 'w'), indent=1)
if DEDUP_ANSWER_CTX:
    tot = dedup_kept + dedup_skipped
    print(f"RUN=run5-dedup  ctx dedup: kept={dedup_kept} skipped={dedup_skipped} "
          f"({100*dedup_skipped/max(tot,1):.1f}% atildi)")
print('skorlama: Mem0 evals.py sekline uygun; judge ayri adimda.')
