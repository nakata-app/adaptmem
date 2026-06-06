"""LoCoMo headroom analizi — para nereye gitmeli?

Her yanlis cevap icin: kanit baglamdaydi ama cevap yanlis (CEVAPLAYICI kolu)
mi, kanit baglamda yoktu (RETRIEVAL kolu) mu? run6'nin locomo_contexts.json +
locomo_verdicts.json ciktilarindan, API'siz/bedava hesaplanir.

Kullanim: python locomo_headroom.py [run_adi]   (varsayilan: locomo_e2e_run6)
"""
import json, sys
from collections import defaultdict

RUN = sys.argv[1] if len(sys.argv) > 1 else 'locomo_e2e_run6'
BASE = f'/Users/macmini/Projects/adaptmem/results/{RUN}/results'
DATA = '/Users/macmini/Projects/_competitors/locomo-orig/data/locomo10.json'

contexts = json.load(open(f'{BASE}/locomo_contexts.json'))
verdicts = json.load(open(f'{BASE}/locomo_verdicts.json'))
data = json.load(open(DATA))
assert len(contexts) == len(verdicts), 'context/verdict sayisi uyusmuyor'

# kernel'deki damgalama birebir: '[{dt}] {spk}: {txt}' (+caption augment)
dia2text, qa_ev = {}, {}
for conv in data:
    sid = conv['sample_id']
    for key in conv['conversation']:
        if not key.startswith('session_') or key.endswith('_date_time'):
            continue
        dt = conv['conversation'].get(f'{key}_date_time', '')
        turns = conv['conversation'][key]
        if not isinstance(turns, list):
            continue
        for t in turns:
            if not isinstance(t, dict):
                continue
            txt = t.get('text', '')
            cap = t.get('blip_caption') or ''
            if cap:
                txt = (txt + ' ' if txt else '') + f'[shares photo: {cap}]'
            if txt:
                dia2text[(sid, t.get('dia_id',''))] = f"[{dt}] {t.get('speaker','')}: {txt}"
    for qa in conv['qa']:
        qa_ev[(sid, qa['question'])] = qa.get('evidence') or []

def ev_in_ctx(sid, ev, rows):
    """Kanit dia'si baglamda mi: fact'in DIA= prefix'inde VEYA ham satir metni."""
    fact_dias = set()
    raw_texts = set()
    for t in rows:
        if t.startswith('DIA=') and '|' in t:
            fact_dias |= set(t.split('DIA=', 1)[1].split('|', 1)[0].split('+'))
        else:
            raw_texts.add(t)
    return {d: (d in fact_dias or dia2text.get((sid, d), '\x00') in raw_texts)
            for d in ev}

# kesisim: verdict[i] ile context[i] ayni soru (ayni dongu sirasi)
agg = defaultdict(lambda: defaultdict(int))
for c, v in zip(contexts, verdicts):
    assert c['question'] == v['question'], 'sira kaymis'
    sid = c['sample_id']
    ev = qa_ev.get((sid, c['question']), [])
    cat = v['category']
    if not ev:
        agg[cat]['no_ev'] += 1
        continue
    cov = ev_in_ctx(sid, ev, c['ctx_a'] + c['ctx_b'])
    full = all(cov.values())
    any_ = any(cov.values())
    key = ('ok' if v['verdict'] else 'wrong') + ('_cov' if full else ('_part' if any_ else '_miss'))
    agg[cat][key] += 1
    agg['TOPLAM'][key] += 1

def row(d):
    n = sum(cnt for k, cnt in d.items() if k != 'no_ev')
    w_cov, w_part, w_miss = d['wrong_cov'], d['wrong_part'], d['wrong_miss']
    o_cov = d['ok_cov']
    cov_n = w_cov + o_cov
    return (n, w_cov, w_part, w_miss,
            (o_cov / cov_n) if cov_n else 0.0)

print(f'run={RUN}  (kanitli soru bazinda; no_ev haric)')
print(f"{'kategori':>9} {'n':>5} | {'YANLIS kanit-tam':>17} {'kanit-kismi':>12} {'kanit-yok':>10} | {'acc(kanit-tam)':>14}")
names = {'1': 'multi', '2': 'temporal', '3': 'open', '4': 'single'}
for cat in sorted(agg, key=lambda c: (c == 'TOPLAM', c)):
    n, wc, wp, wm, acc_cov = row(agg[cat])
    label = names.get(cat, cat)
    print(f'{label:>9} {n:>5} | {wc:>17} {wp:>12} {wm:>10} | {acc_cov:>14.3f}')
print()
t = agg['TOPLAM']
total_wrong = t['wrong_cov'] + t['wrong_part'] + t['wrong_miss']
print(f"toplam yanlis (kanitli): {total_wrong}")
print(f"  CEVAPLAYICI kolu (kanit tam baglamda, cevap yine yanlis): {t['wrong_cov']} ({100*t['wrong_cov']/total_wrong:.0f}%)")
print(f"  RETRIEVAL kolu (kanit kismen/hic yok):                    {t['wrong_part']+t['wrong_miss']} ({100*(t['wrong_part']+t['wrong_miss'])/total_wrong:.0f}%)")
