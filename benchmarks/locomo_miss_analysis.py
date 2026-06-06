"""LoCoMo retrieval-headroom derin analizi — para harcamadan, kayitli veriden.

Kanit-eksik yanlislari (cevaplayici degil, RETRIEVAL sucu) parcalar:
hangi kategori, kismi-kanit mi hic-kanit mi, cok-adimlida kanitin % kaci
geliyor, kacan kanit fact olarak mi ham-turn olarak mi gelmeli/gelmis.
Amac: retrieval kolunu (Kaggle ~$1) kosmadan ONCE 'neyi duzeltirsek en cok
kazaniriz' sorusunu cevaplamak.

Kullanim: python locomo_miss_analysis.py [run_adi]   (varsayilan run6)
"""
import json, sys
from collections import defaultdict, Counter

RUN = sys.argv[1] if len(sys.argv) > 1 else 'locomo_e2e_run6'
BASE = f'/Users/macmini/Projects/adaptmem/results/{RUN}/results'
DATA = '/Users/macmini/Projects/_competitors/locomo-orig/data/locomo10.json'

contexts = json.load(open(f'{BASE}/locomo_contexts.json'))
verdicts = json.load(open(f'{BASE}/locomo_verdicts.json'))
data = json.load(open(DATA))

# headroom.py ile birebir ayni dia2text + evidence kurulumu
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

def ev_status(sid, ev, rows):
    """Her evidence dia icin durum: 'fact' (fis olarak bagamda),
    'raw' (ham-turn olarak baglamda), 'miss' (baglamda yok)."""
    fact_dias, raw_texts = set(), set()
    for t in rows:
        if t.startswith('DIA=') and '|' in t:
            fact_dias |= set(t.split('DIA=', 1)[1].split('|', 1)[0].split('+'))
        else:
            raw_texts.add(t)
    out = {}
    for d in ev:
        if d in fact_dias:
            out[d] = 'fact'
        elif dia2text.get((sid, d), '\x00') in raw_texts:
            out[d] = 'raw'
        else:
            out[d] = 'miss'
    return out

names = {'1': 'multi-hop', '2': 'temporal', '3': 'open', '4': 'single'}

# Sadece kanit-eksik YANLISLAR (retrieval sucu): en az bir evidence 'miss'
miss_cat = Counter()          # kategori dagilimi
part_vs_full_miss = Counter() # 'part' (bazi var) vs 'allmiss' (hicbiri yok)
hop_coverage = defaultdict(list)  # kategori -> [kapsanan evidence orani]
src_of_covered = Counter()    # gelen evidence'lar fact mi raw mi
miss_evidence_total = 0
examples = defaultdict(list)

for c, v in zip(contexts, verdicts):
    if v['verdict']:            # dogru cevap -> retrieval sorunu degil
        continue
    sid = c['sample_id']
    ev = qa_ev.get((sid, c['question']), [])
    if not ev:
        continue
    st = ev_status(sid, ev, c['ctx_a'] + c['ctx_b'])
    n_miss = sum(1 for s in st.values() if s == 'miss')
    if n_miss == 0:
        continue               # kanit tam geldi -> cevaplayici sucu, atla
    cat = v['category']
    miss_cat[cat] += 1
    covered = sum(1 for s in st.values() if s != 'miss')
    hop_coverage[cat].append(covered / len(ev))
    part_vs_full_miss['allmiss' if covered == 0 else 'part'] += 1
    miss_evidence_total += n_miss
    for s in st.values():
        if s != 'miss':
            src_of_covered[s] += 1
    if len(examples[cat]) < 3:
        examples[cat].append((c['question'][:60], len(ev), covered, n_miss))

# --- kismi-kanit vakalarinda: kacan evidence, gelen evidence ile ayni
# oturumda mi? dia_id format 'D<session>:<turn>' -> session = 'D<n>'.
# Ayni-oturum yuksekse komsu/oturum-ici genisleme ise yarar (ucuz fix);
# farkli-oturum yuksekse cross-session ikinci-tur retrieve gerek (pahali).
def sess_of(dia):
    return dia.split(':', 1)[0] if ':' in dia else dia
same_sess_miss = cross_sess_miss = 0
for c, v in zip(contexts, verdicts):
    if v['verdict']:
        continue
    sid = c['sample_id']
    ev = qa_ev.get((sid, c['question']), [])
    if not ev:
        continue
    st = ev_status(sid, ev, c['ctx_a'] + c['ctx_b'])
    covered_sess = {sess_of(d) for d, s in st.items() if s != 'miss'}
    if not covered_sess:
        continue  # tam-kayip; komsu-genisleme dayanacak gelen kanit yok
    for d, s in st.items():
        if s == 'miss':
            if sess_of(d) in covered_sess:
                same_sess_miss += 1
            else:
                cross_sess_miss += 1

total = sum(miss_cat.values())
print(f'=== run={RUN}  RETRIEVAL-sucu yanlislar (>=1 evidence miss): {total} ===\n')
print(f"{'kategori':>11} {'yanlis':>6} {'pay':>5} | {'ort.kapsama':>11} | ornek (soru | #ev kapsanan miss)")
for cat in sorted(miss_cat, key=lambda c: -miss_cat[c]):
    cov = sum(hop_coverage[cat]) / len(hop_coverage[cat])
    print(f"{names.get(cat,cat):>11} {miss_cat[cat]:>6} {100*miss_cat[cat]/total:>4.0f}% | {cov:>11.2f} |")
    for q, ne, cv, nm in examples[cat]:
        print(f"{'':>26} | {'':>11} |   {q!r}  ev={ne} kapsanan={cv} miss={nm}")

print(f"\n--- kismi mi tam-kayip mi ---")
print(f"  kismi-kanit (bazi evidence geldi, bazi kacti): {part_vs_full_miss['part']}")
print(f"  tam-kayip   (hicbir evidence gelmedi):         {part_vs_full_miss['allmiss']}")
print(f"\n--- gelen (kacmAYAN) evidence kaynagi ---")
print(f"  fact (fis olarak geldi): {src_of_covered['fact']}")
print(f"  raw  (ham-turn olarak):  {src_of_covered['raw']}")
print(f"\ntoplam kacan evidence dia: {miss_evidence_total}")
print(f"\n--- kismi-kanit vakalarinda kacan evidence'in oturumu (komsu-genisleme tavani) ---")
tot_ss = same_sess_miss + cross_sess_miss
print(f"  ayni oturum (gelen kanitla):  {same_sess_miss}  ({100*same_sess_miss/max(tot_ss,1):.0f}%) -> ucuz fix: oturum-ici/komsu genisleme")
print(f"  farkli oturum:                {cross_sess_miss}  ({100*cross_sess_miss/max(tot_ss,1):.0f}%) -> pahali: cross-session ikinci-tur retrieve")
print(f"\nYORUM ipucu: kismi-kanit yuksekse -> multi-hop ikinci-hop genislemesi en umutlu.")
print(f"tam-kayip yuksekse -> ingest/extraction o turn'u hic uretmemis (recall taban sorunu).")
