"""LoCoMo cevaplayici-degisim deneyi — kayitli baglamlar uzerinde yerel replay.

run6'nin locomo_contexts.json'i uzerinde AYNI prompt'la farkli bir answerer
calistirir (retrieval sabit, sadece cevap katmani degisir), inline judge'lar
ve run6 verdicts ile soru-bazli flip tablosu cikarir. Kaggle gerekmez.

Kullanim: python locomo_answerer_swap.py [--model deepseek-v4-pro]
          [--run locomo_e2e_run6] [--stride 4] [--max-tokens 2000]
stride=4 -> her 4. soru (385/1540, kategori karisimi korunur). stride=1 -> tam.
"""
import json, os, re, sys, time, argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='deepseek-v4-pro')
ap.add_argument('--run', default='locomo_e2e_run6')
ap.add_argument('--stride', type=int, default=4)
ap.add_argument('--max-tokens', type=int, default=2000)
ap.add_argument('--base-url', default='https://api.deepseek.com/v1')
ap.add_argument('--key-env', default='DEEPSEEK_API_KEY')
ap.add_argument('--reuse-from', default=None,
                help='onceki swap json\'u; ayni modelin odenmis cevap+verdict\'leri atlanir')
args = ap.parse_args()

BASE = f'/Users/macmini/Projects/adaptmem/results/{args.run}/results'
DATA = '/Users/macmini/Projects/_competitors/locomo-orig/data/locomo10.json'
# 2026-06-06 api-docs.deepseek.com/quick_start/pricing ($/1M token, cache-miss)
PRICE = {'deepseek-v4-pro': (0.435, 0.87), 'deepseek-v4-flash': (0.14, 0.28)}

# answerer istenen saglayicida; judge SABIT deepseek-chat (kosular arasi
# kiyas ayni hakemle yapilmali)
client = OpenAI(api_key=os.environ[args.key_env], base_url=args.base_url)
judge_client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],
                      base_url='https://api.deepseek.com/v1')

def strip_think(txt):
    """MiniMax M3 tarzi inline <think>...</think> bloklarini ayikla."""
    out = re.sub(r'<think>.*?</think>', '', txt, flags=re.S).strip()
    if '<think>' in txt and '</think>' not in txt:
        return ''  # dusunce kesilmis, cevap yok -> bos say (retry tetikler)
    return out
contexts = json.load(open(f'{BASE}/locomo_contexts.json'))
verdicts = json.load(open(f'{BASE}/locomo_verdicts.json'))
speakers = {c['sample_id']: (c['conversation']['speaker_a'],
                             c['conversation']['speaker_b'])
            for c in json.load(open(DATA))}

idx = list(range(0, len(contexts), args.stride))
# Odenmis cevaplari yeniden satin alma: onceki kosunun cevap+verdict'i aynen
# tasinir, sadece eksik indeksler API'ye gider.
reused = {}
if args.reuse_from:
    prev = json.load(open(args.reuse_from))
    assert prev['model'] == args.model, 'reuse farkli model — kiyas bozulur'
    assert prev['run'] == args.run, 'reuse farkli run'
    reused = {a['i']: (a['response'], a['new_verdict']) for a in prev['answers']
              if not str(a['response']).startswith('ERROR')}
idx_todo = [i for i in idx if i not in reused]
print(f'model={args.model}  subset={len(idx)}/{len(contexts)} (stride={args.stride})'
      f'{f"  reuse={len(idx)-len(idx_todo)}  yeni={len(idx_todo)}" if reused else ""}')

# kernel'dekiyle birebir ayni template
TEMPLATE = (
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
    'Memories for user {spk_a}:\n\n{mem_a}\n\n'
    'Memories for user {spk_b}:\n\n{mem_b}\n\n'
    'Question: {q}\n\nAnswer:')

JUDGE_PROMPT = """You are grading a short answer against a gold answer.
Rules: semantic equivalence counts as CORRECT; the answer must contain the gold key fact;
dates/times must refer to the same date; extra words are fine; contradictions,
hedges without the fact, or missing key fact = WRONG.
Respond with exactly one word: CORRECT or WRONG.

Question: {q}
Gold answer: {gt}
Model answer: {resp}"""

usage = {'in': 0, 'out': 0}

def answer(i):
    c = contexts[i]
    spk_a, spk_b = speakers[c['sample_id']]
    prompt = TEMPLATE.format(
        spk_a=spk_a, spk_b=spk_b, q=c['question'],
        mem_a='\n'.join(f'- {t}' for t in c['ctx_a']),
        mem_b='\n'.join(f'- {t}' for t in c['ctx_b']))
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=args.model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=args.max_tokens, temperature=0)
            usage['in'] += r.usage.prompt_tokens
            usage['out'] += r.usage.completion_tokens
            txt = strip_think((r.choices[0].message.content or '').strip())
            if txt:
                return i, txt
            # reasoning butceyi yemis, cevap bos: bir kez genis butceyle dene
            if attempt == 0:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=args.max_tokens * 3, temperature=0)
                usage['in'] += r.usage.prompt_tokens
                usage['out'] += r.usage.completion_tokens
                txt = strip_think((r.choices[0].message.content or '').strip())
                return i, txt or 'ERROR: empty'
        except Exception as e:
            if attempt < 4:
                time.sleep(8 * (attempt + 1))
            else:
                return i, f'ERROR: {e}'
    return i, 'ERROR: empty'

def judge(i_resp):
    i, resp = i_resp
    v = verdicts[i]
    for attempt in range(4):
        try:
            r = judge_client.chat.completions.create(
                model='deepseek-chat',
                messages=[{'role': 'user', 'content': JUDGE_PROMPT.format(
                    q=v['question'], gt=v['answer'], resp=resp)}],
                max_tokens=4, temperature=0)
            j = r.choices[0].message.content.strip().upper()
            return i, 1 if j.startswith('CORRECT') else 0
        except Exception:
            time.sleep(5 * (attempt + 1))
    return i, 0

t0 = time.time()
answers = {i: r for i, (r, _) in reused.items()}
new_verdict = {i: v for i, (_, v) in reused.items()}
todo_answers = {}
with ThreadPoolExecutor(max_workers=8) as ex:
    for n, (i, resp) in enumerate(ex.map(answer, idx_todo), 1):
        todo_answers[i] = resp
        if n % 50 == 0:
            print(f'  cevap {n}/{len(idx_todo)}  {time.time()-t0:.0f}s', flush=True)
answers.update(todo_answers)

with ThreadPoolExecutor(max_workers=8) as ex:
    for n, (i, v) in enumerate(ex.map(judge, todo_answers.items()), 1):
        new_verdict[i] = v
        if n % 100 == 0:
            print(f'  judge {n}/{len(idx_todo)}  {time.time()-t0:.0f}s', flush=True)

# --- run6 ile kiyas (ayni subset) ---
by = defaultdict(lambda: {'n': 0, 'old': 0, 'new': 0, 'fixed': 0, 'broken': 0})
for i in idx:
    cat = verdicts[i]['category']
    old, new = verdicts[i]['verdict'], new_verdict[i]
    for k in (cat, 'TOPLAM'):
        b = by[k]
        b['n'] += 1; b['old'] += old; b['new'] += new
        b['fixed'] += (not old) and new
        b['broken'] += old and (not new)

errs = sum(1 for r in answers.values() if str(r).startswith('ERROR'))
names = {'1': 'multi', '2': 'temporal', '3': 'open', '4': 'single'}
print(f"\n{'kategori':>9} {'n':>4} | {'run6':>6} {args.model:>16} | {'duzelen':>7} {'bozulan':>7}")
for cat in sorted(by, key=lambda c: (c == 'TOPLAM', c)):
    b = by[cat]
    print(f"{names.get(cat, cat):>9} {b['n']:>4} | {b['old']/b['n']:>6.3f} {b['new']/b['n']:>16.3f} "
          f"| {b['fixed']:>7} {b['broken']:>7}")
pin, pout = PRICE.get(args.model, (0, 0))
cost = usage['in'] / 1e6 * pin + usage['out'] / 1e6 * pout
print(f"\nERROR={errs}  token: {usage['in']:,} in / {usage['out']:,} out  "
      f"~${cost:.2f} (answerer; judge haric)")

out = {'model': args.model, 'run': args.run, 'stride': args.stride,
       'n': len(idx), 'errors': errs, 'usage': usage, 'cost_usd': round(cost, 3),
       'by_category': {k: dict(v) for k, v in by.items()},
       'answers': [{'i': i, 'question': verdicts[i]['question'],
                    'gold': verdicts[i]['answer'], 'response': answers[i],
                    'old_verdict': verdicts[i]['verdict'],
                    'new_verdict': new_verdict[i]} for i in idx]}
slug = args.model.replace('/', '_')
json.dump(out, open(f'{BASE}/swap_{slug}_stride{args.stride}.json', 'w'), indent=1)
print(f"-> {BASE}/swap_{slug}_stride{args.stride}.json")
