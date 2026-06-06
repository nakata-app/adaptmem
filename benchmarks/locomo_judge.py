"""LoCoMo cevap puanlama — DeepSeek judge (CORRECT/WRONG), 8 paralel.

Kullanim: python locomo_judge.py [run_adi]   (varsayilan: locomo_e2e_run6)
Cikti: locomo_judged.json (kategori toplami) + locomo_verdicts.json
(soru-bazli karar dokumu — run'lar arasi flip analizi icin, run5 kor-noktasi).
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

RUN = sys.argv[1] if len(sys.argv) > 1 else 'locomo_e2e_run6'
BASE = f'/Users/macmini/Projects/adaptmem/results/{RUN}/results'
client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],
                base_url='https://api.deepseek.com/v1')
answers = json.load(open(f'{BASE}/locomo_answers.json'))['mnemonics']
print(f'puanlanacak: {len(answers)}  (run={RUN})')

PROMPT = """You are grading a short answer against a gold answer.
Rules: semantic equivalence counts as CORRECT; the answer must contain the gold key fact;
dates/times must refer to the same date; extra words are fine; contradictions,
hedges without the fact, or missing key fact = WRONG.
Respond with exactly one word: CORRECT or WRONG.

Question: {q}
Gold answer: {gt}
Model answer: {resp}"""

def judge(i_a):
    i, a = i_a
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model='deepseek-chat',
                messages=[{'role':'user','content': PROMPT.format(
                    q=a['question'], gt=a['answer'], resp=a['response'])}],
                max_tokens=4, temperature=0)
            v = r.choices[0].message.content.strip().upper()
            return i, 1 if v.startswith('CORRECT') else 0
        except Exception:
            time.sleep(5*(attempt+1))
    return i, 0

t0 = time.time()
results = [None]*len(answers)
with ThreadPoolExecutor(max_workers=8) as ex:
    for n, (i, v) in enumerate(ex.map(judge, enumerate(answers)), 1):
        results[i] = v
        if n % 200 == 0:
            print(f'  {n}/{len(answers)}  acc-so-far={sum(r for r in results if r is not None)/n:.3f}  {time.time()-t0:.0f}s', flush=True)

from collections import defaultdict
by = defaultdict(lambda: [0,0])
for a, v in zip(answers, results):
    by[a['category']][0] += v; by[a['category']][1] += 1
out = {
    'n': len(answers),
    'accuracy': round(sum(results)/len(answers), 4),
    'by_category': {c: {'n': t, 'acc': round(s/t, 4)} for c, (s, t) in sorted(by.items())},
    'judge': 'deepseek-chat (CORRECT/WRONG, temp0)',
}
json.dump(out, open(f'{BASE}/locomo_judged.json','w'), indent=1)
# Soru-bazli karar dokumu: flip analizi (hangi soru run'lar arasi dondu)
json.dump([{'sample_id': a.get('sample_id'), 'question': a['question'],
            'answer': a['answer'], 'response': a['response'],
            'category': a['category'], 'verdict': v}
           for a, v in zip(answers, results)],
          open(f'{BASE}/locomo_verdicts.json','w'), indent=1)
print(json.dumps(out, indent=1))
print(f'verdicts -> {BASE}/locomo_verdicts.json')
