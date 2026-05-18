"""
CodeCrossEnc-v1: CPU-only local training (Mac mini safe mode)
- MPS disabled, force CPU
- 30K pairs (small enough for 8GB RAM)
- batch_size=8
- Checkpoint save with sync
"""
import os, sys, time, json
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import torch
torch.backends.mps.is_available = lambda: False  # force CPU

DEVICE = 'cpu'
N_POS   = 30_000   # pairs
N_NEG   = 2        # neg per pos
BATCH   = 8
LR      = 2e-5
WARMUP  = 300
OUT_DIR = os.path.expanduser('~/Projects/adaptmem/checkpoints/codecrossenc-v1-local')
LOG_FILE= os.path.expanduser('~/Projects/adaptmem/benchmarks/codecrossenc_train.log')

os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

log('=== CodeCrossEnc-v1 local CPU training ===')
log(f'device: {DEVICE}, pairs: {N_POS} pos + {N_POS*N_NEG} neg = {N_POS*(1+N_NEG)} total')
log(f'batch: {BATCH}, lr: {LR}, warmup: {WARMUP}')

log('loading CodeSearchNet python train...')
import random
random.seed(42)
from datasets import load_dataset

ds = load_dataset('code_search_net', 'python', split='train')
log(f'dataset loaded: {len(ds)} samples')

# build corpus list for negatives
all_codes = [r['func_code_string'] for r in ds
             if r.get('func_code_string') and len(r['func_code_string']) >= 40]
log(f'corpus for negatives: {len(all_codes)} codes')

# build pairs
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

samples = []
for r in ds:
    doc  = (r.get('func_documentation_string') or '').strip()
    code = (r.get('func_code_string') or '').strip()
    if len(doc) < 10 or len(code) < 40:
        continue
    samples.append((doc.splitlines()[0][:200], code))
    if len(samples) >= N_POS:
        break

log(f'positive pairs: {len(samples)}')

train_examples = []
for q, pos in samples:
    train_examples.append(InputExample(texts=[q, pos], label=1.0))
    for _ in range(N_NEG):
        neg = random.choice(all_codes)
        train_examples.append(InputExample(texts=[q, neg], label=0.0))

random.shuffle(train_examples)
log(f'total examples: {len(train_examples)}')

# free dataset from memory before training
del ds, all_codes, samples
import gc; gc.collect()

log('loading cross-encoder model...')
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels=1, max_length=384, device=DEVICE)
log('model loaded on CPU')

loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH)
steps  = len(loader)
log(f'steps per epoch: {steps} (~{steps*0.4/60:.0f}-{steps*1.5/60:.0f} min on CPU)')

t0 = time.time()
model.fit(
    train_dataloader=loader,
    epochs=1,
    warmup_steps=WARMUP,
    optimizer_params={'lr': LR},
    show_progress_bar=True,
)
elapsed = time.time() - t0
log(f'training done in {elapsed/60:.1f} min')

log(f'saving to {OUT_DIR} ...')
model.save(OUT_DIR)
log('save done')

# verify files saved
files = os.listdir(OUT_DIR)
log(f'saved files: {files}')
assert len(files) >= 2, f'save may have failed: {files}'

log('=== TRAINING COMPLETE ===')
log(f'checkpoint: {OUT_DIR}')
summary = {
    'n_pos': N_POS, 'n_neg': N_POS * N_NEG,
    'batch': BATCH, 'lr': LR,
    'runtime_min': round(elapsed / 60, 1),
    'out_dir': OUT_DIR,
    'files': files,
}
log(json.dumps(summary))
print('\n\nDONE. Checkpoint:', OUT_DIR)
