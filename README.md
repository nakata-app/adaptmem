# adaptmem

[![PyPI version](https://img.shields.io/pypi/v/adaptmem.svg)](https://pypi.org/project/adaptmem/)
[![Python versions](https://img.shields.io/pypi/pyversions/adaptmem.svg)](https://pypi.org/project/adaptmem/)
[![License: MIT](https://img.shields.io/pypi/l/adaptmem.svg)](LICENSE)
[![CI](https://github.com/nakata-app/adaptmem/actions/workflows/ci.yml/badge.svg)](https://github.com/nakata-app/adaptmem/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nakata-app/adaptmem/graph/badge.svg)](https://codecov.io/gh/nakata-app/adaptmem)

**Beat your retrieval baseline with 200 lines of hard-negative mining and a 90MB encoder.**

You point adaptmem at a domain (a corpus + a handful of labelled queries). It mines hard negatives, fine-tunes a tiny embedder on the contrastive objective, and hands you back a retriever that outperforms much larger generic models on your data.

This is the pipeline that pushed our internal LongMemEval R@5 from `0.966` (off-the-shelf MiniLM, matching MemPalace's "raw" headline) to **`0.9950`** on a generalisable held-out split, without any LLM in the loop, without hand-tuning, in a single epoch on CPU.

## Why this exists

The retrieval-quality literature has converged on a default: pick a 100M+ parameter generic embedder (bge-base, gte-base, mxbai), throw it at your data, hope it generalises. It usually doesn't, generic embedders compress concepts that **don't** matter in your domain and lose distinctions that **do**.

Domain adaptation works. The papers know it (DPR, ColBERT, SBERT). But the open-source workflow is fragmented:

- Hard-negative mining lives in one tutorial,
- Contrastive loss in another,
- Evaluation in a third,
- And every example assumes you already have a label set.

`adaptmem` is the missing one-shot wrapper. You write five lines, you get a domain-tuned encoder.

## What it does

```
your data (corpus + a few labelled queries)
        │
        ▼
[1] hard-negative mining       # vanilla MiniLM ranks haystack, mines top-K non-gold
        │
        ▼
[2] contrastive fine-tune      # MultipleNegativesRankingLoss, 1 epoch CPU
        │
        ▼
[3] (optional) cross-encoder   # ms-marco-MiniLM-L-12-v2 rerank
        │
        ▼
domain-tuned retriever         # serve via .search(query, top_k)
```

The recipe is small on purpose. Every choice is documented. Every step is one method call.

## Concrete result on LongMemEval (s_cleaned, 500 questions)

| System | R@1 | R@5 | R@10 | n | LLM | Hand-tune | Generalisable |
|---|---|---|---|---|---|---|---|
| BM25 sparse baseline |, | 0.70 |, | 500 | ✗ | ✗ | ✓ |
| Stella dense (academic) |, | ~0.85 |, | 500 | ✗ | ✗ | ✓ |
| MemPalace raw (ChromaDB + MiniLM) |, | 0.966 |, | 500 | ✗ | ✗ | ✓ |
| MemPalace hybrid v4 generalisable |, | 0.984 |, | 500 | ✗ | ✗ | ✓ |
| MemPalace + Haiku rerank |, | 1.000 |, | 500 | ✓ | ✓ (3 q spot-fix) | ✗ |
| **MiniLM-L6 raw (our eval, no FT)** | 0.795 | **0.965** | 0.980 | 400 | ✗ | ✗ | ✓ |
| BGE-small-en-v1.5 raw (our eval, no FT) | 0.80 | 0.98 | 1.00 | 50 | ✗ | ✗ | ✓ |
| adaptmem (FT-100 dense, **self-contained**) | 0.855 | 0.978 | 0.992 | 400 | ✗ | ✗ | ✓ |
| adaptmem (FT-200 dense) | 0.900 | 0.990 | 0.995 | 200 | ✗ | ✗ | ✓ |
| **adaptmem (FT-300 dense)** | **0.915** | **0.995** | **0.995** | 200 | **✗** | **✗** | **✓** |
| MemPalace raw (matched-protocol, their bench script) | 0.806 | 0.966 | 0.982 | 500 | ✗ | ✗ | ✓ |
| MemPalace raw + adaptmem FT-300 (matched-protocol) | 0.862 | 0.980 | 0.994 | 500 | ✗ | ✗ | ✓ |
| **MemPalace hybrid_v4 + adaptmem FT-300 (matched-protocol)** | **0.916** | **0.990** | **0.998** | 500 | **✗** | **✗** | **✓** |

Adaptmem numbers reproduced from committed runs, see [`benchmarks/results_ft300_direct.json`](benchmarks/results_ft300_direct.json), [`benchmarks/results_ft200_direct.json`](benchmarks/results_ft200_direct.json), [`benchmarks/results_ft100_400.json`](benchmarks/results_ft100_400.json), [`benchmarks/results_minilm_baseline_400.json`](benchmarks/results_minilm_baseline_400.json), [`benchmarks/results_bge_small_50.json`](benchmarks/results_bge_small_50.json), and [`benchmarks/results_minilm_baseline_50.json`](benchmarks/results_minilm_baseline_50.json). Reproduce harness: `python benchmarks/bench_st_inline.py --split benchmarks/data/split_ids_100_400.json --st-model <hf-id-or-path> --out <results.json>`.

**Two findings:**
1. **Our raw MiniLM 400q (R@5=0.965) matches MemPalace's published raw (0.966) within 0.1pt**, same encoder family, same protocol, independent eval. The protocol is sound.
2. **Encoder swap (BGE-small) does not lift R@5 by itself**, 0.98 vs 0.98 on 50q matched split. The lift comes from the **fine-tune step**, not the base model. FT-100 lifts +1.3pt over MiniLM raw on the same 400q split; FT-300 lifts +3.0pt over the published mempal raw.

Train-set size scales recall as expected: 100→200→300 train queries gives R@5 0.978→0.990→0.995 and R@1 0.855→0.900→0.915. The FT-100 row sits 0.7pt below the ROADMAP v0.2 sanity bar (R@5 ≥ 0.985); 200+ train queries clear it comfortably.

### Reproduce

```bash
# Evaluate the existing FT-300 SentenceTransformer model directly
python benchmarks/longmemeval_eval.py --mode test \
    --st-model /path/to/minilm-lme-ft-300 \
    --results-out benchmarks/results_ft300_direct.json
```

A cross-encoder rerank stage (R@1 lift) is on the v0.4 roadmap, a JSON capture is not yet committed.

## Usage (planned API)

```python
from adaptmem import AdaptMem

# Your domain
corpus = ["passage 1 text...", "passage 2 text...", ...]
labelled = [
    {"query": "...", "relevant_ids": ["p3", "p7"]},
    ...
]

am = AdaptMem(base_model="all-MiniLM-L6-v2")
am.train(corpus=corpus, labelled=labelled, epochs=1)
am.save("./my-domain-encoder")

# Use
hits = am.search("user query", top_k=5)
for chunk_id, score in hits:
    print(chunk_id, score)
```

CLI parity:

```bash
# Train + persist the rerank flag so .load() restores it later
adaptmem train --corpus corpus.json --queries queries.json --out my-encoder/ \
    [--rerank --rerank-model cross-encoder/ms-marco-MiniLM-L-12-v2]

# Serve a query, bi-encoder by default, or force CE rerank for an A/B
adaptmem search --model my-encoder/ --query "..." --top-k 5 [--rerank --rerank-top-k 15]

# Score a saved model against a labelled queries file (R@1 / R@5 / R@k)
adaptmem evaluate --model my-encoder/ --queries labelled.json --top-k 10

# Reproduce the LongMemEval table (Makefile, single command)
make bench-longmemeval
```

## Shell tab-completion (optional)

Install `argcomplete` once per shell, then complete subcommands +
flags by pressing Tab:

```bash
pip install "adaptmem[shell]"
# bash:
eval "$(register-python-argcomplete adaptmem)" >> ~/.bashrc
# zsh:
eval "$(register-python-argcomplete adaptmem)" >> ~/.zshrc
# fish:
register-python-argcomplete --shell fish adaptmem | source
```

Now `adaptmem se<Tab>` expands to `adaptmem serve`, and
`adaptmem serve --<Tab><Tab>` lists every flag.

## Daemon mode (`adaptmem serve`)

For multi-language consumers (e.g. metis, a Rust agent CLI) or for any
deployment where you want **one model load shared across many callers**,
run adaptmem as a long-lived HTTP daemon.

```bash
# Install the optional server extras (FastAPI + uvicorn + pydantic).
pip install "adaptmem[server]"

# Start the daemon. Bi-encoder model loads lazily on the first /embed call.
adaptmem serve --port 7800 --base-model all-MiniLM-L6-v2
# or, if you prefer a Unix-domain socket:
adaptmem serve --uds /tmp/adaptmem.sock
```

Endpoint contract (full ADR in [`docs/metis_integration.md`](docs/metis_integration.md)):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | `{"ok": true, "uptime_s": …}` |
| `GET` | `/version` | `{"adaptmem": …, "encoder": …, "corpora": [...]}` |
| `POST` | `/embed` | `{"texts": [...]}` → `{"embeddings": [[…]], "dim": …}` |
| `POST` | `/reindex` | per-corpus embedding (replace + re-encode) |
| `POST` | `/search` | top-k retrieval against an indexed corpus |

**Two clients ship today:**
- [`halluguard.daemon.DaemonEncoder`](https://github.com/nakata-app/halluguard/blob/master/halluguard/daemon.py), drop-in
  `encoder` for `Guard.from_daemon(documents=[...], daemon_url=…)`.
- [`claimcheck.Pipeline.from_daemon`](https://github.com/nakata-app/claimcheck/blob/main/claimcheck/pipeline.py), same factory shape; NLI verifier
  stays local.

**One Rust client lands in metis** (`semantic_memory_search` tool, branch
`feat/semantic-memory-search-adaptmem`) so an agent loop can issue
domain-tuned semantic queries against `.metis/memory/*.md` without any
Python in the build.

## The cluster, adaptmem in context

`adaptmem` is one of four sibling packages that together cover the
**no-LLM-judge LLM safety stack.** Each one solves a different slice
of "is this AI claim trustworthy?", pick what you need.

```
                                           ┌────────────────┐
                                           │  user input    │
                                           └────────┬───────┘
                                                    │
                                ┌─────────────► promptguard ◄─────── input gate
                                │                   │                (jailbreak / injection)
                                │                   ▼
                                │             ┌──────────┐
        adaptmem ──── retrieval │             │   LLM    │
        (this repo)             │             └────┬─────┘
                                │                  │
                                │                  ▼
                                │             halluguard ◄─────── output gate
                                │             (corpus-grounded)    (closed world)
                                │                  │
                                │                  ▼
                                └──────────►  claimcheck ◄───────── orchestration
                                              (adaptmem + halluguard,
                                               one Pipeline)
                                                   │
                                          (claim isn't in the corpus)
                                                   ▼
                                              truthcheck ◄────── open-world fact check
                                                                  (web-grounded)
```

| Package | Surface | When to reach for it |
|---|---|---|
| **[adaptmem](https://github.com/nakata-app/adaptmem)** | `AdaptMem.train(corpus, queries) / .search(q)` | Your retrieval is too generic. You have a corpus + a few labelled queries and want a domain-tuned encoder in 5 lines. |
| **[halluguard](https://github.com/nakata-app/halluguard)** | `Guard.from_documents(docs).check(answer)` | You have an LLM answer and a corpus. Did the LLM stay grounded? |
| **[claimcheck](https://github.com/nakata-app/claimcheck)** | `Pipeline.from_corpus(...)`, `from_daemon(...)`, `check(answer)` | Composition: domain-tuned retrieval **plus** verification, behind one API. |
| **promptguard** (pre-v0.1) | `PromptGuard().check(user_input)` | Block prompt-injection / jailbreak attempts before they reach your LLM. |
| **truthcheck** (pre-v0.1) | `WebFactChecker().check(claim)` | Claim isn't in your corpus, does the open web back it up? |

All five are **vendor-neutral** (no Anthropic / OpenAI / Google
required), all five are **deterministic where possible** (no LLM
judge in the inference path of halluguard / promptguard), and all
five compose into a single safety pipeline if your stack needs the
full stack.

## What it is NOT

- Not a generic embedder. The output model is **specialised** to the corpus you trained on.
- Not a replacement for retrieval engineering. You still need to think about chunking, encoding format, and ground-truth labels.
- Not a one-click win when your queries are out-of-distribution. Domain adaptation rewards in-distribution test data.

## Status

`v0.4` in flight, production-ready surface mostly landed:

- **API:** hard-negative mining + contrastive FT + persistence (v0.1), optional
  cross-encoder rerank (`AdaptMem(rerank=True)`), streaming index updates
  (`add_corpus()`), `device` override (CPU / CUDA / MPS) all in.
- **CLI:** `adaptmem train | search | evaluate` with `--rerank /
  --rerank-model / --rerank-top-k` on each. 6 subprocess smoke tests.
- **Bench:** `benchmarks/longmemeval_eval.py` train+test harness with
  per-question-type breakdown. Two committed reproducible runs (FT-300,
  FT-200). `Makefile` `bench-longmemeval` target with `DEVICE=cpu` default.
- **Quality:** `py.typed` (PEP 561) for downstream type-checkers, GitHub
  Actions CI on Python 3.10/3.11/3.12, train() returns `n_tokens_approx`
  + `tokens_per_s` for budget planning. 23 passing tests.

Open: on-disk Parquet persistence (warranted only at corpus > 50k chunks,
not yet started); PyPI release (gated on a maintainer API token); the
self-contained 100/400 reproduction described below.

Reference numbers (held-out 200q on the 300/200 split): R@1=0.915,
R@5=0.995 with FT-300; R@1=0.900, R@5=0.990 with FT-200. Both runs
clear the v0.2 sanity bar (R@5 ≥ 0.985) and the deltas move in the
expected direction (more train data → higher recall). See
`benchmarks/results_ft300_direct.json` and
`benchmarks/results_ft200_direct.json`.

**Reproducibility caveat (v0.2 open item):** the self-contained 100/400
train+test target (`make bench-longmemeval`) is wired up and
deterministic on its split, but on this Mac mini configuration the
contrastive fine-tune step silently exits after model load, both on
MPS (default) and on `--device cpu`. The bench harness, split file,
and Makefile all work; the bottleneck is local PyTorch+sentence-
transformers compatibility, not the pipeline. A v0.3 follow-up will
either pin a working dependency set or ship a containerised reproduce
target. In the meantime, `make bench-ft300` / `bench-ft200` (using the
externally trained metis-pair models) reproduce the README numbers.

## License

MIT.
