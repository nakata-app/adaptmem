# adaptmem — progress & resume notes

**Last updated:** 2026-04-26 (post-Claude-session, before reboot).

This file is the resume contract: open the repo, read this, you know the
state of play. Updated at the end of each working session.

## Where we are

```
v0.1 skeleton          ████████████  done
v0.2 first bench       ██████████░░  ~85%   (100/400 self-contained train BLOCKED, see below)
v0.3 multi-bench       ████░░░░░░░░  ~30%   (per-qtype done, ConvoMem/MemBench/encoder swap kalan)
v0.4 production-ready  ███████████░  ~90%   (CE rerank/streaming/evaluate/token-cost/py.typed/CI done; PyPI release credential bekliyor)
v0.5 mempal outreach   ░░░░░░░░░░░░  0%     (matched-protocol run + GitHub Discussion)
```

## Bench results — what's actually committed

| Model | Split | R@1 | R@5 | R@10 | JSON |
|---|---|---|---|---|---|
| FT-300 (metis-pair) | 300 train / 200 test | 0.915 | 0.995 | 0.995 | `benchmarks/results_ft300_direct.json` |
| FT-200 (metis-pair) | 200 train / 200 test | 0.900 | 0.990 | 0.995 | `benchmarks/results_ft200_direct.json` |
| FT-100 (self-contained) | 100 train / 400 test | 0.855 | 0.978 | 0.992 | `benchmarks/results_ft100_400.json` ✅ (Colab-trained, Mac-tested) |

Both numbers clear v0.2 sanity bar (R@5 ≥ 0.985); deltas move in the
expected direction (more train data → higher recall).

## MemPalace comparison — where we stand

| System | R@5 | Source | Verified |
|---|---|---|---|
| MemPalace raw | 0.966 | their published number | not independently re-run |
| MemPalace hybrid v4 | 0.984 | their published number | not independently re-run |
| MemPalace + Haiku rerank | 1.000 | their published number | LLM + 3 q spot-fix (out of category) |
| **adaptmem FT-300** | **0.995** | `results_ft300_direct.json` | committed |

**+2.9 pt over raw, +1.1 pt over hybrid v4.** LLM-free, no spot-fix.

**Caveat (v0.5 work):** sayılar same dataset / same protocol description
üzerinde, ama mempal'ın **kendi eval script'iyle** koşturulmadı. Apples-
to-apples kanıt v0.5'te: clone `milla-jovovich/mempalace`, plug
`AdaptMem.encode` into their `longmemeval_bench.py`, re-run, post the
matched table.

## Open items

### v0.2 — self-contained reproduce
- ✅ FT-100 (100 train / 400 test) JSON committed — Colab-trained,
  Mac-tested. Train pipeline still has a Mac-local PyTorch deadlock
  (workaround: train elsewhere, drop the model dir into
  `benchmarks/bench-model-100/`, run `python benchmarks/longmemeval_eval.py
  --mode test --st-model benchmarks/bench-model-100/model` locally).
- v0.3 follow-up: pin a known-good dep set or ship a Docker reproduce
  target so a stranger doesn't need Colab.

### v0.3 — multi-bench
- [ ] ConvoMem bench (Salesforce). Dataset arama: `Salesforce/convai_*`
  HF Hub'da var mı kontrol; varsa `benchmarks/convomem_eval.py`
  longmemeval_eval.py pattern'ında.
- [ ] MemBench (ACL 2025). Aynı şekilde.
- [ ] Encoder swap test: `AdaptMem(base_model="BAAI/bge-small-en-v1.5")`
  end-to-end koş, R@5 ölç. Param zaten var, sadece proof-of-concept run.
- [x] Per-question-type breakdown — done (`2ce0132`).

### v0.4 — production-ready
- [x] Cross-encoder rerank (`5beec31`)
- [x] Streaming `add_corpus` (`409eb8f`)
- [x] `evaluate` CLI subcommand (`12057d5`)
- [x] Token cost report (`2d6ef87`)
- [x] `device` parameter (`04aa59a`)
- [x] py.typed (`c4db02c`)
- [x] GitHub Actions CI (`4386d8b`)
- [x] CLI subprocess smoke tests (`5ba9f34`)
- [ ] On-disk Parquet index (corpus > 50k chunks). Skeleton not yet.
- [ ] PyPI release. Needs maintainer-controlled API token (out of repo).

### v0.5 — mempal outreach
- [ ] Three-seed reproduce of FT-300 with mean ± stddev.
- [ ] Matched-protocol run via mempal's `longmemeval_bench.py`.
- [ ] Open a GitHub Discussion on `milla-jovovich/mempalace` framed as
  "we extended your work", not "we beat your benchmark."

## How to resume (next session)

1. Reboot done? `vm.swapusage` should show low utilisation.
2. Read this file + `README.md` + `ROADMAP.md`.
3. Run `make bench-longmemeval` — if it completes:
   - Commit `benchmarks/results_ft100_400.json`
   - Add the FT-100 row to README table
   - Mark v0.2 closed.
4. If it still fails the same way, the blocker isn't memory — drop into
   `python -c "from adaptmem import AdaptMem; AdaptMem().train(...)"`
   with tiny inputs (corpus=2 strings, 1 query) to isolate the layer
   that crashes. Then file findings here.
5. Pick the next v0.3 / v0.5 item from the lists above.

## Toolchain

- Python 3.14 via `~/Projects/metis-pair/benchmarks/.venv` (has `adaptmem`
  installed editable, plus `pytest`, `numpy`, `sentence-transformers`,
  `torch`, `datasets`, `accelerate`).
- `make bench-longmemeval` — self-contained reproduction (see Makefile).
- Tests: `cd ~/Projects/adaptmem && ../metis-pair/benchmarks/.venv/bin/pytest -q`
- Current suite: **23/23 pass**.

## Commit log highlights (this session)

```
ba50505 README: v0.2 status + honest note on train pipeline silently exiting
c0eec2f make: DEVICE=cpu default + MODEL_100/model path fix
04aa59a core: device override (CPU forcing for MPS-deadlock workaround)
5ba9f34 test: subprocess CLI smoke tests
2ce0132 bench: per-question-type recall breakdown in eval output
2d6ef87 core: report n_tokens_approx + tokens_per_s in train() stats
12057d5 cli: add `adaptmem evaluate` — recall@k against labelled queries
409eb8f core: streaming add_corpus — append entries without re-encoding
4386d8b ci: GitHub Actions matrix (py 3.10/3.11/3.12)
c4db02c package: ship py.typed marker (PEP 561)
5a26e96 cli: surface --rerank, --rerank-model, --rerank-top-k
5beec31 core: optional cross-encoder rerank in AdaptMem.search
1d3149f make: bench-longmemeval self-contained reproduction target
90d52af bench: add FT-200 sanity row + second results JSON
eb2daf2 README: replace placeholder R@5 row with reproduced numbers + audit ref
06e593e bench: longmemeval reproduction harness + first results JSON
```

15 commits on top of `52dc52c` (v0.1 ROADMAP commit). All shipped to
`master` locally. Push to remote when ready (no remote currently
configured).
