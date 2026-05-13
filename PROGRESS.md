# adaptmem, progress & resume notes

**Last updated:** 2026-04-27 (post-public-push session).

This file is the resume contract: open the repo, read this, you know the
state of play. Updated at the end of each working session.

## Where we are

```
v0.1 skeleton          ████████████  done
v0.2 first bench       ████████████  done   (FT-100 R@5=0.978 self-contained shipped, Colab-trained Mac-tested)
v0.3 multi-bench       █████░░░░░░░  ~45%   (per-qtype done, encoder swap PoC done = honest null,
                                             raw MiniLM 400q baseline matches mempal published 0.965/0.966;
                                             ConvoMem + MemBench dataset-blocked)
v0.4 production-ready  ████████████  done   (CE rerank/streaming/evaluate/token-cost/py.typed/CI/release-workflow
                                             + mypy --strict; PyPI publish token Atakan-gated)
v0.5 mempal outreach   ░░░░░░░░░░░░  0%     (matched-protocol run via MemPalace/mempalace + GitHub Discussion;
                                             3-seed reproduce blocker is the same Mac/Py3.14 train deadlock)
v0.6 multi-bench/PyPI  ████████░░░░  ~65%  (PyPI release done, `pip install adaptmem` works;
                                             ConvoMem/MemBench/FActScore + Parquet on-disk pending)
v0.7 Metis integration ████████████  done  (ADR + adaptmem.server FastAPI + DaemonEncoder + Guard.from_daemon
                                             + Pipeline.from_daemon + metis semantic_memory_search tool all
                                             shipped; metis PR #6 merged to master 2026-04-27)
```

**Public:** https://github.com/nakata-app/adaptmem (master, CI green) · **PyPI:** https://pypi.org/project/adaptmem/ (v0.6.0, `pip install adaptmem` / `pip install "adaptmem[server]"`).

## Bench results, what's actually committed

| Model | Split | R@1 | R@5 | R@10 | JSON |
|---|---|---|---|---|---|
| FT-300 (metis-pair) | 300 train / 200 test | 0.915 | 0.995 | 0.995 | `benchmarks/results_ft300_direct.json` |
| FT-200 (metis-pair) | 200 train / 200 test | 0.900 | 0.990 | 0.995 | `benchmarks/results_ft200_direct.json` |
| FT-100 (self-contained) | 100 train / 400 test | 0.855 | 0.978 | 0.992 | `benchmarks/results_ft100_400.json` ✅ (Colab-trained, Mac-tested) |

Both numbers clear v0.2 sanity bar (R@5 ≥ 0.985); deltas move in the
expected direction (more train data → higher recall).

## MemPalace comparison, where we stand

| System | R@5 | Source | Verified |
|---|---|---|---|
| MemPalace raw | 0.966 | their published number | not independently re-run |
| MemPalace hybrid v4 | 0.984 | their published number | not independently re-run |
| MemPalace + Haiku rerank | 1.000 | their published number | LLM + 3 q spot-fix (out of category) |
| **adaptmem FT-300** | **0.995** | `results_ft300_direct.json` | committed |

**+2.9 pt over raw, +1.1 pt over hybrid v4.** LLM-free, no spot-fix.

**Caveat (v0.5 work):** sayılar same dataset / same protocol description
üzerinde, ama mempal'ın **kendi eval script'iyle** koşturulmadı. Apples-
to-apples kanıt v0.5'te: clone `MemPalace/mempalace`, plug
`AdaptMem.encode` into their `longmemeval_bench.py`, re-run, post the
matched table.

## Open items

### v0.2, self-contained reproduce, **CLOSED**
- ✅ FT-100 (100 train / 400 test) shipped, `results_ft100_400.json`,
  R@1=0.855 / R@5=0.978 / R@10=0.992. Colab-trained, Mac-tested.
  Train pipeline has a Mac-local PyTorch deadlock under memory pressure
  (workaround: train elsewhere, drop model dir into
  `benchmarks/bench-model-100/`, run
  `python benchmarks/longmemeval_eval.py --mode test --st-model benchmarks/bench-model-100/model`).
- v0.3 follow-up: pin a known-good dep set or ship a Docker reproduce
  target so a stranger doesn't need Colab.

### v0.3, multi-bench
- [ ] ConvoMem bench (Salesforce). Dataset arama: `Salesforce/convai_*`
  HF Hub'da var mı kontrol; varsa `benchmarks/convomem_eval.py`
  longmemeval_eval.py pattern'ında.
- [ ] MemBench (ACL 2025). Aynı şekilde.
- [ ] Encoder swap test: `AdaptMem(base_model="BAAI/bge-small-en-v1.5")`
  end-to-end koş, R@5 ölç. Param zaten var, sadece proof-of-concept run.
- [x] Per-question-type breakdown, done (`2ce0132`).

### v0.4, production-ready
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

### v0.5, mempal outreach
- [ ] Three-seed reproduce of FT-300 with mean ± stddev.
- [ ] Matched-protocol run via mempal's `longmemeval_bench.py`.
  - Wrapper hazır: `benchmarks/mempal_bench_with_ft.py` (monkey-patches
    `_bench_embed_fn` global, ZERO modification to mempal script).
  - Colab notebook: `benchmarks/colab_mempal_matched_protocol.ipynb`
    (3 ardışık run: raw default sanity → raw+FT-300 → hybrid_v4+FT-300).
  - Mac 8GB'da koşmaz (memory tight) → Colab'da koşulacak. Drive layout
    notebook header'ında. Bu Mac sırasında dataset/model upload + run.
- [ ] Open a GitHub Discussion on `MemPalace/mempalace` framed as
  "we extended your work", not "we beat your benchmark."
  - Draft hazır: `drafts/mempal_discussion.md`, matched-protocol
    sayıları gelince caveat satırı silinecek + tabloya ek sütun.

## How to resume (next session)

1. Read this file + `README.md` + `ROADMAP.md` + `docs/metis_integration.md`.
2. v0.7 (Metis integration) sign-off, Linux live-daemon smoke:
   - `pip install "adaptmem[server]"`
   - `adaptmem serve --port 7800 --base-model all-MiniLM-L6-v2`
   - `curl -X POST http://127.0.0.1:7800/reindex -d '{"corpus_id":"demo","documents":[{"id":"a","text":"..."}]}'`
   - `curl -X POST http://127.0.0.1:7800/search -d '{"query":"...","corpus_id":"demo","top_k":3}'`
   - Mac local smoke deadlocks inside `model.encode()` (Py3.14 + sentence-transformers + uvicorn cluster).
3. Merge metis PR `feat/semantic-memory-search-adaptmem` (Atakan-gated).
4. v0.5 mempal outreach, clone `MemPalace/mempalace` (default branch
   `develop`), plug `AdaptMem.encode` into their `longmemeval_bench.py`,
   run, commit a `results_mempal_protocol.json` row. **3-seed reproduce
   blocker:** Mac/Py3.14 train pipeline deadlock; need a Linux box.
5. PyPI release, token-gated; needs Atakan onayı.

## Toolchain

- Python 3.14 via `~/Projects/metis-pair/benchmarks/.venv` (has `adaptmem`
  installed editable, plus `pytest`, `numpy`, `sentence-transformers`,
  `torch`, `datasets`, `accelerate`, `ruff`).
- `make bench-longmemeval`, self-contained reproduction (see Makefile).
- Tests: `cd ~/Projects/adaptmem && ../metis-pair/benchmarks/.venv/bin/pytest -q`
- Current suite: **26/26 pass**, lint clean.

## Public

- Repo: https://github.com/nakata-app/adaptmem (master, MIT, CI green).
- Sibling repos: `nakata-app/claimcheck`.
- PyPI: `pip install adaptmem` (core) or `pip install "adaptmem[server]"` (HTTP daemon).
