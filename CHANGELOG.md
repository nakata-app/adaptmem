# Changelog

All notable changes to adaptmem are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — v0.7 Metis integration (PoC shipped)
- **`adaptmem.server`** — FastAPI app exposing `/healthz`, `/version`,
  `/embed`, `/reindex`, `/search`. Lazy encoder load (model only built
  on first embed/reindex call). Pydantic schemas at module level so
  FastAPI introspection works.
- **`adaptmem serve`** CLI subcommand — `--port`, `--host`, `--device`,
  `--uds` (Unix-domain-socket alternative for zero-TCP-overhead
  callers).
- **`[server]` optional dep** — `pip install "adaptmem[server]"` pulls
  fastapi + uvicorn[standard] + pydantic. Core install stays minimal.
- **`tests/test_server.py`** — 5 endpoint tests via FastAPI TestClient.
  Real encoder, real ranking, no mocks for the retrieval pass.
- **`docs/metis_integration.md`** — full ADR. Three bridge options
  considered (subprocess shell-out, PyO3, HTTP daemon); HTTP daemon
  chosen. v1 endpoint contract, compatibility matrix, failure modes,
  rollout plan.

### Added — v0.4 production-ready surface (closed)
- **mypy --strict pass.** `dict[str, Any]` annotations, `DataLoader[Any]`
  local annotation, `from typing import Any`, type-ignore on
  sentence_transformers.losses (3rd-party stub gap).
- **`release.yml`** — wheel build + sdist + tag-gated PyPI publish
  (skipped via shell guard when `PYPI_API_TOKEN` is absent).

### Added — bench results
- `benchmarks/results_minilm_baseline_400.json` — raw MiniLM 400q
  R@5=0.965, matches MemPalace published 0.966 within 0.1pt.
  Independent confirmation of the protocol.
- `benchmarks/results_minilm_baseline_50.json` — raw MiniLM 50q
  baseline alongside the BGE comparison.
- `benchmarks/results_bge_small_50.json` — BAAI/bge-small-en-v1.5 raw
  50q. Encoder swap = honest null result (R@5=0.98 vs MiniLM 0.98).
  The lift comes from fine-tuning, not the base model.
- `benchmarks/bench_st_inline.py` — minimal harness that imports
  helpers from `longmemeval_eval.py` but bypasses `cmd_test` (which
  deadlocks on Mac/Py3.14 + sentence-transformers + uvicorn-thread-
  pool cluster).

### Added — v0.4 production-ready surface
- Optional cross-encoder rerank stage on `AdaptMem.search()` (default
  off). Enable via `AdaptMem(rerank=True)`; CE candidates pulled from a
  widened bi-encoder set (`rerank_top_k or top_k * 3`). Lazy load.
  `RetrievalHit.score` is the CE score when reranking is on.
- `AdaptMem.add_corpus()` for streaming index updates — encode new
  entries and append to the in-memory embedding matrix without
  re-encoding the existing corpus. De-duplicates by id.
- `adaptmem evaluate --model PATH --queries queries.json --top-k N`
  CLI subcommand. Writes a JSON dict with R@1, R@5, R@top_k.
- `device` parameter on `AdaptMem(...)` and `--device` flag on the
  bench harness. Forces PyTorch device (`"cpu"`, `"cuda"`, `"mps"`,
  or autodetect). Persisted in `config.json` so `.load()` honours the
  same choice.
- Token cost report in `train()` stats output: `n_tokens_approx`
  (whitespace word count over corpus + every TrainingPair) and
  `tokens_per_s`.
- Per-question-type recall breakdown in `benchmarks/longmemeval_eval.py`
  output (`per_question_type` map: multi-session, temporal-reasoning,
  knowledge-update, single-session-{user,assistant,preference}).
- `py.typed` (PEP 561) marker so consumers can pick up inline type
  hints with mypy / pyright.
- GitHub Actions CI matrix on Python 3.10 / 3.11 / 3.12 (lint + tests).
- CLI flags `--rerank`, `--rerank-model`, `--rerank-top-k` on `train`,
  `search`, and `evaluate` subcommands.
- 6 subprocess smoke tests for the CLI (argparse plumbing, missing
  args, no-subcommand error).

### Added — v0.2 first benchmark
- `benchmarks/longmemeval_eval.py` — train + test CLI matching the
  MemPalace eval protocol (per-question fresh corpus, user-only
  encoding, drop empty sessions, R@k macro mean).
- `benchmarks/results_ft300_direct.json` — first reproducible run
  (FT-300, R@1=0.915, R@5=0.995, 200 held-out questions, CPU).
- `benchmarks/results_ft200_direct.json` — sanity second run
  (FT-200, R@1=0.900, R@5=0.990).
- `benchmarks/results_ft100_400.json` — self-contained reproduce
  (FT-100 trained on the shipped 100/400 split, evaluated on the
  400 held-out questions): R@1=0.855, R@5=0.978, R@10=0.992. Includes
  per-question-type breakdown.
- `benchmarks/data/split_ids_100_400.json` — deterministic
  shuffle(seed=42) of 500 LongMemEval question_ids → first 100 train,
  remaining 400 test. Committed for bit-reproducible runs.
- `Makefile` with self-contained `bench-longmemeval` target. Default
  `DEVICE=cpu` (Apple-silicon MPS deadlock workaround).
- README table with reproduced FT-300 / FT-200 rows + audit links to
  the JSONs + reproduce CLI snippet.

### Changed
- README placeholder R@5 row replaced with the reproduced numbers (R@1
  added as a column). Previous "FT-300 + CE top-5" row removed
  (claim was unbacked by a committed JSON; CE rerank capture now lands
  with v0.4 shipped above, JSON measurement still pending).
- MemPalace numbers in the README explicitly tagged as "from their
  published results, not independently re-run here." Honest about the
  measurement boundary.

### Open (carried into v0.3 / v0.5)
- Mempal protocol-matched run (run our model through their
  `longmemeval_bench.py`) is v0.5.

### Notes
- The FT-100 self-contained training path runs end-to-end on Colab
  CPU; on this Mac mini the contrastive `fit()` step deadlocks
  (PyTorch + sentence-transformers + memory-pressure interaction,
  six local attempts). Test-mode (`make bench-ft100` against a model
  trained elsewhere) works on Mac. v0.3 will land a pinned-deps /
  Docker target so a stranger has a Mac-local fallback.

## [0.1.0] — 2026-04-26 (initial)
- `AdaptMem` skeleton with hard-negative mining + contrastive fine-tune
  (MultipleNegativesRankingLoss) + persistence.
- 9 unit tests (`test_normalise`, `test_miner`).
- Public roadmap (v0.2 → v0.5) with exit criteria.
