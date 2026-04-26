# Changelog

All notable changes to adaptmem are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- 100/400 self-contained train+test (`make bench-longmemeval`) is
  wired but blocked on this Mac mini configuration — six attempts,
  silent exit after model load, both on MPS and CPU. Pipeline + harness
  work (the bench-ft300 / bench-ft200 targets reproduce the table); the
  blocker is local PyTorch / sentence-transformers env interaction
  (memory-pressure-related, swap saturation). v0.3 will pin a known-
  good dependency set or ship a Docker reproduce target.
- Mempal protocol-matched run (run our model through their
  `longmemeval_bench.py`) is v0.5.

## [0.1.0] — 2026-04-26 (initial)
- `AdaptMem` skeleton with hard-negative mining + contrastive fine-tune
  (MultipleNegativesRankingLoss) + persistence.
- 9 unit tests (`test_normalise`, `test_miner`).
- Public roadmap (v0.2 → v0.5) with exit criteria.
