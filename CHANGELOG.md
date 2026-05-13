# Changelog

All notable changes to adaptmem are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0], 2026-05-14

The "server is ready for someone else's machine" release. Production
hardening on the HTTP daemon (auth, rate limit, OTel, durable store,
multi-tenant, federated search, graceful shutdown) + a second-domain
bench track (CodeSearchNet python) that validates the FT recipe outside
LongMemEval.

### Added, v0.6 production hardening (this release)
- `--api-keys-file` RBAC with `viewer` / `admin` roles. Reads embed /
  search are viewer-callable; mutations require admin.
- Rate limiting via `slowapi`, middleware mode so `Retry-After`
  headers actually land on `429` responses.
- OpenTelemetry tracing, opt-in via `pip install adaptmem[telemetry]`,
  OTLP exporter + auto-instrumentation for FastAPI and httpx.
- Graceful shutdown via FastAPI lifespan, SIGTERM closes the
  `CorpusStore` cleanly before the process exits.
- SQLite-backed `CorpusStore` survives daemon restarts, no more
  re-encode on every boot.
- Multi-tenant: `tenant_id` filter isolates per-corpus operations
  between callers behind a single daemon.
- Federated search: `POST /v1/search-many` fan-outs across multiple
  named corpora in one request.
- `DELETE /v1/corpora/{id}` for explicit cleanup.
- `adaptmem corpora` CLI subcommand group, `list`, `delete`,
  `search-many` talk to a running daemon.
- Optional shell tab-completion via `pip install adaptmem[shell]`
  (argcomplete).

### Added, v0.6 bench expansion
- CodeSearchNet python full bench (`benchmarks/codesearchnet_*`).
  FT-Code-300 / 1000 / 5000 checkpoints; full 22k test split.
  FT-Code-5000 hits R@1=0.926, MRR=0.952 vs off-the-shelf MiniLM
  baseline R@1=0.6477, MRR=0.7406 (+0.278 R@1, +0.211 MRR). The
  encoder FT recipe is not LongMemEval-specific.
- Matched-protocol harness (`benchmarks/mempal_bench_with_ft.py`)
  monkey-patches mempal's `_bench_embed_fn` with an FT encoder so the
  measurement runs end to end through mempal's own scorer, no shim
  shape divergence. Companion Colab notebook
  (`benchmarks/colab_mempal_matched_protocol.ipynb`) reproduces the
  raw / hybrid_v4 + FT-300 numbers on the public 500-query split.
- Cross-harness probe (`benchmarks/jphein_chunk_x_encoder.py`) reuses
  the mempalace fork's `chunk_strategy_ablation.py` with an FT-encoder
  swap, isolates encoder-axis lift inside someone else's
  chunking-axis harness.
- Paired bootstrap MRR helper (`benchmarks/bootstrap_paired_mrr.py`)
  + RRF surrogate fusion (`rrf_ensemble.py`, `rrf_ensemble_nway.py`)
  for adding statistical rigor to cross-harness comparisons.

### Added, v0.6 packaging / ops
- Helm chart (`charts/adaptmem`), templated equivalent of the example
  raw k8s manifests.
- `examples/k8s/` raw manifests for direct apply.
- `koyeb.yaml` for free-tier hosting.
- `Makefile` targets, `test`, `lint`, `typecheck`, `verify`.

### Changed in v0.6
- `server.embed` / `server.reindex` made async to prevent uvicorn
  deadlock on macOS when called from another async context.
- README cluster section, ROADMAP, `docs/metis_integration.md`,
  CONTRIBUTING, PROGRESS, issue templates and the encoder docstring
  no longer reference `halluguard`. It was never imported and never
  a runtime dependency; the docs implied an integration that did not
  exist.

### Fixed in v0.6
- `AdaptMem.load()` corpus-tsv parser choked on function bodies that
  contained newlines (CodeSearchNet); the eval path now bypasses the
  loader and constructs the SentenceTransformer directly.
- mypy `ignore_missing_imports` for `requests` in bench venvs (CI
  green on stricter type checks).
- Colab notebook `FT_MODEL` path corrected after the `model/model/`
  save-layout change.

### Security in v0.6
- README contact rewired to `hey@nakata.app` (public mailbox).

---

The sections below carry forward the v0.2 → v0.7 history that
shipped as `[Unreleased]` in the previous CHANGELOG; they are kept
under v0.6.0 because that is the first release tag that publishes
them to PyPI.

### Added, v0.7 Metis integration (PoC shipped)
- **`adaptmem.server`**, FastAPI app exposing `/healthz`, `/version`,
  `/embed`, `/reindex`, `/search`. Lazy encoder load (model only built
  on first embed/reindex call). Pydantic schemas at module level so
  FastAPI introspection works.
- **`adaptmem serve`** CLI subcommand, `--port`, `--host`, `--device`,
  `--uds` (Unix-domain-socket alternative for zero-TCP-overhead
  callers).
- **`[server]` optional dep**, `pip install "adaptmem[server]"` pulls
  fastapi + uvicorn[standard] + pydantic. Core install stays minimal.
- **`tests/test_server.py`**, 5 endpoint tests via FastAPI TestClient.
  Real encoder, real ranking, no mocks for the retrieval pass.
- **`docs/metis_integration.md`**, full ADR. Three bridge options
  considered (subprocess shell-out, PyO3, HTTP daemon); HTTP daemon
  chosen. v1 endpoint contract, compatibility matrix, failure modes,
  rollout plan.

### Added, v0.4 production-ready surface (closed)
- **mypy --strict pass.** `dict[str, Any]` annotations, `DataLoader[Any]`
  local annotation, `from typing import Any`, type-ignore on
  sentence_transformers.losses (3rd-party stub gap).
- **`release.yml`**, wheel build + sdist + tag-gated PyPI publish
  (skipped via shell guard when `PYPI_API_TOKEN` is absent).

### Added, bench results
- `benchmarks/results_minilm_baseline_400.json`, raw MiniLM 400q
  R@5=0.965, matches MemPalace published 0.966 within 0.1pt.
  Independent confirmation of the protocol.
- `benchmarks/results_minilm_baseline_50.json`, raw MiniLM 50q
  baseline alongside the BGE comparison.
- `benchmarks/results_bge_small_50.json`, BAAI/bge-small-en-v1.5 raw
  50q. Encoder swap = honest null result (R@5=0.98 vs MiniLM 0.98).
  The lift comes from fine-tuning, not the base model.
- `benchmarks/bench_st_inline.py`, minimal harness that imports
  helpers from `longmemeval_eval.py` but bypasses `cmd_test` (which
  deadlocks on Mac/Py3.14 + sentence-transformers + uvicorn-thread-
  pool cluster).

### Added, v0.4 production-ready surface
- Optional cross-encoder rerank stage on `AdaptMem.search()` (default
  off). Enable via `AdaptMem(rerank=True)`; CE candidates pulled from a
  widened bi-encoder set (`rerank_top_k or top_k * 3`). Lazy load.
  `RetrievalHit.score` is the CE score when reranking is on.
- `AdaptMem.add_corpus()` for streaming index updates, encode new
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

### Added, v0.2 first benchmark
- `benchmarks/longmemeval_eval.py`, train + test CLI matching the
  MemPalace eval protocol (per-question fresh corpus, user-only
  encoding, drop empty sessions, R@k macro mean).
- `benchmarks/results_ft300_direct.json`, first reproducible run
  (FT-300, R@1=0.915, R@5=0.995, 200 held-out questions, CPU).
- `benchmarks/results_ft200_direct.json`, sanity second run
  (FT-200, R@1=0.900, R@5=0.990).
- `benchmarks/results_ft100_400.json`, self-contained reproduce
  (FT-100 trained on the shipped 100/400 split, evaluated on the
  400 held-out questions): R@1=0.855, R@5=0.978, R@10=0.992. Includes
  per-question-type breakdown.
- `benchmarks/data/split_ids_100_400.json`, deterministic
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

## [0.1.0], 2026-04-26 (initial)
- `AdaptMem` skeleton with hard-negative mining + contrastive fine-tune
  (MultipleNegativesRankingLoss) + persistence.
- 9 unit tests (`test_normalise`, `test_miner`).
- Public roadmap (v0.2 → v0.5) with exit criteria.
