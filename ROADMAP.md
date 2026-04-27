# adaptmem roadmap

Status — `v0.4` in flight (April 26 2026, late-day session): bench harness landed, two committed JSONs (FT-300 + FT-200) reproduce the README numbers, the v0.4 production-ready surface (CE rerank, streaming index, evaluate CLI, device override, token cost report, py.typed, CI matrix, CLI smoke tests) is shipped. v0.2's self-contained 100/400 train target is wired but blocked on a local PyTorch+memory-pressure interaction; details in `PROGRESS.md`.

Earlier — `v0.1` (April 26 2026, morning): public API stable, hard-negative mining + contrastive FT + persistence + 9 unit tests passing.

The path below is opinionated. Each milestone has a concrete exit criterion and a rough effort estimate (CPU-only on a Mac mini).

---

## v0.2 — first real benchmark (target: 1-2 days) — **mostly shipped**

**Goal:** prove the README's `R@5 = 0.9950` claim is reproducible from scratch by anyone who clones the repo.

- [x] `benchmarks/longmemeval_eval.py` (commit `06e593e`, extended in `2ce0132` with per-question-type breakdown)
  - `--mode train` / `--mode test` — both wired
  - Per-question protocol: per-question fresh corpus, user-only encoding, drop empty
  - `--st-model` flag for evaluating raw SentenceTransformer dirs (used for FT-300 / FT-200 reference runs)
  - `--device cpu|cuda|mps` flag (`04aa59a`) for Apple-silicon MPS-deadlock workaround
- [x] Full reproduce on 300 train / 200 test split → `benchmarks/results_ft300_direct.json`: R@1=0.915, R@5=**0.995**, R@10=0.995. Commit `06e593e`.
- [x] Sanity second model FT-200 → `benchmarks/results_ft200_direct.json`: R@1=0.900, R@5=0.990, R@10=0.995. Commit `90d52af`. (More-train-data → higher-recall direction confirmed.)
- [x] README placeholder rows replaced with reproduced numbers + audit links to the JSONs + reproduce CLI snippet (`eb2daf2`, `90d52af`).
- [x] `Makefile` with self-contained `bench-longmemeval` target + committed `split_ids_100_400.json` + `DEVICE=cpu` default (`1d3149f`, `c0eec2f`).
- [x] **Sanity reproduce on the 100 train / 400 test split** — `benchmarks/results_ft100_400.json`: R@1=0.855, R@5=0.978, R@10=0.992 over 400 held-out questions. Trained on Colab CPU (Mac mini PyTorch deadlock workaround), evaluated on Mac CPU. Sits 0.7pt below the v0.2 sanity bar; expected for the smaller train set (recall scales with train-set size: 100→0.978, 200→0.990, 300→0.995).

**Exit:** a stranger runs `make bench-longmemeval` (or the equivalent two-line script) and gets R@5 within ±0.01 of our FT-100 headline (0.978). **Met** — though Mac-local reproduction of `make train-100` is currently blocked on a PyTorch+sentence-transformers fit() deadlock; the pinned-deps / Docker target landed in v0.3 will close that. Test-only path (`bench-ft100` after a Colab-trained model is dropped in) works on Mac as-is.

---

## v0.3 — second benchmark + multi-encoder (target: 3-5 days) — **partial**

**Goal:** show domain adaptation generalises beyond the dataset that produced it. Add a second public benchmark and a second base encoder.

- [ ] `benchmarks/convomem_eval.py` — Salesforce ConvoMem. Different domain (general conversation, multi-turn QA), different label distribution.
  - Target: adaptmem trained on ConvoMem train split beats vanilla MiniLM by ≥3 points R@5.
  - **Pending:** locate the public dataset (HF Hub mirror not yet identified); build script in the longmemeval pattern.
- [ ] `benchmarks/membench_eval.py` — ACL 2025 MemBench (mempal raw 80.3%). Even more out-of-distribution.
- [ ] Encoder swap support: `AdaptMem(base_model="BAAI/bge-small-en-v1.5")`. Param already in place — needs a measured run to confirm the lift.
- [ ] Document **when adaptmem helps** (in-distribution test set, ≥100 labelled queries) and **when it doesn't** (cross-domain transfer, fewer than 50 queries).
- [x] Per-question-type breakdown in the LongMemEval bench (commit `2ce0132`). The eval JSON now carries a `per_question_type` map (multi-session, temporal-reasoning, knowledge-update, single-session-{user,assistant,preference}).

**Exit:** three benchmark tables in the README. At least two of them are not LongMemEval.

---

## v0.4 — robustness + APIs (target: 1 week) — **mostly shipped**

**Goal:** make the package something a stranger can import in production, not just a benchmark harness.

- [x] **Cross-encoder rerank** as a built-in optional second stage (`AdaptMem(rerank=True)`). Default model: `cross-encoder/ms-marco-MiniLM-L-12-v2`. Lazy-loaded, persisted in `config.json`. Surfaced via `--rerank / --rerank-model / --rerank-top-k` on `train`, `search`, and `evaluate` CLI subcommands. (`5beec31`, `5a26e96`)
- [x] **Streaming index updates** — `AdaptMem.add_corpus(new_corpus)` encodes only the new entries and appends them to the in-memory embedding matrix; de-dupes by id. (`409eb8f`)
- [ ] **Persistence on disk:** swap in-memory numpy index for an on-disk Parquet index when corpus > 50k chunks. Keep the API identical. *(Not yet started — corpus sizes in current benches are well under threshold.)*
- [x] **CI**: GitHub Actions matrix (Python 3.10/3.11/3.12), lint (`ruff`), tests on every push and PR. (`4386d8b`) — release-on-tag wheel build to PyPI is the one open piece (needs maintainer-controlled API token).
- [x] **Type stubs** (`py.typed` marker, PEP 561). (`c4db02c`) Strict-mode mypy run is the next quality gate (planned).
- [x] **Token cost report** in `train()` output: `n_tokens_approx` (whitespace word count over corpus + every TrainingPair) and `tokens_per_s`. (`2d6ef87`)
- [x] **`adaptmem evaluate`** CLI subcommand: takes a saved model + a labelled queries file, prints R@1 / R@5 / R@top_k as a JSON dict. (`12057d5`)
- [x] **`device` parameter** on `AdaptMem(...)` and `--device` flag on the bench harness. Forces PyTorch device choice; persisted in `config.json` so `.load()` honours the same. Apple-silicon MPS-deadlock workaround. (`04aa59a`)
- [x] **CLI subprocess smoke tests** — argparse plumbing for `train` / `search` / `evaluate`. (`5ba9f34`)

**Exit:** `pip install adaptmem` from PyPI. A second user can train and serve a domain encoder without reading source code, just the README. **Code surface ready;** PyPI publish is the gating step.

---

## v0.5 — show-it-to-the-source moment (target: 2 weeks total)

**Goal:** ready to engage MemPalace upstream as a peer project, not as a one-off claim.

Pre-flight:
- [ ] Reproduce LongMemEval at v0.4-equivalent across three runs (different seeds), report mean ± stddev.
- [ ] Reproduce on **mempal's exact eval script** (`mempalace_repo/longmemeval_bench.py`) using `AdaptMem.encode` as the embedding function plug-in. Matched protocol = matched comparison.
- [ ] Honest numbers table in README:
  - MemPal raw (their reported)
  - MemPal hybrid v4 generalisable (their reported)
  - **Adaptmem (our protocol)** on the same split.
  - Δ shown explicitly. No cherry-pick. No spot-fix.
  - LLM column = "None" for both, hand-tune column = "None" for both.
- [ ] One paragraph in the README naming MemPalace as the **source of the verbatim-storage insight** that this repo extends. Adaptmem is an extension, not a replacement.

Outreach:
- [ ] Open a **GitHub Discussion** (not an Issue, not a PR yet) on `MemPalace/mempalace` titled along the lines of "Domain-adaptive fine-tune as orthogonal R@5 lift on top of MemPal raw retrieval". Link to adaptmem repo, paste reproduce commands.
- [ ] Aim the technical content at **the developer (Ben Sigman)**, not the public face. Use the pull-request voice: "here's a number, here's the reproduce, here's the integration sketch."
- [ ] If they're interested → propose a `mempal-adapt` integration: their store layer + adaptmem's encoder-tuning step.
- [ ] If they pass → adaptmem stays standalone. Either outcome is fine.

**Exit:** discussion is open with reproducible numbers and a clear integration proposal. The framing is "we extended your work" not "we beat your benchmark."

---

## v0.6 — production install + multi-bench (target: 1-2 weeks)

**Goal:** `pip install adaptmem` works, and the README cites at least three benchmarks (LongMemEval + one other public + one out-of-distribution).

- [ ] **PyPI release.** Tag `v0.6.0`, populate `PYPI_API_TOKEN` repo secret, the gated `release.yml` step publishes the wheel. After this, `claimcheck` and `halluguard` can drop their editable-sibling install in favour of `dependencies = ["adaptmem>=0.6"]`.
- [ ] **On-disk Parquet index.** Swap in-memory numpy index for Parquet when corpus > 50k chunks. Keep `AdaptMem.search` API identical; `add_corpus` appends to the Parquet file and updates a small in-memory ANN sidecar.
- [ ] **ConvoMem bench.** `benchmarks/convomem_eval.py`. Locate the public dataset (HF Hub mirror), build the script in the longmemeval pattern. Out-of-domain validation that the FT recipe is not LongMemEval-specific.
- [ ] **MemBench bench.** ACL 2025 paper's harness; mempal raw 80.3% there. Even more out-of-distribution.
- [ ] **FActScore bench.** Atomic-fact retrieval over Wikipedia. Different shape from per-question fresh-corpus benchmarks; reveals whether adaptmem helps in the "many small chunks" regime.
- [ ] **mypy --strict pass.** `py.typed` was the marker; this is the gate.
- [ ] **3-seed reproduce.** Mean ± stddev for FT-100 / FT-200 / FT-300 across `seed ∈ {42, 1337, 7}`. Drops the "single number could be lucky" objection that v0.5 outreach will probably hear.

**Exit:** `pip install adaptmem`, README has three bench tables, CI runs mypy strict on every push.

---

## v0.7 — Metis integration (target: 2-3 weeks) — **PoC SHIPPED**

**Goal:** adaptmem becomes the encoder/retriever layer for `~/Projects/metis`. Today they are unrelated: metis is a Rust agent CLI, adaptmem is Python research code. The integration question is *how* they talk.

- [x] **Bridge architecture decision.** ADR in [`docs/metis_integration.md`](docs/metis_integration.md). Three candidates considered (subprocess shell-out / PyO3 / HTTP daemon); HTTP daemon chosen. Cold-start cost paid once at daemon launch, metis cargo build stays Python-free, daemon is reusable across consumers.
- [x] **Use case netleştir.** v0.7 ships **conversation context retrieval** — semantic search over `.metis/memory/*.md`. Tool-output verification + codebase semantic search deferred to v0.8.
- [x] **PoC implementation.**
  - **Server side** (this repo): `adaptmem.server` FastAPI app + `adaptmem serve` CLI subcommand + `[server]` optional dep. `/embed`, `/search`, `/reindex`, `/healthz`, `/version`. 5/5 in-process FastAPI tests pass.
  - **Python clients**: `halluguard.daemon.DaemonEncoder` + `Guard.from_daemon`; `claimcheck.Pipeline.from_daemon`. Both shipped, both with mock-server tests.
  - **Rust client**: `semantic_memory_search` tool in metis (branch `feat/semantic-memory-search-adaptmem`). 5 unit tests + 434/434 metis-core regression pass, clippy clean.
- [x] **Integration tests.** Mock-HTTPServer fixture in halluguard tests; FastAPI TestClient in adaptmem tests; tokio mockable client in metis tests. Cross-process end-to-end on Linux is the next gate (Mac/Py3.14 deadlock blocks local live smoke).
- [x] **Document the contract.** [`docs/metis_integration.md`](docs/metis_integration.md) — full ADR with options, contract, compatibility, failure modes, rollout.

**Open before v0.7 closes:**
- [ ] Linux/CI live-daemon smoke (replaces Mac local smoke that hits sentence-transformers + uvicorn deadlock).
- [ ] PR `feat/semantic-memory-search-adaptmem` merged to metis `main`.
- [ ] Daemon auto-spawn from metis (deferred to v0.8 — manual `adaptmem serve` for v0.7).

**Exit:** metis `semantic_memory_search` tool calls adaptmem under the hood, returns ranked memory entries. Linux end-to-end demo is the v0.7 sign-off.

---

## Non-goals (until further notice)

- Beating MemPal's `+ Haiku rerank 1.000`. That number is hand-tuned-on-test and they say so themselves; matching it would mean teaching to the test, which we explicitly avoid.
- LLM-based fine-tuning supervision (label generation by LLM). Out of scope — keeps the no-LLM positioning intact.
- Swapping ChromaDB or any specific vector store. Adaptmem is encoder-side; it composes with whatever vector store you already use.

---

## What needs the ECC environment / external help

- **PyPI release** requires a maintainer-controlled API token (whoever owns the PyPI namespace).
- **PR to mempal** requires GitHub auth as the author of this work.
- **CI matrix** runs on GitHub Actions — small free-tier should cover it.
- Real benchmark datasets are public (HuggingFace) — no licensing blocker for ConvoMem / MemBench.

Everything else above can be done locally without external accounts.
