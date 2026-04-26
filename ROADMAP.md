# adaptmem roadmap

Status — `v0.1` (April 26 2026): public API stable, hard-negative mining + contrastive FT + persistence + 9 unit tests passing. No real benchmark integration yet.

The path below is opinionated. Each milestone has a concrete exit criterion and a rough effort estimate (CPU-only on a Mac mini).

---

## v0.2 — first real benchmark (target: 1-2 days)

**Goal:** prove the README's `R@5 = 0.9950` claim is reproducible from scratch by anyone who clones the repo.

- [ ] `benchmarks/longmemeval_eval.py`
  - `--mode train`: load `longmemeval_s_cleaned.json`, take first N questions, build LabelledQuery list, call `AdaptMem.train`, save model. Report training stats.
  - `--mode test`: load saved model, evaluate on remaining questions, report R@1 / R@5 / R@10.
  - Per-question protocol (mempal-compatible): each question's `haystack_sessions` = fresh corpus; `answer_session_ids` = relevant ids.
  - User-only encoding: `"\n".join(t.content for t in session if t.role == "user")`.
- [ ] Sanity reproduce on 100 train / 400 test split (matches our existing FT-100). Expected R@5 ≥ 0.985.
- [ ] Full reproduce on 300 train / 200 test split. Expected R@5 ≥ 0.99.
- [ ] Commit a `benchmarks/results.json` with the numbers and the `git rev-parse HEAD` they came from. Reproducibility is the deliverable.
- [ ] README replaces the placeholder table with the real numbers + the exact CLI commands to reproduce.

**Exit:** a stranger runs `make bench-longmemeval` (or the equivalent two-line script) and gets R@5 within ±0.01 of our headline.

---

## v0.3 — second benchmark + multi-encoder (target: 3-5 days)

**Goal:** show domain adaptation generalises beyond the dataset that produced it. Add a second public benchmark and a second base encoder.

- [ ] `benchmarks/convomem_eval.py` — Salesforce ConvoMem (cited in MemPal's BENCHMARKS.md as "MemPal 92.9%"). Different domain (general conversation, multi-turn QA), different label distribution.
  - Target: adaptmem trained on ConvoMem train split beats vanilla MiniLM by ≥3 points R@5.
- [ ] `benchmarks/membench_eval.py` — ACL 2025 MemBench (mempal raw 80.3%). Even more out-of-distribution.
  - Target: ≥1 point lift over baseline.
- [ ] Encoder swap support: `AdaptMem(base_model="BAAI/bge-small-en-v1.5")`. Pick a second encoder family.
  - Target: bge-small + adaptmem on LongMemEval reaches the same 0.99 ceiling MiniLM hit, faster or with fewer pairs.
- [ ] Document **when adaptmem helps** (in-distribution test set, ≥100 labelled queries) and **when it doesn't** (cross-domain transfer, fewer than 50 queries).
- [ ] Per-question-type breakdown in the LongMemEval bench (matches what MemPal's table publishes), so `temporal-reasoning` / `multi-session` / `single-session-preference` etc. are visible separately.

**Exit:** three benchmark tables in the README. At least two of them are not LongMemEval.

---

## v0.4 — robustness + APIs (target: 1 week)

**Goal:** make the package something a stranger can import in production, not just a benchmark harness.

- [ ] **Cross-encoder rerank** as a built-in optional second stage (`AdaptMem.rerank=True`). Default model: `cross-encoder/ms-marco-MiniLM-L-12-v2`. Already shown to add R@1 +5 points on our internal LongMemEval runs.
- [ ] **Streaming index updates** — add new corpus chunks without re-encoding the whole corpus. Persist embeddings incrementally.
- [ ] **Persistence on disk:** swap in-memory numpy index for an on-disk Parquet index when corpus > 50k chunks. Keep the API identical.
- [ ] **CI**: GitHub Actions matrix (Python 3.10/3.11/3.12), lint (`ruff`), tests, release-on-tag wheel build to PyPI.
- [ ] **Type stubs** (`py.typed` marker), strict-mode mypy clean.
- [ ] **Token cost report** in `train()` output: how many tokens were encoded, how many GPU/CPU seconds were spent. Helps users budget.
- [ ] **`adaptmem evaluate`** CLI subcommand: take a saved model + a labelled queries file, dump R@k.

**Exit:** `pip install adaptmem` from PyPI. A second user can train and serve a domain encoder without reading source code, just the README.

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
- [ ] Open a **GitHub Discussion** (not an Issue, not a PR yet) on `milla-jovovich/mempalace` titled along the lines of "Domain-adaptive fine-tune as orthogonal R@5 lift on top of MemPal raw retrieval". Link to adaptmem repo, paste reproduce commands.
- [ ] Aim the technical content at **the developer (Ben Sigman)**, not the public face. Use the pull-request voice: "here's a number, here's the reproduce, here's the integration sketch."
- [ ] If they're interested → propose a `mempal-adapt` integration: their store layer + adaptmem's encoder-tuning step.
- [ ] If they pass → adaptmem stays standalone. Either outcome is fine.

**Exit:** discussion is open with reproducible numbers and a clear integration proposal. The framing is "we extended your work" not "we beat your benchmark."

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
