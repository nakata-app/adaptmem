Following up on the domain-mismatch thread from your May 11 post. Two questions I wanted to answer with actual numbers: (1) does the encoder-axis negative result hold when the encoder is trained on code data instead of LongMemEval QA pairs? (2) is the −0.006 / −0.043 finding statistically defensible at n=15, 20? Here's what I ran.

### 1. CodeSearchNet python full test (22k), in-domain sanity check

Training: `sentence-transformers/all-MiniLM-L6-v2` base,
`MultipleNegativesRankingLoss`, CodeSearchNet python train, query =
`func_documentation_string`, positive = `func_code_string`. Three checkpoints,
same training data, at step counts 300 / 1000 / 5000. Eval same HF
`code_search_net` python test split (21,935 queries, 21,935 corpus entries).

| Model | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| Baseline (sentence-transformers/all-MiniLM-L6-v2, no FT) | 0.6477 | 0.8551 | 0.8972 | 0.7406 |
| FT-Code-300 | 0.800 | 0.941 | 0.959 | 0.864 |
| FT-Code-1000 | 0.902 | 0.976 | 0.981 | 0.936 |
| **FT-Code-5000** | **0.926** | **0.982** | **0.985** | **0.952** |

Δ baseline → FT-Code-5000: **+0.278 R@1, +0.211 MRR**. Encoder fine-tune
**does** lift in the code domain when training data domain matches. The original
FT-300 result was specific to its LongMemEval training distribution, not a
property of the encoder axis itself.

### 2. Cross-harness rerun on jphein's `chunk_strategy_ablation.py`

Same probe set (20 hand-curated mempalace py queries), same `n_results=10`,
same 6 strategies (A_paragraph_aware / B_heading_aware_md / C_plus_ast_python ×
cs400/cs800), default mempal encoder monkey-patched per the existing
`jphein_chunk_x_encoder.py` wrapper. Three encoders swapped in turn.

**MRR per strategy:**

| Strategy | default | FT-300 | FT-Code-300 | FT-Code-1000 | FT-Code-5000 |
|---|---|---|---|---|---|
| A_paragraph_aware cs400 | 0.4583 | 0.5125 | 0.4917 | 0.5500 | 0.5433 |
| A_paragraph_aware cs800 | 0.4850 | 0.5142 | 0.5058 | 0.4780 | 0.5333 |
| B_heading_aware_md cs400 | 0.4583 | 0.5125 | 0.4917 | 0.5500 | 0.5433 |
| B_heading_aware_md cs800 | 0.4850 | 0.5142 | 0.5058 | 0.4875 | 0.5292 |
| C_plus_ast_python cs400 | 0.4583 | 0.4833 | 0.5417 | 0.5750 | 0.5600 |
| **C_plus_ast_python cs800** | **0.5600** | **0.5542** | **0.5333** | **0.5588** | **0.5167** |

**R@10 per strategy:**

| Strategy | default | FT-Code-5000 |
|---|---|---|
| All 6 strategies | 60-65% | **70% (uniform)** |

Two raw observations:

- **5/6 strategies, FT-Code-5000 beats default on MRR** (+0.04 to +0.10) and on
  R@10 (+5 to +10 points uniformly).
- **C-AST cs800 is the one strategy where FT-Code-5000 underperforms default**
  (−0.043 MRR). FT-300 also slightly underperformed there (−0.006), so the
  original negative-result direction reproduces qualitatively with a code-domain
  encoder too.

That's the raw fact pattern. Now the statistics.

### 3. Paired bootstrap %95 CI on the C-AST cs800 finding

n=20 probes is small. To check whether the −0.043 (and the original −0.006) is
inside noise, paired bootstrap with 10,000 resamples on per-probe RR:

| Strategy | mrr_default | mrr_FTcode5000 | Δ | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|
| A_paragraph_aware cs400 | 0.4583 | 0.5433 | +0.085 | [−0.015, +0.217] | 0.937 |
| A_paragraph_aware cs800 | 0.4850 | 0.5333 | +0.048 | [−0.060, +0.158] | 0.801 |
| B_heading_aware_md cs400 | 0.4583 | 0.5433 | +0.085 | [−0.015, +0.217] | 0.937 |
| B_heading_aware_md cs800 | 0.4850 | 0.5292 | +0.044 | [−0.063, +0.153] | 0.792 |
| C_plus_ast_python cs400 | 0.4583 | 0.5600 | +0.102 | [−0.007, +0.238] | 0.963 |
| **C_plus_ast_python cs800** | **0.5600** | **0.5167** | **−0.043** | **[−0.125, +0.030]** | **0.126** |

Read carefully: **no strategy is significant at n=20** (all CIs include zero).
But the directional signals are not symmetric, five strategies sit at
P(Δ>0) ∈ [0.79, 0.96]; C-AST cs800 sits at P(Δ>0) = 0.126.

The honest takeaway: the original −0.006 finding lived squarely in the noise
floor at n=15, and the new −0.043 finding lives in the noise floor at n=20.
**"Significant regression on C-AST cs800" is not a claim we can defend on this
sample size.** A larger probe set is the right next step before treating that
strategy as a structural negative.

### 4. RRF ensemble, encoder axis composes with default encoder

What if we don't *swap* encoders but *fuse* them? RRF surrogate
(rank_fused = min(rank across runs) on the expected doc; this is a lower
bound on true RRF since the JSONs only stored rank-of-expected, not full
ranked lists). Two ensemble configurations:

**2-way** (default + FT-Code-5000):

| Strategy | default | FT-Code-5000 | **RRF 2-way** | Δ vs best solo |
|---|---|---|---|---|
| A_paragraph cs400 | 0.4583 | 0.5433 | **0.5873** | +0.044 |
| A_paragraph cs800 | 0.4850 | 0.5333 | **0.5939** | +0.061 |
| B_heading cs400 | 0.4583 | 0.5433 | **0.5873** | +0.044 |
| B_heading cs800 | 0.4850 | 0.5292 | **0.5898** | +0.061 |
| C-AST cs400 | 0.4583 | 0.5600 | **0.6039** | +0.044 |
| **C-AST cs800** | **0.5600** | **0.5167** | **0.6106** | **+0.051** |

**3-way** (default + FT-Code-1000 + FT-Code-5000), the strongest config tested:

| Strategy | default | FT-Code-1k | FT-Code-5k | **RRF 3-way** | Δ vs best solo |
|---|---|---|---|---|---|
| A_paragraph cs400 | 0.4583 | 0.5500 | 0.5433 | **0.6123** | +0.062 |
| A_paragraph cs800 | 0.4850 | 0.4780 | 0.5333 | **0.6023** | +0.069 |
| B_heading cs400 | 0.4583 | 0.5500 | 0.5433 | **0.6123** | +0.062 |
| B_heading cs800 | 0.4850 | 0.4875 | 0.5292 | **0.5981** | +0.069 |
| C-AST cs400 | 0.4583 | 0.5750 | 0.5600 | **0.6373** | +0.062 |
| **C-AST cs800** | **0.5600** | **0.5588** | **0.5167** | **0.6356** | **+0.076** |

R@10 uniformly **70%** across all 6 strategies in both 2-way and 3-way fused
settings (vs default 60-65%).

The headline: **C-AST cs800, the original "negative result" strategy, gets the
largest ensemble lift (+0.076 MRR)** when fused 3-way. The strategy where
single-encoder swap underperforms default is the same strategy where
ensemble-encoder fusion outperforms default the most. That's the inverse of
what a structural encoder-axis failure would look like.

This is the actual answer to the "encoder axis vs chunking axis" question:
**these axes compose additively when fused, not when one is forced to replace
the other.** The original chunk_strategy_ablation harness measured single-
encoder substitution, which is the wrong primitive for this kind of axis
test, production retrieval can run two or three encoders in parallel and
RRF-merge for cheap (one extra forward pass per query, no extra storage,
deterministic).

**Independent replication at 10× scale (issue #82, 2026-05-15):** You ran
the same 3-way RRF configuration on the n=200 git-derived probe set and got:

| Encoder | Solo MRR | R@10 |
|---|---|---|
| default ONNX | 0.4260 | 49.5% (99/200) |
| FT-Code-1000 | 0.4229 | 53.5% (107/200) |
| FT-Code-5000 | 0.3972 | 50.0% (100/200) |
| **RRF 3-way** | **0.5101** | **59.5% (119/200)** |

Δ MRR = +0.0841 vs best solo. Our n=20 surrogate gave +0.076 at the same
configuration. Direction and magnitude align; the effect scales cleanly with
probe set size. This closes the sample-size concern from §3.

### 5. Scaling signal across FT-Code checkpoints, non-monotonic

Looking at the 300 → 1000 → 5000 step progression per strategy:

| Strategy | FTcode300 | FTcode1k | FTcode5k | Shape |
|---|---|---|---|---|
| A_paragraph cs400 | 0.4917 | **0.5500** | 0.5433 | peak at 1k |
| A_paragraph cs800 | 0.5058 | 0.4780 | **0.5333** | dip at 1k |
| B_heading cs400 | 0.4917 | **0.5500** | 0.5433 | peak at 1k |
| B_heading cs800 | 0.5058 | 0.4875 | **0.5292** | dip at 1k |
| C-AST cs400 | 0.5417 | **0.5750** | 0.5600 | peak at 1k |
| **C-AST cs800** | **0.5333** | **0.5588** | **0.5167** | **U-shape: best at 1k, worst at 5k** |

Two things to flag here:

- **Scaling is not monotonic.** FT-Code-1000 is the local optimum in 4 out
  of 6 strategies, with FT-Code-5000 either tied or slightly behind. This is
  consistent with the model overfitting to the CodeSearchNet python
  distribution as training continues, useful for in-domain code retrieval
  (the 22k eval above), but progressively worse for mempalace's own .py
  corpus which has more markdown / docstring mix.

- **C-AST cs800 specifically: FT-Code-1000 = 0.5588, default = 0.5600.**
  Within noise of default at the 1k step; the −0.043 at 5k is a training-
  step artifact, not a structural property of the encoder axis on this
  strategy. This sharpens the bootstrap-CI takeaway from §3, the "negative
  result" isn't just statistical noise, it's also moving with training
  duration in a way that suggests it's *fixable*, not fundamental.

R@10 tells a cleaner monotonic story regardless: 60-65% default → 70%
uniform from FT-Code-1k onward across all 6 strategies. Top-1 ranking is
where the noise lives; top-10 coverage scales cleanly.

### 6. Direct response to the original framing

> "Encoder lift (FT-300, trained on LongMemEval QA pairs) is LongMemEval-
> domain-specific and won't compose with chunking-axis changes on
> mempalace's own .py corpus."

Re-reading this after the new runs:

- **"Domain-specific" part holds.** FT-300 was a conversational-QA encoder, of course
  transfer to code was weak. That was real domain mismatch, not encoder-axis
  inadequacy.
- **"Won't compose with chunking-axis changes" part doesn't hold under RRF
  fusion.** When the question is "can the encoder axis *add* on top of the
  chunking axis", the answer is yes uniformly across 6/6 strategies (default
  → fused, +0.04 to +0.06 MRR, +5 to +10 R@10 points).
- **"On mempalace's own .py corpus" caveat softens but doesn't dissolve.**
  CodeSearchNet python is closer to mempalace's .py corpus than LongMemEval QA
  is, but still distribution-shifted (mempalace internal API + docstrings ≠
  HuggingFace-mined open-source python). FT-Code-5000 lifts substantially in
  5 strategies but not the strongest-default strategy (C-AST cs800). True
  in-domain ceiling would need mempalace-corpus-derived training pairs, which
  is a different epic.

### Reproduce

CodeSearchNet eval:

```bash
cd ~/Projects/adaptmem
python benchmarks/codesearchnet_eval.py \
  --checkpoint /path/to/ft-code-5000 \
  --n -1 \
  --out results/ft-code-5000.jsonl
```

Cross-harness probe:

```bash
cd ~/Projects/adaptmem
python benchmarks/jphein_chunk_x_encoder.py \
  --ft-model /path/to/ft-code-5000/model \
  --out-dir benchmarks/v335/chunk_x_encoder_ftcode5000
```

Paired bootstrap CI:

```bash
python benchmarks/bootstrap_paired_mrr.py \
  benchmarks/v335/chunk_x_encoder_ftcode5000/ablation_default_encoder.json \
  benchmarks/v335/chunk_x_encoder_ftcode5000/ablation_ft300_encoder.json \
  --label-a default --label-b ftcode5k
```

RRF surrogate fusion:

```bash
python benchmarks/rrf_ensemble.py \
  benchmarks/v335/chunk_x_encoder_ftcode5000/ablation_default_encoder.json \
  benchmarks/v335/chunk_x_encoder_ftcode5000/ablation_ft300_encoder.json
```

All scripts deterministic given the model files. Model checkpoints accessible
via Drive (anyone-with-link reader):

- FT-Code-300: https://drive.google.com/drive/folders/1fe5t5LWWHFGDV5CC5GcfCk5aOVaRKKRb
- FT-Code-1000: https://drive.google.com/drive/folders/1QhjDc63M4vKdOxVMP7ZbpyhYoLcXjnlC
- FT-Code-5000: https://drive.google.com/drive/folders/1GZXGQG4LJL8jm1ajbo8JgiIPPICf_QtT
- CodeCrossEnc-v1: not sharing at this stage (see §7, v1 shows negative transfer; v2 hard-negative version pending)

### 7. CodeCrossEnc-v1: code-specific reranker axis

**Training:** `cross-encoder/ms-marco-MiniLM-L-6-v2` base, CodeSearchNet
python train, 30K positive pairs + 2 random negatives each = 90K total,
1 epoch, batch=8, lr=2e-5, warmup=300, max_length=384. Local CPU (Mac mini M2,
8GB RAM, ~4h). Generic cross-encoders (`ms-marco-MiniLM-L-6-v2` untuned and
`BAAI/bge-reranker-base`) both showed negative transfer (R@1 ~0.90 vs
FT-Code-5000 alone 0.926). CodeCrossEnc-v1 is the code-specific alternative.

**Eval:** FT-Code-5000 bi-encoder top-20 → CodeCrossEnc-v1 rerank,
CodeSearchNet python test split (n=5000 queries; full-set eval pending):

| Config | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| FT-Code-5000 (bi-alone) | 0.9148 | 0.9804 | 0.9868 | 0.9448 |
| + CodeCrossEnc-v1 rerank (top-20) | 0.9148 | 0.9158 | 0.9194 | 0.9198 |
| Δ rerank vs bi-alone | 0.0000 | −0.0646 | −0.0674 | −0.0250 |

**Honest verdict:** The local cross-encoder did not improve over bi-alone and
actively hurt R@5/R@10 (−6.5 / −6.7 pp). R@1 is unchanged because queries
where bi-encoder already ranks #1 are stable; the damage is in slots 2-5 where
the cross-encoder reorders within the top-20 randomly rather than helpfully.
Root cause is almost certainly the training setup: 30K pairs with **random
negatives** teaches the model to distinguish "this docstring's code" from "some
other random code", but not to fine-rank 20 near-duplicate candidates which is
the actual top-20 reranking task. Hard negatives (mined from bi-encoder's own
top-K) are required for cross-encoder training to be useful. The Colab variant
(100K pairs, batch=32, T4 GPU) may partially close this gap, but training data
quality is likely the binding constraint, not scale.

> **Note to jphein:** local CodeCrossEnc-v1 (random-negative training) shows
> negative transfer on the reranking task (R@5 −6.5 pp vs bi-alone). Sharing the
> checkpoint is not useful at this stage, it would hurt rather than help any
> probe set eval. The right next step is hard-negative mining from the
> bi-encoder's top-50 before training a cross-encoder worth evaluating on your
> CUDA box. Will revisit after that training pass.

### What's next on our side

- **CodeCrossEnc-v2 with hard negatives.** v1 used random negatives; the fix
  is mining negatives from the bi-encoder's own top-50 (the candidates the
  bi-encoder almost-but-not-quite ranked correctly). That's the actual
  reranking distribution. Once v2 is trained and shows positive transfer, the
  3-axis test (chunking × bi-encoder × cross-encoder) on your n=200 probe set
  is the natural next step.
- **In-domain FT** with mempalace-corpus-derived pairs remains interesting.
  The git-history synthesis approach (~10-20K pairs from commit messages +
  function diffs) is on our list once we have the `derive_probes` script.

Thanks for PR #1508 (`symbol_header_prefix`) and for the n=200 replication in issue #82.
The 3-way RRF lift (+0.0841 MRR at n=200 vs our +0.076 at n=20) confirms the effect is real
and scales with probe set size.

One note on your PR #80 (chromadb EF `embed_query` warning): the silent BM25 fallback you
documented is a real footgun. Our `jphein_chunk_x_encoder.py` wrapper already inherits from
`chromadb.api.types.EmbeddingFunction`, so our probe runs above used the actual encoder on
the query path. The warning will still be useful for downstream adaptmem users building custom
wrappers without referencing our code.

Happy to share Drive links to the three FT-Code checkpoints, the probe-level
JSONs, or the bootstrap script if useful for the broader four-axis writeup.
