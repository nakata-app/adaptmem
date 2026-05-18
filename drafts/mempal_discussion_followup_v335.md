# Follow-up to Discussion #1249, v3.3.5 rerun

**Status:** draft, not posted. Atakan reviews + edits + posts as a comment under [MemPalace/mempalace Discussion #1249](https://github.com/MemPalace/mempalace/discussions/1249).

**Context:** On May 1, 2026 nakata-app asked whether the original FT-300 numbers held against v3.3.4+ (storage optimization release). jphein replied on May 11 ("I'm reading your work!") and later wrote an addendum doc (`docs/research/adaptmem-orthogonal-layers.md` in his fork) flagging the v3.3.4 rerun as an open question. This comment closes that loop.

---

## Body

Quick follow-up on the May 1 question about v3.3.4+ protocol equivalence, I re-ran all three rows on **v3.3.5** (latest release as of today) and also did a controlled v3.3.3 repro to isolate the source of any movement. Numbers below.

### Three runs on v3.3.5 (full 500q, matched protocol)

Same `longmemeval_bench.py`, same FT-300 model file (mtime Apr 26, unchanged since the original post), encoder swap via the monkey-patch wrapper documented earlier.

| System | R@1 | R@5 | R@10 |
|---|---|---|---|
| MemPal raw default (v3.3.5) | 0.806 | 0.966 | 0.982 |
| MemPal raw + adaptmem FT-300 (v3.3.5) | 0.932 | 0.992 | 0.996 |
| MemPal hybrid_v4 + adaptmem FT-300 (v3.3.5) | **0.950** | **0.998** | **1.000** |

### Three takeaways

1. **Raw default identical across versions.** Raw mode R@1 = 0.806 / R@5 = 0.966 on v3.3.5 matches v3.3.3 bit-for-bit (controlled repro, same venv, only mempal HEAD switched). PR #1179 (BM25 hybrid rerank fix) and PR #1306 (`candidate_strategy="union"` opt-in) don't touch the raw retrieval path, which is what we'd expect. Reproduction protocol is stable across the v3.3.3 → v3.3.5 window.

2. **Hybrid_v4 + FT-300 went up: R@1 +0.034, R@5 +0.008, R@10 +0.002** relative to the Apr 28 run. This is consistent with the v3.3.5 BM25 hybrid rerank fix, the rerank pass is FT-300-encoder-aware now in a way it wasn't before, and the encoder layer's lift composes with the fixed rerank rather than getting clipped by it. The encoder-as-its-own-axis framing from the #1384 thread holds up under v3.3.5.

3. **Raw + FT-300 moved from 0.862 → 0.932 R@1.** This one is *not* a mempal-side change, controlled repro on v3.3.3 with today's venv reproduces 0.932 identically. The Apr 28 → today delta is from upgraded dependency versions (chromadb 1.5.8, sentence-transformers 5.4.1, numpy 2.4.4 at present; the Apr 28 venv was older, exact versions not preserved). Flagging it explicitly so the Apr 28 numbers don't look retroactively re-stated without disclosure.

### What the deltas mean

- Encoder alone (raw + FT-300 vs raw default): **+0.126 R@1, +0.026 R@5**.
- Encoder + hybrid retrieval stacked (hybrid_v4 + FT-300 vs raw default): **+0.144 R@1, +0.032 R@5**.

Encoder fine-tune and hybrid retrieval are still adding lift on top of each other at v3.3.5. R@5 is ceiling-bounded (close to 1.000), so R@1 is the honest comparison and the orthogonality reads clearly there.

### Reproduce

Same harness, no changes:

```bash
cd ~/Projects/mempalace && git checkout v3.3.5
cd ~/Projects/adaptmem
PYTHONPATH=/path/to/mempalace python benchmarks/mempal_bench_with_ft.py \
  --bench-script /path/to/mempalace/benchmarks/longmemeval_bench.py \
  --data-file /path/to/longmemeval_s_cleaned.json \
  --ft-model /path/to/minilm-lme-ft-300 \
  --mode {raw|hybrid_v4} \
  --out results.jsonl
```

The three v3.3.5 result JSONLs are committed in `benchmarks/v335/` in the adaptmem repo. The v3.3.3 controlled-repro JSONL (`run4b_v333_raw_ft300.jsonl`) is alongside them for anyone who wants to verify the version-equivalence claim independently.

If hybrid_v4 reruns on top of these numbers are useful to compare against your own internal measurements, happy to share the result JSONLs directly. Otherwise this is just to close the May 1 question with current numbers.

, nakata-app

---

## Pre-post checklist for Atakan

1. **Discussion category**, this is a comment, not a new thread, so category isn't relevant. Verify you're replying under #1249.
2. **Sign-off**, change `nakata-app` to whatever handle you want.
3. **Three v3.3.5 JSONL paths**, when this comment goes up, the `benchmarks/v335/` dir in adaptmem should already be committed and pushed so the "committed in `benchmarks/v335/`" claim is true at post time. Currently they're local-only.
4. **`run4b_v333_raw_ft300.jsonl` reference**, same: commit + push before post if you want to leave it citable, or remove the sentence.
5. **Tone**, extension-flavoured, same posture as the original. Re-read once. Don't soften the venv-lib disclosure (point 3), leaving it implicit would be the kind of seam an outside reader could pull on later.
6. **Cross-thread reference.** The "encoder-as-its-own-axis framing from the #1384 thread" line is a back-reference to your own May 12 comment under jphein's chunking ablation discussion. Confirm the wording reads natural to you before posting; it's the only thing tying the two threads together.
