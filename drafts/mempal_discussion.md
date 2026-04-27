# Mempal Discussion taslağı

**Status:** draft, not posted. Atakan should review + edit + post on
[`MemPalace/mempalace` Discussions](https://github.com/MemPalace/mempalace/discussions).

**Category:** "Show and tell" or "Ideas" (whichever the maintainers
prefer for community-extension posts).

**Tone:** "we extended your work" — not "we beat your benchmark." The
mempal team published an open project; we're contributing back.

---

## Title (≤72 chars)

`Domain-adaptive fine-tune as orthogonal R@5 lift on top of MemPal raw`

(Variant if too long: `R@5 lift via domain-adaptive bi-encoder fine-tune (orthogonal to MemPal raw)`)

---

## Body

Hi MemPal team,

We've been using LongMemEval to evaluate a small open-source library
called [`adaptmem`](https://github.com/nakata-app/adaptmem) — a 200-
line hard-negative mining + contrastive fine-tune wrapper around
SentenceTransformers — and the numbers we got line up cleanly with
the work you've already published. Wanted to share back, see if
it's interesting.

### What we measured

Same dataset (`longmemeval_s_cleaned.json`), same protocol (per-
question fresh corpus, user-only encoding), same encoder family
(MiniLM-L6, ~90MB). Only the **fine-tune step** is different.

| System | R@1 | R@5 | n | LLM | Hand-tune |
|---|---|---|---|---|---|
| MemPal raw (your published) | — | 0.966 | 500 | — | — |
| MemPal hybrid v4 (your published) | — | 0.984 | 500 | — | — |
| **Our raw MiniLM (independent eval)** | 0.795 | **0.965** | 400 | — | — |
| adaptmem FT-100 dense | 0.855 | 0.978 | 400 | — | — |
| adaptmem FT-200 dense | 0.900 | 0.990 | 200 | — | — |
| **adaptmem FT-300 dense** | **0.915** | **0.995** | 200 | — | — |

Two findings worth flagging:

1. **Our raw MiniLM 400q R@5 = 0.965 matches your published raw 0.966
   within 0.1pt.** This is independent confirmation that the protocol
   description in your repo is reproducible from scratch — we didn't
   need any additional hints. Thanks for that level of detail.

2. **The lift comes from fine-tuning, not from swapping the base
   encoder.** We tried `BAAI/bge-small-en-v1.5` raw on a 50q subset
   (R@5 = 0.98) — within noise of MiniLM raw (0.98). What moves the
   needle is the contrastive fine-tune on labelled queries: 100→200→
   300 train queries gives R@5 0.978→0.990→0.995.

### Caveat we want to flag

Our numbers are on **our own eval driver** (a thin reproduction of
your protocol description, in `benchmarks/longmemeval_eval.py`), not
on your `longmemeval_bench.py`. They line up at the protocol level
(see point 1 above), but a matched-protocol run via your own bench
script is the next thing on our roadmap. If you'd like to gate on
that before considering this further, totally fair — we'll come back
with that number.

### Possible integration shape

If interesting, a `mempal-adapt` integration could look like:

- mempal stays the storage / room / dialect / hybrid-retrieval layer.
- adaptmem adds the **encoder-side fine-tune step** as an optional
  "adapter": before ingestion, point adaptmem at the labelled-query
  set (if available), it produces a domain-tuned encoder that mempal
  then uses for embedding.
- No changes to the mempal API surface; the encoder swap happens at
  config load time.

We don't have strong feelings about the shape — happy to defer to
your design preferences. The point of this thread is just to put
the numbers in front of you and see whether there's a productive
conversation here.

### Reproduce

```bash
pip install adaptmem
git clone https://github.com/nakata-app/adaptmem
cd adaptmem
make bench-longmemeval   # FT-100 self-contained run
```

Three committed result JSONs in `benchmarks/`:
- `results_minilm_baseline_400.json` — raw protocol confirmation.
- `results_ft100_400.json` — self-contained FT-100 reproduce.
- `results_ft300_direct.json` — FT-300 reference run.

### Either outcome is fine

If this isn't a fit for mempal's direction, no problem — adaptmem
will keep on as a standalone tool. Just thought it was worth showing
the numbers and the integration sketch given how cleanly the
protocol confirmation came out.

Thanks again for the open work — the project structure made
independent reproduction straightforward.

— atakan

---

## Notes for Atakan before posting

1. **Replace the sign-off** with whatever name / handle you prefer
   for this attribution.
2. **Run the v0.5 matched-protocol** before posting if you want to
   address the "their bench script" caveat preemptively. The script
   is already cloned at `~/Projects/metis-pair/benchmarks/mempalace_repo/`.
3. **GitHub Discussion category:** check what categories
   `MemPalace/mempalace` has (Discussions tab → Categories). Likely
   targets: "Show and tell," "Ideas," or "General." Don't post under
   "Q&A" — it's not a question.
4. **Respect the repo's CONTRIBUTING.md / README** for any
   community-norm hints we're missing.
5. **Tone check:** the draft is intentionally extension-flavoured,
   not benchmark-bragging. Re-read once with that lens before posting.
6. After posting, drop the discussion URL into adaptmem's ROADMAP
   v0.5 section so future contributors can find it.
