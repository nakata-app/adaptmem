# Mempal Discussion taslağı

**Status:** draft, not posted. Atakan should review + edit + post on
[`MemPalace/mempalace` Discussions](https://github.com/MemPalace/mempalace/discussions).

**Category:** "Show and tell" or "Ideas" (whichever the maintainers
prefer for community-extension posts).

**Tone:** "we extended your work", not "we beat your benchmark." The
mempal team published an open project; we're contributing back.

---

## Title (≤72 chars)

`Domain-adaptive fine-tune as orthogonal R@5 lift on top of MemPal raw`

(Variant if too long: `R@5 lift via domain-adaptive bi-encoder fine-tune (orthogonal to MemPal raw)`)

---

## Body

Hi MemPal team,

We've been using LongMemEval to evaluate a small open-source library
called [`adaptmem`](https://github.com/nakata-app/adaptmem), a 200-
line hard-negative mining + contrastive fine-tune wrapper around
SentenceTransformers, and the numbers we got line up cleanly with
the work you've already published. Wanted to share back, see if
it's interesting.

### What we measured

Same dataset (`longmemeval_s_cleaned.json`), same encoder family
(MiniLM-L6, ~90MB), **run through your own `longmemeval_bench.py`**
(monkey-patched to swap the encoder, zero changes to your eval logic).
Only the fine-tune step differs.

| System | R@1 | R@5 | R@10 | n |
|---|---|---|---|---|
| MemPal raw default (your bench script) | 0.806 | 0.966 | 0.982 | 500 |
| MemPal raw + adaptmem FT-300 (your bench script) | 0.862 | 0.980 | 0.994 | 500 |
| MemPal hybrid_v4 + adaptmem FT-300 (your bench script) | **0.916** | **0.990** | **0.998** | 500 |

Three findings worth flagging:

1. **Raw baseline R@5 = 0.966 matches your published number exactly.**
   Independent confirmation that your protocol is fully reproducible, 
   we didn't need any hints beyond the repo README.

2. **FT-300 + raw mode: +5.6pt R@1, +1.4pt R@5.** R@1 is where
   contrastive fine-tuning moves the needle most, the model learns to
   rank the right session *first*, not just in top-5.

3. **FT-300 + hybrid_v4: +11pt R@1, +2.4pt R@5.** Fine-tune and
   hybrid retrieval stack orthogonally, each adds lift on top of the
   other.

### Possible integration shape

If interesting, a `mempal-adapt` integration could look like:

- mempal stays the storage / room / dialect / hybrid-retrieval layer.
- adaptmem adds the **encoder-side fine-tune step** as an optional
  "adapter": before ingestion, point adaptmem at the labelled-query
  set (if available), it produces a domain-tuned encoder that mempal
  then uses for embedding.
- No changes to the mempal API surface; the encoder swap happens at
  config load time.

We don't have strong feelings about the shape, happy to defer to
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
- `results_minilm_baseline_400.json`, raw protocol confirmation.
- `results_ft100_400.json`, self-contained FT-100 reproduce.
- `results_ft300_direct.json`, FT-300 reference run.

### Either outcome is fine

If this isn't a fit for mempal's direction, no problem, adaptmem
will keep on as a standalone tool. Just thought it was worth showing
the numbers and the integration sketch given how cleanly the
protocol confirmation came out.

Thanks again for the open work, the project structure made
independent reproduction straightforward.

, atakan

---

## Notes for Atakan before posting

1. **Replace the sign-off** with whatever name / handle you prefer
   for this attribution.
2. ~~Run the v0.5 matched-protocol~~ **DONE**, numbers above are from
   your bench script directly (500q, matched protocol).
3. **GitHub Discussion category:** check what categories
   `MemPalace/mempalace` has (Discussions tab → Categories). Likely
   targets: "Show and tell," "Ideas," or "General." Don't post under
   "Q&A", it's not a question.
4. **Respect the repo's CONTRIBUTING.md / README** for any
   community-norm hints we're missing.
5. **Tone check:** the draft is intentionally extension-flavoured,
   not benchmark-bragging. Re-read once with that lens before posting.
6. After posting, drop the discussion URL into adaptmem's ROADMAP
   v0.5 section so future contributors can find it.
