# adaptmem

**Beat your retrieval baseline with 200 lines of hard-negative mining and a 90MB encoder.**

You point adaptmem at a domain (a corpus + a handful of labelled queries). It mines hard negatives, fine-tunes a tiny embedder on the contrastive objective, and hands you back a retriever that outperforms much larger generic models on your data.

This is the pipeline that pushed our internal LongMemEval R@5 from `0.966` (off-the-shelf MiniLM, matching MemPalace's "raw" headline) to **`0.9950`** on a generalisable held-out split — without any LLM in the loop, without hand-tuning, in a single epoch on CPU.

## Why this exists

The retrieval-quality literature has converged on a default: pick a 100M+ parameter generic embedder (bge-base, gte-base, mxbai), throw it at your data, hope it generalises. It usually doesn't — generic embedders compress concepts that **don't** matter in your domain and lose distinctions that **do**.

Domain adaptation works. The papers know it (DPR, ColBERT, SBERT). But the open-source workflow is fragmented:

- Hard-negative mining lives in one tutorial,
- Contrastive loss in another,
- Evaluation in a third,
- And every example assumes you already have a label set.

`adaptmem` is the missing one-shot wrapper. You write five lines, you get a domain-tuned encoder.

## What it does

```
your data (corpus + a few labelled queries)
        │
        ▼
[1] hard-negative mining       # vanilla MiniLM ranks haystack, mines top-K non-gold
        │
        ▼
[2] contrastive fine-tune      # MultipleNegativesRankingLoss, 1 epoch CPU
        │
        ▼
[3] (optional) cross-encoder   # ms-marco-MiniLM-L-12-v2 rerank
        │
        ▼
domain-tuned retriever         # serve via .search(query, top_k)
```

The recipe is small on purpose. Every choice is documented. Every step is one method call.

## Concrete result on LongMemEval (s_cleaned, 500 questions)

| System | R@1 | R@5 | LLM | Hand-tune | Generalisable |
|---|---|---|---|---|---|
| BM25 sparse baseline | — | 0.70 | ✗ | ✗ | ✓ |
| Stella dense (academic) | — | ~0.85 | ✗ | ✗ | ✓ |
| MemPalace raw (ChromaDB + MiniLM) | — | 0.966 | ✗ | ✗ | ✓ |
| MemPalace hybrid v4 generalisable | — | 0.984 | ✗ | ✗ | ✓ |
| MemPalace + Haiku rerank | — | 1.000 | ✓ | ✓ (3 q spot-fix) | ✗ |
| adaptmem (FT-200 dense, held-out 200q) | 0.900 | 0.990 | ✗ | ✗ | ✓ |
| **adaptmem (FT-300 dense, held-out 200q)** | **0.915** | **0.995** | **✗** | **✗** | **✓** |

Adaptmem numbers are reproduced from committed runs — see [`benchmarks/results_ft300_direct.json`](benchmarks/results_ft300_direct.json) and [`benchmarks/results_ft200_direct.json`](benchmarks/results_ft200_direct.json), each over the same 200 held-out questions on CPU. MemPalace numbers are quoted from their published results, not independently re-run here.

**Sanity:** training on more labelled queries (300 vs 200) lifts both R@1 (+1.5pt) and R@5 (+0.5pt) in the expected direction. Both runs clear the ROADMAP v0.2 sanity bar (R@5 ≥ 0.985).

Same encoder family (MiniLM-L6, 90MB) as MemPalace, same dataset, same evaluation protocol (per-question fresh corpus, user-only encoding) — only the **fine-tune step** is different.

### Reproduce

```bash
# Evaluate the existing FT-300 SentenceTransformer model directly
python benchmarks/longmemeval_eval.py --mode test \
    --st-model /path/to/minilm-lme-ft-300 \
    --results-out benchmarks/results_ft300_direct.json
```

A cross-encoder rerank stage (R@1 lift) is on the v0.4 roadmap — a JSON capture is not yet committed.

## Usage (planned API)

```python
from adaptmem import AdaptMem

# Your domain
corpus = ["passage 1 text...", "passage 2 text...", ...]
labelled = [
    {"query": "...", "relevant_ids": ["p3", "p7"]},
    ...
]

am = AdaptMem(base_model="all-MiniLM-L6-v2")
am.train(corpus=corpus, labelled=labelled, epochs=1)
am.save("./my-domain-encoder")

# Use
hits = am.search("user query", top_k=5)
for chunk_id, score in hits:
    print(chunk_id, score)
```

CLI parity:

```bash
adaptmem train --corpus corpus.json --queries queries.json --out my-encoder/
adaptmem search --model my-encoder/ --query "..." --top-k 5
adaptmem bench longmemeval --data longmemeval_s_cleaned.json
```

## What it is NOT

- Not a generic embedder. The output model is **specialised** to the corpus you trained on.
- Not a replacement for retrieval engineering. You still need to think about chunking, encoding format, and ground-truth labels.
- Not a one-click win when your queries are out-of-distribution. Domain adaptation rewards in-distribution test data.

## Status

`v0.1` skeleton in flight. Real code, real tests, real benchmark numbers are landing.

## License

MIT.
