Following up on §7 from the earlier post, CodeCrossEnc-v2 (hard-negative training) results are in.

### CodeCrossEnc-v2: hard negatives vs random negatives

**Training delta:** v1 used 30K positive pairs with 2 random negatives each (90K total). v2 re-uses the same 30K positives but replaces the random negatives with **hard negatives mined from FT-Code-5000's own top-50**, the candidates the bi-encoder ranked highly but incorrectly. These are the actual reranking-difficulty distribution. Same base model (`cross-encoder/ms-marco-MiniLM-L-6-v2`), same 1 epoch, batch=8, lr=2e-5.

**Eval:** FT-Code-5000 bi-encoder top-20 → rerank, CodeSearchNet python test split (22k full):

| Config | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| FT-Code-5000 (bi-alone) | 0.926 | 0.982 | 0.985 | 0.952 |
| + CodeCrossEnc-v1 (random-neg, n=5k) | 0.9148 | 0.9158 | 0.9194 | 0.9198 |
| **+ CodeCrossEnc-v2 (hard-neg, 22k)** | **__R1__** | **__R5__** | **__R10__** | **__MRR__** |
| Δ v2 vs bi-alone | __DR1__ | __DR5__ | __DR10__ | __DMRR__ |

__VERDICT__

[Fill after Colab 22K eval. Karar matrisi:
- R@1 ≥ 0.95 → pozitif, bi+cross pipeline 3-axis test'e taşı
- R@1 0.93-0.95 → marjinal pozitif, kabul
- R@1 < 0.93 → trivial overfit, hard-negative mining v2 (harder budget)]

**On the 3-axis test:** if v2 shows positive transfer, the natural next step is running `chunk_strategy_ablation` with bi-encoder swap *and* cross-encoder rerank in the same pass, chunking × bi-encoder × cross-encoder. The n=200 probe set from issue #82 is the right scale for that (n=20 CIs all include zero). Would you be open to running the 3-axis variant on your CUDA box once we share the v2 checkpoint?
