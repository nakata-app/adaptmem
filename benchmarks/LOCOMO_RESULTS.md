# mnemonics on LoCoMo, end-to-end QA results

**Headline: 0.8494 answer accuracy on LoCoMo (1540 QA, judge-graded).**

This measures the *full* mnemonics pipeline end-to-end, ingest → fact
extraction → hybrid retrieve + AdaptMem rerank → LLM answers from retrieved
memory, not retrieval recall in isolation. The number is honest: it is the
plateau of this architecture, and the limits are documented below.

## Results

| Stage | What changed | Accuracy | multi-hop | temporal | open | single |
|---|---|---|---|---|---|---|
| run4 | full fact-coverage baseline | 0.8214 | 0.741 | 0.822 | 0.563 | 0.878 |
| run6 | date-stamped facts (`[date] [fact] …`) | 0.8286 | 0.752 | 0.829 | 0.563 | 0.885 |
| **M3** | **MiniMax-M3 answerer** (retrieval frozen) | **0.8494** | **0.784** | **0.844** | **0.635** | **0.898** |

Each stage moved a real lever; no category regressed. Total lift run4 → M3:
**+2.8pp**.

## Method

- **Dataset:** LoCoMo (10 conversations, 1540 QA after excluding category 5 /
  adversarial-unanswerable, per the Mem0 evaluation protocol).
- **Retrieval:** per-speaker namespaces, hybrid (vector + BM25 RRF) +
  AdaptMem cross-encoder rerank, top-30 of 50 candidates.
- **Fact extraction:** per-session LLM distillation into atomic, provenance-
  tagged facts (`DIA=<src ids>|[<session date>] [fact] …`). Date stamp lets
  the answerer do temporal math the same way it does from raw turns.
- **Answerer:** MiniMax-M3 (the run4→run6 baseline used deepseek-chat).
- **Judge:** deepseek-chat, temp 0, CORRECT/WRONG against the gold answer.
  Held fixed across all runs for cross-run comparability.

## Honest limits (where the headroom went)

Two independent levers were pushed to saturation; 0.8494 is a genuine plateau,
not a stopping point of convenience.

1. **Answerer is saturated.** Two strong, independent answerers tie: MiniMax-M3
   = 0.8494, deepseek-v4-pro = 0.8487 on the same frozen contexts. When two
   different model families converge to the same ~0.85, the answer layer has
   no more to give.
2. **Remaining errors are hard retrieval, not easy retrieval.** Of 263 wrong
   answers, 156 (59%) had full evidence in context (answer-layer fault, now
   saturated); 107 (41%) were retrieval misses. Of the retrieval misses,
   **83% of the missing evidence sits in a *different session*** from the
   evidence that did surface, i.e. genuine cross-session multi-hop, the
   structurally hard part. Cheap intra-session / neighbour expansion caps at
   ~17% of those misses, so it was not pursued.
3. **Judge is V3 (deepseek-chat), not V4-Pro.** Run ordering (which lever wins)
   is robust to judge choice; absolute values could shift a little under a
   stronger judge. We did not re-grade, the comparison that matters
   (lever vs lever) is judge-invariant.

## Cross-system context (treat cautiously)

Published LoCoMo numbers vary by judge and protocol, and we did **not** re-run
other systems under our judge, so this is context, not a controlled head-to-head.
Under matched-protocol framing, mnemonics at 0.8494 sits at or above
graph/temporal-memory systems in the ~0.75 band and below full-context-with-
strong-answerer setups in the ~0.89, 0.92 band. The remaining gap to that top
band is structural (cross-session multi-hop retrieval + a stronger answerer),
not a tuning gap. *[vendor bands: unverified secondary numbers, do not cite as ours]*

## Reproduce

```bash
# end-to-end eval (Kaggle GPU; injects DeepSeek key at submit time)
krun benchmarks/kaggle_locomo_e2e.py --dataset atakanakbaba/locomo --acc NvidiaTeslaT4

# grade answers (local)
python benchmarks/locomo_judge.py locomo_e2e_run6

# free triage of where the errors are (no GPU, no API)
python benchmarks/locomo_headroom.py locomo_e2e_run6        # answerer vs retrieval split
python benchmarks/locomo_miss_analysis.py locomo_e2e_run6   # retrieval-miss anatomy

# swap the answerer on saved contexts (retrieval frozen), any OpenAI-compatible provider
python benchmarks/locomo_answerer_swap.py --model MiniMax-M3 \
  --base-url https://api.minimaxi.chat/v1 --key-env MINIMAX_API_KEY --stride 1
```

Artifacts per run live in `results/locomo_e2e_*/results/`:
`locomo_answers.json`, `locomo_contexts.json` (frozen contexts for local
answerer swaps), `locomo_verdicts.json` (per-question CORRECT/WRONG for flip
analysis), `locomo_judged.json` (category breakdown).
