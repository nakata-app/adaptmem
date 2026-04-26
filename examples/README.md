# adaptmem examples

Three runnable scripts covering the typical paths through the API. All run on
CPU; the longest is well under a minute on an M-series Mac.

| Script | What it shows | Runtime (CPU) |
|---|---|---|
| [`01_basic_usage.py`](01_basic_usage.py) | Train on a 6-document corpus + 4 labelled queries, save, reload, search. The "does my install work?" smoke test. | ~30 s |
| [`02_with_rerank.py`](02_with_rerank.py) | Same shape but with `rerank=True`. Lazy-loads the cross-encoder on the first `.search()`. Returned scores are CE logits, not cosines. | ~30 s + first-time CE download |
| [`03_streaming_corpus.py`](03_streaming_corpus.py) | Use `add_corpus()` to extend the index without retraining. Demonstrates id-based de-duplication. | ~15 s |

## Run

```bash
# from the repo root
pip install -e ".[dev]"
python examples/01_basic_usage.py
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB) into the HuggingFace
cache. Subsequent runs are offline.

## Apple silicon note

All examples pass `device="cpu"` to `AdaptMem(...)`. The default would be
PyTorch's autodetect (which prefers MPS on Apple silicon), but contrastive
fine-tunes have hit MPS deadlocks under memory pressure on this hardware.
Drop `device="cpu"` if you're on Linux/CUDA or have proven MPS stable on
your system.
