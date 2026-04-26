"""LongMemEval reproduction benchmark for adaptmem.

Reproduces the README's R@5=0.9950 claim on the held-out 200-question split.

Protocol (matches MemPalace + metis-pair eval_retrieval.py):
  - Per-question fresh corpus: each question's haystack_sessions is its own pool
  - User-only encoding: session = "\\n".join(t.content for t in turns if role=="user")
  - Empty sessions (no user turns) are dropped
  - relevant_ids = answer_session_ids
  - Recall@k: 1.0 if any GT id is in top-k, else 0.0; reported as macro-mean

Modes:
  --mode train   Mine + fine-tune on N train questions, save AdaptMem package.
                 Train hyperparams match the FT-300 recipe: epochs=3, batch=16, lr=2e-5.
  --mode test    Evaluate a saved adaptmem model on a held-out set.
                 If --st-model is given (a raw SentenceTransformer dir), evaluate
                 that directly using the same protocol — used to verify parity
                 with the standalone FT-300 model.

Splits:
  By default, uses the seed=42 shuffle from metis-pair/training_300:
    300 train / 200 test (matches the existing FT-300 model).
  Override with --split-ids PATH to a JSON {train_question_ids: [...], test_question_ids: [...]}.

CLI:
  longmemeval_eval.py --mode train --dataset longmemeval_s_cleaned.json \\
                      --split-ids split_ids.json --n-train 300 \\
                      --model-out ./bench-model

  longmemeval_eval.py --mode test  --dataset longmemeval_s_cleaned.json \\
                      --split-ids split_ids.json --model-in ./bench-model \\
                      [--results-out results.json]

  longmemeval_eval.py --mode test  --dataset longmemeval_s_cleaned.json \\
                      --split-ids split_ids.json --st-model ./minilm-lme-ft-300

No LLM. No API. CPU-friendly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def session_to_text(session_turns) -> str:
    """User-only encoding: concatenate user contents with newlines."""
    return "\n".join(
        t.get("content", "") for t in session_turns if t.get("role") == "user"
    )


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_question_index(data: list[dict]) -> dict[str, dict]:
    return {q["question_id"]: q for q in data}


def select_questions(
    data: list[dict], split_ids: dict | None, kind: str, n: int | None
) -> list[dict]:
    """kind: 'train' or 'test'. Returns the question dicts in their split-ID order."""
    if split_ids:
        key = "train_question_ids" if kind == "train" else "test_question_ids"
        ids = split_ids[key]
        if n is not None and kind == "train":
            ids = ids[:n]
        idx = build_question_index(data)
        return [idx[qid] for qid in ids if qid in idx]
    # Fallback: first N as train, rest as test
    if kind == "train":
        return data[: n or 300]
    return data[(n or 300):]


def make_per_question_corpus(q: dict) -> tuple[list[str], list[str]]:
    """Return (active_sids, active_docs) — non-empty sessions only, aligned."""
    sids = list(q["haystack_session_ids"])
    sessions = list(q["haystack_sessions"])
    docs = [session_to_text(s) for s in sessions]
    active = [(sid, d) for sid, d in zip(sids, docs) if d]
    if not active:
        return [], []
    active_sids, active_docs = zip(*active)
    return list(active_sids), list(active_docs)


# ---- Train mode ---------------------------------------------------------
def build_labelled_queries(train_questions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Build (corpus_entries, labelled_queries) suitable for AdaptMem.train.

    The corpus is the union of all train haystack sessions across questions,
    keyed by session_id. Sessions with no user turns are dropped. Duplicate
    session_ids across questions are de-duped on first sight (LongMemEval
    haystacks are per-question disjoint in practice, but we guard anyway).
    """
    corpus_seen: dict[str, str] = {}
    labelled: list[dict] = []
    for q in train_questions:
        sids, docs = make_per_question_corpus(q)
        if not sids:
            continue
        for sid, doc in zip(sids, docs):
            if sid not in corpus_seen:
                corpus_seen[sid] = doc
        # Only keep relevant_ids that are actually present in the (filtered) corpus
        relevant = [sid for sid in q["answer_session_ids"] if sid in corpus_seen]
        if not relevant:
            continue
        labelled.append({"query": q["question"], "relevant_ids": relevant})
    corpus = [{"id": sid, "text": text} for sid, text in corpus_seen.items()]
    return corpus, labelled


def cmd_train(args) -> None:
    from adaptmem import AdaptMem
    from adaptmem.types import TrainConfig

    print(f"loading dataset {args.dataset}…", file=sys.stderr, flush=True)
    data = load_dataset(args.dataset)
    print(f"  {len(data)} questions total", file=sys.stderr, flush=True)

    split_ids = json.load(open(args.split_ids)) if args.split_ids else None
    train_qs = select_questions(data, split_ids, "train", args.n_train)
    print(f"  train questions: {len(train_qs)}", file=sys.stderr, flush=True)

    corpus, labelled = build_labelled_queries(train_qs)
    print(
        f"  corpus entries: {len(corpus)}, labelled queries: {len(labelled)}",
        file=sys.stderr,
        flush=True,
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        top_k_mine=args.top_k_mine,
    )
    am = AdaptMem(base_model=args.base_model)

    t0 = time.time()
    stats = am.train(corpus=corpus, labelled=labelled, config=cfg)
    total = time.time() - t0

    out = Path(args.model_out)
    am.save(out)
    stats["wall_clock_s"] = round(total, 2)
    stats["n_train_questions"] = len(train_qs)
    stats["n_corpus"] = len(corpus)
    stats["epochs"] = cfg.epochs
    stats["batch_size"] = cfg.batch_size
    stats["learning_rate"] = cfg.learning_rate
    stats["base_model"] = args.base_model
    print(json.dumps(stats, indent=2))
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2))


# ---- Test mode ---------------------------------------------------------
def recall_at_k(retrieved_ids: list[str], gt: set[str], k: int) -> float:
    return 1.0 if any(g in retrieved_ids[:k] for g in gt) else 0.0


def evaluate_with_st_model(model, test_qs: list[dict], top_k: int = 10) -> dict:
    """Per-question evaluation using a raw SentenceTransformer model."""
    import numpy as np

    n = len(test_qs)
    r1 = r5 = r10 = 0.0
    skipped = 0
    t0 = time.time()
    for i, q in enumerate(test_qs):
        sids, docs = make_per_question_corpus(q)
        if not sids:
            skipped += 1
            continue
        embs = model.encode(
            docs,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        qv = model.encode(
            [q["question"]],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        scores = embs @ qv
        order = np.argsort(-scores)
        ranked_sids = [sids[j] for j in order]
        gt = set(q["answer_session_ids"])
        r1 += recall_at_k(ranked_sids, gt, 1)
        r5 += recall_at_k(ranked_sids, gt, 5)
        r10 += recall_at_k(ranked_sids, gt, 10)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n - i - 1) / (i + 1)
            print(f"  q{i+1}/{n}  eta {eta:.0f}s", file=sys.stderr, flush=True)
    n_eff = n - skipped
    return {
        "n_test": n,
        "n_evaluated": n_eff,
        "n_skipped_empty": skipped,
        "r1": round(r1 / n, 4) if n else 0.0,
        "r5": round(r5 / n, 4) if n else 0.0,
        "r10": round(r10 / n, 4) if n else 0.0,
        "wall_clock_s": round(time.time() - t0, 2),
    }


def cmd_test(args) -> None:
    print(f"loading dataset {args.dataset}…", file=sys.stderr, flush=True)
    data = load_dataset(args.dataset)
    split_ids = json.load(open(args.split_ids)) if args.split_ids else None
    test_qs = select_questions(data, split_ids, "test", args.n_train)
    print(f"  test questions: {len(test_qs)}", file=sys.stderr, flush=True)

    if args.st_model:
        from sentence_transformers import SentenceTransformer

        print(
            f"  loading raw SentenceTransformer {args.st_model}…",
            file=sys.stderr,
            flush=True,
        )
        model = SentenceTransformer(args.st_model)
        results = evaluate_with_st_model(model, test_qs, top_k=10)
        results["mode"] = "st-model"
        results["model_path"] = args.st_model
    else:
        if not args.model_in:
            print("--model-in required when --st-model is not used", file=sys.stderr)
            sys.exit(2)
        from adaptmem import AdaptMem
        from sentence_transformers import SentenceTransformer

        print(f"  loading adaptmem model {args.model_in}…", file=sys.stderr, flush=True)
        am = AdaptMem.load(args.model_in)
        # We bypass AdaptMem.search per-question because its index is the global
        # train corpus, while LongMemEval evaluation needs a per-question fresh
        # corpus. We reuse only the underlying SentenceTransformer.
        model: SentenceTransformer = am._model  # noqa: SLF001
        results = evaluate_with_st_model(model, test_qs, top_k=10)
        results["mode"] = "adaptmem"
        results["model_path"] = args.model_in

    results["dataset"] = args.dataset
    results["split_ids"] = args.split_ids
    print()
    print(f"# Mode: {results['mode']}")
    print(f"# Model: {results['model_path']}")
    print(f"# n_test={results['n_test']} (skipped_empty={results['n_skipped_empty']})")
    print(f"# R@1={results['r1']}  R@5={results['r5']}  R@10={results['r10']}")

    if args.results_out:
        Path(args.results_out).write_text(json.dumps(results, indent=2))
        print(f"# wrote {args.results_out}", file=sys.stderr)


# ---- Main ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=["train", "test"])
    ap.add_argument(
        "--dataset",
        default="/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json",
        help="Path to longmemeval_s_cleaned.json",
    )
    ap.add_argument(
        "--split-ids",
        default="/Users/macmini/Projects/metis-pair/benchmarks/data/training_300/split_ids.json",
        help="Path to split_ids.json with train/test question_id lists",
    )
    ap.add_argument("--n-train", type=int, default=300, help="Number of train questions")
    # train-only
    ap.add_argument("--model-out", default="./bench-model", help="Output dir (train)")
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--top-k-mine", type=int, default=10)
    # test-only
    ap.add_argument("--model-in", help="Adaptmem model dir (test)")
    ap.add_argument(
        "--st-model",
        help="Raw SentenceTransformer dir (test) — used to evaluate the existing FT-300 model directly",
    )
    ap.add_argument("--results-out", help="Write results JSON here")
    args = ap.parse_args()

    if args.mode == "train":
        cmd_train(args)
    else:
        cmd_test(args)


if __name__ == "__main__":
    main()
