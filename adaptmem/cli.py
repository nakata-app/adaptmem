"""adaptmem CLI: train, search, bench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> None:
    from adaptmem import AdaptMem
    from adaptmem.types import TrainConfig

    corpus = json.loads(Path(args.corpus).read_text())
    queries = json.loads(Path(args.queries).read_text())
    am = AdaptMem(
        base_model=args.base_model,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
    )
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        top_k_mine=args.top_k_mine,
    )
    stats = am.train(corpus=corpus, labelled=queries, config=cfg)
    am.save(args.out)
    print(json.dumps(stats, indent=2))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Compute Recall@k for a saved model against a labelled queries file.

    queries.json shape: list of {"query": str, "relevant_ids": [str, ...]}
    """
    from adaptmem import AdaptMem

    queries = json.loads(Path(args.queries).read_text())
    am = AdaptMem.load(args.model)
    if args.rerank:
        am.rerank_enabled = True
        if args.rerank_model:
            am.rerank_model_name = args.rerank_model
            am._rerank_model = None

    ks = sorted({1, 5, args.top_k})
    max_k = max(ks)
    hits_at = {k: 0 for k in ks}
    n = 0
    for q in queries:
        relevant = set(q["relevant_ids"])
        if not relevant:
            continue
        ranked = [h.chunk_id for h in am.search(q["query"], top_k=max_k)]
        for k in ks:
            if any(rid in ranked[:k] for rid in relevant):
                hits_at[k] += 1
        n += 1

    if n == 0:
        print(json.dumps({"error": "no labelled queries with relevant_ids"}, indent=2))
        return
    out = {"n": n, "recall": {f"@{k}": round(hits_at[k] / n, 4) for k in ks}}
    print(json.dumps(out, indent=2))


def _cmd_search(args: argparse.Namespace) -> None:
    from adaptmem import AdaptMem

    am = AdaptMem.load(args.model)
    # Allow CLI override of stored rerank state (e.g. "trained without rerank,
    # serve with rerank for an A/B").
    if args.rerank:
        am.rerank_enabled = True
        if args.rerank_model:
            am.rerank_model_name = args.rerank_model
            am._rerank_model = None  # force re-resolve on next search
    hits = am.search(args.query, top_k=args.top_k, rerank_top_k=args.rerank_top_k)
    for h in hits:
        print(f"{h.score:.4f}\t{h.chunk_id}\t{h.text[:120]}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="adaptmem")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="mine + fine-tune + save")
    t.add_argument("--corpus", required=True, help="JSON list of {id, text} or strings")
    t.add_argument("--queries", required=True, help="JSON list of {query, relevant_ids}")
    t.add_argument("--out", required=True, help="output directory")
    t.add_argument("--base-model", default="all-MiniLM-L6-v2")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--top-k-mine", type=int, default=10)
    t.add_argument(
        "--rerank",
        action="store_true",
        help="Persist rerank=True so .load() restores cross-encoder rerank.",
    )
    t.add_argument(
        "--rerank-model",
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        help="Cross-encoder model name (only used when --rerank is set).",
    )
    t.set_defaults(func=_cmd_train)

    s = sub.add_parser("search", help="run a query on a saved model")
    s.add_argument("--model", required=True, help="path saved by `adaptmem train`")
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument(
        "--rerank",
        action="store_true",
        help="Force-enable cross-encoder rerank for this search (overrides "
             "the saved model's rerank flag).",
    )
    s.add_argument(
        "--rerank-model",
        default=None,
        help="Override the saved cross-encoder model name (requires --rerank).",
    )
    s.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help="Bi-encoder candidate set size before CE rerank (default: top-k * 3).",
    )
    s.set_defaults(func=_cmd_search)

    e = sub.add_parser("evaluate", help="recall@k against a labelled queries file")
    e.add_argument("--model", required=True, help="path saved by `adaptmem train`")
    e.add_argument(
        "--queries",
        required=True,
        help='JSON list of {"query": str, "relevant_ids": [str,...]}',
    )
    e.add_argument("--top-k", type=int, default=10, help="Largest k to compute (R@1/R@5/R@k)")
    e.add_argument("--rerank", action="store_true")
    e.add_argument("--rerank-model", default=None)
    e.set_defaults(func=_cmd_evaluate)

    sv = sub.add_parser(
        "serve",
        help="run the long-lived HTTP daemon (Metis / claimcheck consumers)",
    )
    sv.add_argument("--base-model", default="all-MiniLM-L6-v2")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=7800)
    sv.add_argument(
        "--device",
        default=None,
        help="PyTorch device override: 'cpu', 'cuda', 'mps'",
    )
    sv.add_argument(
        "--uds",
        default=None,
        help="Optional Unix-domain-socket path (overrides --host/--port).",
    )
    sv.set_defaults(func=_cmd_serve)

    args = ap.parse_args()
    args.func(args)


def _cmd_serve(args: argparse.Namespace) -> None:
    from adaptmem.server import serve

    serve(
        base_model=args.base_model,
        host=args.host,
        port=args.port,
        device=args.device,
        uds=args.uds,
    )


if __name__ == "__main__":
    main()
