"""adaptmem CLI: train, search, bench."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_train(args):
    from adaptmem import AdaptMem
    from adaptmem.types import TrainConfig

    corpus = json.loads(Path(args.corpus).read_text())
    queries = json.loads(Path(args.queries).read_text())
    am = AdaptMem(base_model=args.base_model)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        top_k_mine=args.top_k_mine,
    )
    stats = am.train(corpus=corpus, labelled=queries, config=cfg)
    am.save(args.out)
    print(json.dumps(stats, indent=2))


def _cmd_search(args):
    from adaptmem import AdaptMem

    am = AdaptMem.load(args.model)
    hits = am.search(args.query, top_k=args.top_k)
    for h in hits:
        print(f"{h.score:.4f}\t{h.chunk_id}\t{h.text[:120]}")


def main():
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
    t.set_defaults(func=_cmd_train)

    s = sub.add_parser("search", help="run a query on a saved model")
    s.add_argument("--model", required=True, help="path saved by `adaptmem train`")
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=5)
    s.set_defaults(func=_cmd_search)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
