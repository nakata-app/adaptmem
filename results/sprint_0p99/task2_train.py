"""Task 2.3 — Fine-tune cross-encoder for LongMemEval chat-domain reranking.

Base: cross-encoder/ms-marco-MiniLM-L-12-v2
Loss: BinaryCrossEntropyLoss
Data: results/sprint_0p99/task2_{train,val}_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path("/Users/macmini/Projects/adaptmem")


def load_pairs(path: Path):
    from sentence_transformers import InputExample
    examples = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            examples.append(InputExample(texts=[r["q"], r["doc"]], label=float(r["label"])))
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--train", default=str(REPO / "results/sprint_0p99/task2_train_pairs.jsonl"))
    ap.add_argument("--val", default=str(REPO / "results/sprint_0p99/task2_val_pairs.jsonl"))
    ap.add_argument("--out", default=str(REPO / "checkpoints/chat-ce-v1-20260516"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader

    print(f"loading base: {args.base}")
    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_length, device=args.device)

    train_ex = load_pairs(Path(args.train))
    val_ex = load_pairs(Path(args.val))
    print(f"train={len(train_ex)}, val={len(val_ex)}")

    train_loader = DataLoader(train_ex, shuffle=True, batch_size=args.batch)

    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    val_pairs = [[ex.texts[0], ex.texts[1]] for ex in val_ex]
    val_labels = [int(ex.label) for ex in val_ex]
    evaluator = CEBinaryClassificationEvaluator(val_pairs, val_labels, name="val")

    warmup = int(0.1 * len(train_loader) * args.epochs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"epochs={args.epochs} batch={args.batch} lr={args.lr} warmup={warmup}")
    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        output_path=str(out),
        save_best_model=True,
        show_progress_bar=True,
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
