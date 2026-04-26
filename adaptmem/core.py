"""High-level AdaptMem class: train + persist + search."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from adaptmem.miner import CorpusEntry, HardNegativeMiner
from adaptmem.types import LabelledQuery, RetrievalHit, TrainConfig


class AdaptMem:
    """One-shot domain adaptation for retrieval.

    Default flow:
        am = AdaptMem(base_model="all-MiniLM-L6-v2")
        am.train(corpus=[...], labelled=[LabelledQuery(...), ...])
        hits = am.search("question text", top_k=5)

    The base model is loaded lazily (first `train` or `load` call). The
    fine-tuned model lives in memory until you call `save(path)`.
    """

    def __init__(self, base_model: str = "all-MiniLM-L6-v2"):
        self.base_model_name = base_model
        self._model = None
        self._corpus: list[CorpusEntry] = []
        self._embeddings: np.ndarray | None = None

    # ---- Training -------------------------------------------------------
    def train(
        self,
        corpus: list[str] | list[CorpusEntry] | list[dict],
        labelled: list[LabelledQuery] | list[dict],
        config: TrainConfig | None = None,
    ) -> dict:
        """Mine hard negatives, fine-tune via MultipleNegativesRankingLoss, build index.

        `corpus` can be:
          - list[str] — auto-assigned ids "c0", "c1", ...
          - list[CorpusEntry]
          - list[dict] with keys {"id", "text"}

        Returns a dict with training stats (n_pairs, train_loss, runtime_s).
        """
        config = config or TrainConfig()
        entries = _normalise_corpus(corpus)
        queries = _normalise_queries(labelled)
        self._corpus = entries

        from sentence_transformers import SentenceTransformer

        base = SentenceTransformer(self.base_model_name)
        miner = HardNegativeMiner(base_model=base, top_k_mine=config.top_k_mine)
        pairs = miner.mine(entries, queries)
        if not pairs:
            raise ValueError("Hard-negative mining produced 0 pairs — check your labels.")

        # Fine-tune
        from sentence_transformers import losses
        from torch.utils.data import DataLoader

        examples = [p.to_input_example() for p in pairs]
        loader = DataLoader(examples, shuffle=True, batch_size=config.batch_size)
        loss = losses.MultipleNegativesRankingLoss(base)

        import time

        t0 = time.time()
        n_steps = max(1, (len(examples) // config.batch_size) * config.epochs)
        warmup = int(n_steps * config.warmup_ratio)
        base.fit(
            train_objectives=[(loader, loss)],
            epochs=config.epochs,
            warmup_steps=warmup,
            optimizer_params={"lr": config.learning_rate},
            show_progress_bar=False,
        )
        runtime = time.time() - t0

        self._model = base
        # Build index over the corpus with the freshly tuned model
        self._embeddings = base.encode(
            [c.text for c in entries],
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return {"n_pairs": len(pairs), "runtime_s": round(runtime, 2), "n_steps": n_steps}

    # ---- Persistence ---------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save. Call .train() or .load() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save(str(out / "model"))
        # Persist corpus + embeddings for inference reload
        np.save(out / "embeddings.npy", self._embeddings)
        with open(out / "corpus.tsv", "w") as f:
            for c in self._corpus:
                # tab-safe: escape tabs/newlines in text
                t = c.text.replace("\t", " ").replace("\n", " ")
                f.write(f"{c.id}\t{t}\n")

    @classmethod
    def load(cls, path: str | Path) -> "AdaptMem":
        from sentence_transformers import SentenceTransformer

        p = Path(path)
        am = cls.__new__(cls)
        am.base_model_name = ""
        am._model = SentenceTransformer(str(p / "model"))
        am._embeddings = np.load(p / "embeddings.npy")
        am._corpus = []
        with open(p / "corpus.tsv") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                cid, text = line.split("\t", 1)
                am._corpus.append(CorpusEntry(id=cid, text=text))
        return am

    # ---- Inference -----------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if self._model is None or self._embeddings is None:
            raise RuntimeError("Not initialised. Call .train() or .load() first.")
        qv = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        scores = self._embeddings @ qv
        k = min(top_k, len(self._corpus))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            RetrievalHit(chunk_id=self._corpus[i].id, text=self._corpus[i].text, score=float(scores[i]))
            for i in idx
        ]


# ---- helpers ---------------------------------------------------------
def _normalise_corpus(corpus) -> list[CorpusEntry]:
    out: list[CorpusEntry] = []
    for i, c in enumerate(corpus):
        if isinstance(c, str):
            out.append(CorpusEntry(id=f"c{i}", text=c))
        elif isinstance(c, CorpusEntry):
            out.append(c)
        elif isinstance(c, dict):
            out.append(CorpusEntry(id=str(c.get("id", f"c{i}")), text=c["text"]))
        else:
            raise TypeError(f"corpus item {i} has unsupported type {type(c).__name__}")
    return out


def _normalise_queries(labelled) -> list[LabelledQuery]:
    out: list[LabelledQuery] = []
    for q in labelled:
        if isinstance(q, LabelledQuery):
            out.append(q)
        elif isinstance(q, dict):
            out.append(LabelledQuery(query=q["query"], relevant_ids=list(q["relevant_ids"])))
        else:
            raise TypeError(f"labelled item has unsupported type {type(q).__name__}")
    return out
