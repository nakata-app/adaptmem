"""High-level AdaptMem class: train + persist + search."""
from __future__ import annotations

from pathlib import Path
from typing import Any

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

    def __init__(
        self,
        base_model: str = "all-MiniLM-L6-v2",
        rerank: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str | None = None,
    ):
        self.base_model_name = base_model
        self._model = None
        self._corpus: list[CorpusEntry] = []
        self._embeddings: np.ndarray[Any, Any] | None = None
        # Optional cross-encoder rerank stage. Disabled by default; when enabled,
        # `search` fetches a wider bi-encoder candidate set and reorders it with
        # the cross-encoder before returning top_k.
        self.rerank_enabled = rerank
        self.rerank_model_name = rerank_model
        self._rerank_model = None
        # Device override for the underlying SentenceTransformer. None lets
        # PyTorch auto-detect (CUDA → MPS → CPU). Passing "cpu" sidesteps
        # MPS deadlocks that have surfaced on some Apple-silicon configs
        # during contrastive fine-tuning. Persisted in config.json so that
        # .load() can restore the same choice.
        self.device = device

    # ---- Training -------------------------------------------------------
    def train(
        self,
        corpus: list[str] | list[CorpusEntry] | list[dict[str, Any]],
        labelled: list[LabelledQuery] | list[dict[str, Any]],
        config: TrainConfig | None = None,
    ) -> dict[str, Any]:
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

        st_kwargs = {"device": self.device} if self.device else {}
        base = SentenceTransformer(self.base_model_name, **st_kwargs)
        miner = HardNegativeMiner(base_model=base, top_k_mine=config.top_k_mine)
        pairs = miner.mine(entries, queries)
        if not pairs:
            raise ValueError("Hard-negative mining produced 0 pairs — check your labels.")

        # Fine-tune
        from sentence_transformers import losses  # type: ignore[attr-defined]
        from torch.utils.data import DataLoader

        examples = [p.to_input_example() for p in pairs]
        loader: DataLoader[Any] = DataLoader(examples, shuffle=True, batch_size=config.batch_size)  # type: ignore[arg-type]
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
        # Approximate token count (whitespace heuristic — tokenizer-free so it
        # works for any base model). Word count is the common proxy in
        # retrieval logs; callers want an order-of-magnitude budget, not an
        # exact tokenizer count.
        all_texts: list[str] = [c.text for c in entries]
        for p in pairs:
            all_texts.append(p.anchor)
            all_texts.append(p.positive)
            all_texts.append(p.negative)
        n_tokens_approx = sum(len(t.split()) for t in all_texts)
        return {
            "n_pairs": len(pairs),
            "runtime_s": round(runtime, 2),
            "n_steps": n_steps,
            "n_tokens_approx": n_tokens_approx,
            "tokens_per_s": round(n_tokens_approx / runtime, 1) if runtime > 0 else 0.0,
        }

    # ---- Public introspection (stable contract for downstream tools) ---
    @property
    def encoder(self) -> Any:
        """The underlying SentenceTransformer (after .train() or .load()).

        Exposed so downstream packages (e.g. halluguard) can plug the tuned
        encoder into their own retrievers without reaching into `_model`.
        """
        return self._model

    @property
    def corpus(self) -> list[CorpusEntry]:
        """The indexed corpus entries, in insertion order."""
        return list(self._corpus)

    @property
    def embeddings(self) -> np.ndarray[Any, Any] | None:
        """The L2-normalised embedding matrix aligned with `corpus`."""
        return self._embeddings

    # ---- Streaming corpus updates -------------------------------------
    def add_corpus(self, new_corpus: list[str] | list[CorpusEntry] | list[dict[str, Any]]) -> int:
        """Append entries to the in-memory index without re-encoding the
        whole corpus. Returns the number of newly added entries.

        Skips entries whose `id` already exists in the current index — safe
        to call repeatedly with overlapping batches. Existing embeddings
        are preserved; only the new chunk texts are encoded.

        Requires that `.train()` or `.load()` was called first (the index
        encoder + embedding matrix must already exist).
        """
        if self._model is None or self._embeddings is None:
            raise RuntimeError("Not initialised. Call .train() or .load() first.")
        new_entries = _normalise_corpus(new_corpus)
        existing_ids = {c.id for c in self._corpus}
        fresh = [e for e in new_entries if e.id not in existing_ids]
        if not fresh:
            return 0
        new_vecs = self._model.encode(
            [e.text for e in fresh],
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self._corpus.extend(fresh)
        self._embeddings = np.vstack([self._embeddings, new_vecs])
        return len(fresh)

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
        # Persist config so `.load(path)` restores rerank settings.
        import json as _json
        (out / "config.json").write_text(
            _json.dumps(
                {
                    "base_model": self.base_model_name,
                    "rerank": self.rerank_enabled,
                    "rerank_model": self.rerank_model_name,
                    "device": self.device,
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "AdaptMem":
        from sentence_transformers import SentenceTransformer

        p = Path(path)
        am = cls.__new__(cls)
        am.base_model_name = ""
        am.device = None
        # Read device from config.json (if present) before constructing the
        # SentenceTransformer so we can honour an explicit "cpu" choice.
        cfg_path = p / "config.json"
        cfg = None
        if cfg_path.exists():
            import json as _json
            cfg = _json.loads(cfg_path.read_text())
            am.device = cfg.get("device")
        st_kwargs = {"device": am.device} if am.device else {}
        am._model = SentenceTransformer(str(p / "model"), **st_kwargs)
        am._embeddings = np.load(p / "embeddings.npy")
        am._corpus = []
        with open(p / "corpus.tsv") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                cid, text = line.split("\t", 1)
                am._corpus.append(CorpusEntry(id=cid, text=text))
        # Restore rerank config (cfg loaded above for device — reuse).
        if cfg is not None:
            am.base_model_name = cfg.get("base_model", "")
            am.rerank_enabled = bool(cfg.get("rerank", False))
            am.rerank_model_name = cfg.get(
                "rerank_model", "cross-encoder/ms-marco-MiniLM-L-12-v2"
            )
        else:
            am.rerank_enabled = False
            am.rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        am._rerank_model = None
        return am

    # ---- Inference -----------------------------------------------------
    def _ensure_rerank_model(self) -> None:
        if self._rerank_model is None:
            from sentence_transformers import CrossEncoder
            self._rerank_model = CrossEncoder(self.rerank_model_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: int | None = None,
    ) -> list[RetrievalHit]:
        """Retrieve top_k passages.

        When `rerank_enabled` is True, fetch `rerank_top_k or top_k * 3`
        candidates from the bi-encoder index, score the (query, candidate)
        pairs with the cross-encoder, and return the top_k by CE score.
        Cross-encoder is lazy-loaded.
        """
        if self._model is None or self._embeddings is None:
            raise RuntimeError("Not initialised. Call .train() or .load() first.")
        qv = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        scores = self._embeddings @ qv
        candidate_k = (rerank_top_k or max(top_k * 3, top_k)) if self.rerank_enabled else top_k
        candidate_k = min(candidate_k, len(self._corpus))
        idx = np.argpartition(-scores, candidate_k - 1)[:candidate_k]
        idx = idx[np.argsort(-scores[idx])]

        if not self.rerank_enabled:
            return [
                RetrievalHit(
                    chunk_id=self._corpus[i].id,
                    text=self._corpus[i].text,
                    score=float(scores[i]),
                )
                for i in idx
            ]

        # Cross-encoder rerank
        self._ensure_rerank_model()
        candidates = [self._corpus[i] for i in idx]
        pairs = [(query, c.text) for c in candidates]
        ce_scores = self._rerank_model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, ce_scores), key=lambda x: -float(x[1]))
        k = min(top_k, len(ranked))
        return [
            RetrievalHit(chunk_id=c.id, text=c.text, score=float(s))
            for c, s in ranked[:k]
        ]


# ---- helpers ---------------------------------------------------------
def _normalise_corpus(
    corpus: list[str] | list[CorpusEntry] | list[dict[str, Any]],
) -> list[CorpusEntry]:
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


def _normalise_queries(
    labelled: list[LabelledQuery] | list[dict[str, Any]],
) -> list[LabelledQuery]:
    out: list[LabelledQuery] = []
    for q in labelled:
        if isinstance(q, LabelledQuery):
            out.append(q)
        elif isinstance(q, dict):
            out.append(LabelledQuery(query=q["query"], relevant_ids=list(q["relevant_ids"])))
        else:
            raise TypeError(f"labelled item has unsupported type {type(q).__name__}")
    return out
