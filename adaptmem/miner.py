"""Hard-negative mining over a corpus.

For each labelled query, we encode the corpus once with a base model, retrieve
the top-K candidates, and pick the first non-relevant ones as hard negatives.
A "hard" negative shares lexical/semantic surface with the query but is
genuinely wrong — these are the examples a contrastive loss learns from.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptmem.types import LabelledQuery, TrainingPair


@dataclass
class CorpusEntry:
    id: str
    text: str


class HardNegativeMiner:
    """Mines (anchor, positive, negative) triples from a corpus + labelled queries.

    Strategy:
    - Encode corpus once with a `base_model` (sentence-transformers SentenceTransformer).
    - For each query, retrieve top-K. Pick the first `n_negatives` non-relevant ids as
      hard negatives. Each (positive, negative) pair becomes a separate TrainingPair,
      multiplying the training signal by n_negatives.
    - If fewer hard negatives found in top-K, fall back to random non-relevant ids.
    - One set of triples is emitted per (query, relevant_id).
    """

    def __init__(
        self,
        base_model: Any,
        top_k_mine: int = 10,
        n_negatives: int = 1,
        seed: int = 42,
    ):
        self.base_model = base_model
        self.top_k_mine = top_k_mine
        self.n_negatives = n_negatives
        self.rng = random.Random(seed)

    def mine(
        self, corpus: list[CorpusEntry], queries: list[LabelledQuery]
    ) -> list[TrainingPair]:
        if not corpus:
            return []
        ids = [c.id for c in corpus]
        texts = [c.text for c in corpus]
        embs = self.base_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        pairs: list[TrainingPair] = []
        for q in queries:
            qv = self.base_model.encode(
                [q.query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            scores = embs @ qv  # (N,)
            k = min(self.top_k_mine, len(ids))
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            top_ids = [ids[i] for i in top_idx]

            relevant = set(q.relevant_ids)

            # Collect up to n_negatives hard negatives from top-K
            neg_indices: list[int] = []
            for j, sid in zip(top_idx, top_ids):
                if sid not in relevant:
                    neg_indices.append(j)
                    if len(neg_indices) >= self.n_negatives:
                        break

            # Fill remaining slots with random non-relevant ids
            if len(neg_indices) < self.n_negatives:
                non_rel_pool = [
                    i for i, sid in enumerate(ids) if sid not in relevant and i not in neg_indices
                ]
                need = self.n_negatives - len(neg_indices)
                if non_rel_pool:
                    neg_indices.extend(self.rng.sample(non_rel_pool, min(need, len(non_rel_pool))))

            if not neg_indices:
                continue

            id_to_text = {sid: txt for sid, txt in zip(ids, texts)}
            for rel_id in q.relevant_ids:
                if rel_id not in id_to_text:
                    continue
                for neg_idx in neg_indices:
                    pairs.append(
                        TrainingPair(
                            anchor=q.query,
                            positive=id_to_text[rel_id],
                            negative=texts[neg_idx],
                            pos_id=rel_id,
                            neg_id=ids[neg_idx],
                        )
                    )

        return pairs
