"""Hard-negative mining over a corpus.

For each labelled query, we encode the corpus once with a base model, retrieve
the top-K candidates, and pick the first non-relevant one as a hard negative.
A "hard" negative shares lexical/semantic surface with the query but is
genuinely wrong — these are the examples a contrastive loss learns from.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

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
    - For each query, retrieve top-K. Pick the first non-relevant id as the hard negative.
    - If all top-K are relevant, fall back to a uniformly-random non-relevant id.
    - One triple is emitted per (query, relevant_id) — multi-label queries spawn multiple
      training pairs, all anchored on the same query but with distinct positives.
    """

    def __init__(self, base_model, top_k_mine: int = 10, seed: int = 42):
        self.base_model = base_model
        self.top_k_mine = top_k_mine
        self.rng = random.Random(seed)

    def mine(
        self, corpus: list[CorpusEntry], queries: list[LabelledQuery]
    ) -> list[TrainingPair]:
        if not corpus:
            return []
        # Encode corpus once
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
            # Pick first non-relevant in top-K
            neg_idx = None
            for j, sid in zip(top_idx, top_ids):
                if sid not in relevant:
                    neg_idx = j
                    break

            if neg_idx is None:
                # Random non-relevant from full corpus
                non_rel_pool = [i for i, sid in enumerate(ids) if sid not in relevant]
                if not non_rel_pool:
                    continue
                neg_idx = self.rng.choice(non_rel_pool)

            neg_text = texts[neg_idx]
            neg_id = ids[neg_idx]

            id_to_text = {sid: txt for sid, txt in zip(ids, texts)}
            for rel_id in q.relevant_ids:
                if rel_id not in id_to_text:
                    continue
                pairs.append(
                    TrainingPair(
                        anchor=q.query,
                        positive=id_to_text[rel_id],
                        negative=neg_text,
                        pos_id=rel_id,
                        neg_id=neg_id,
                    )
                )

        return pairs
