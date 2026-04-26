"""Shared dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import InputExample


@dataclass
class LabelledQuery:
    """A user-supplied training signal: query + ids of corpus chunks that satisfy it."""

    query: str
    relevant_ids: list[str]
    """IDs of chunks/passages in your corpus that are relevant to this query."""


@dataclass
class RetrievalHit:
    chunk_id: str
    text: str
    score: float


@dataclass
class TrainingPair:
    """Output of hard-negative mining; consumed by the trainer."""

    anchor: str
    positive: str
    negative: str
    qid: str = ""
    pos_id: str = ""
    neg_id: str = ""

    def to_input_example(self) -> "InputExample":
        """Convert to sentence_transformers.InputExample (lazy import)."""
        from sentence_transformers import InputExample

        return InputExample(texts=[self.anchor, self.positive, self.negative])


@dataclass
class TrainConfig:
    """Knobs you'll tune. Defaults are the recipe that produced 0.9950 R@5 on LongMemEval."""

    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    top_k_mine: int = 10
    """How many top-K hits to inspect when picking a hard negative per (query, gt) pair."""
    extra_metadata: dict[str, Any] = field(default_factory=dict)
