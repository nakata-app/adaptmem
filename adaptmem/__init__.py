"""adaptmem — domain adaptation for retrieval, in five lines."""
from adaptmem.core import AdaptMem
from adaptmem.miner import HardNegativeMiner
from adaptmem.types import LabelledQuery, RetrievalHit, TrainConfig

__all__ = ["AdaptMem", "HardNegativeMiner", "LabelledQuery", "RetrievalHit", "TrainConfig"]
__version__ = "0.7.0"
