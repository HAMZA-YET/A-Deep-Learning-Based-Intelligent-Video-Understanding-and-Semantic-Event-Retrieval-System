"""Validated, portable JSON configuration."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

@dataclass(frozen=True)
class Config:
    sampling_fps: float = 1.0
    batch_size: int = 16
    model: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "auto"
    top_k: int = 5
    min_gap_seconds: float = 4.0
    min_similarity: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sampling_fps) or self.sampling_fps <= 0:
            raise ValueError("Sampling FPS must be positive and finite")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("Top-K must be a positive integer")
        if not math.isfinite(self.min_gap_seconds) or self.min_gap_seconds < 0:
            raise ValueError("Minimum time gap must be finite and nonnegative")
        if not math.isfinite(self.min_similarity) or not -1 <= self.min_similarity <= 1:
            raise ValueError("Similarity cutoff must lie in [-1, 1]")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu or cuda")
        if not self.model.strip() or not self.pretrained.strip():
            raise ValueError("Model and pretrained weights are required")

    def embedding_config(self) -> dict:
        # Query presentation parameters do not change the indexed representation.
        return {k: v for k, v in asdict(self).items()
                if k not in ("top_k", "min_gap_seconds", "min_similarity")}

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls(**json.loads(path.read_text(encoding="utf-8")))
