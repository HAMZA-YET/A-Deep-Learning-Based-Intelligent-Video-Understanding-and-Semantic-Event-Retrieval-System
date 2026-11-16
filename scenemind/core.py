"""Dependency-free window planning and result selection."""
from dataclasses import asdict, dataclass
import math

@dataclass(frozen=True)
class Config:
    model: str = "ViT-B-32"
    pretrained: str = "openai"
    window_seconds: float = 8.0
    stride_seconds: float = 4.0
    frames_per_window: int = 8
    batch_size: int = 16

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.window_seconds, self.stride_seconds)):
            raise ValueError("Window and stride must be positive finite numbers")
        if self.stride_seconds > self.window_seconds:
            raise ValueError("Stride must not exceed window length")
        if self.frames_per_window < 1 or self.batch_size < 1:
            raise ValueError("Frame count and batch size must be positive")

    def to_dict(self):
        return asdict(self)

def windows(duration: float, config: Config) -> list[tuple[float, float]]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be positive and finite")
    result, start = [], 0.0
    while start < duration:
        end = min(duration, start + config.window_seconds)
        result.append((start, end))
        if end >= duration:
            break
        start += config.stride_seconds
    return result

def sample_times(start: float, end: float, count: int) -> list[float]:
    if count < 1 or not 0 <= start < end:
        raise ValueError("Invalid sampling interval/count")
    return [start + (i + 0.5) * (end - start) / count for i in range(count)]

def temporal_iou(a: dict, b: dict) -> float:
    overlap = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return overlap / union if union > 0 else 0.0

def select_results(candidates: list[dict], top_k: int, threshold: float,
                   overlap_threshold: float = 0.4) -> list[dict]:
    if top_k < 1 or not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Invalid top-k or threshold")
    selected = []
    for candidate in sorted(candidates, key=lambda row: row["score"], reverse=True):
        if not math.isfinite(candidate["score"]) or candidate["score"] < threshold:
            continue
        if any(temporal_iou(candidate, row) > overlap_threshold for row in selected):
            continue
        selected.append(candidate)
        if len(selected) == top_k:
            break
    return selected
