"""Validated tracking configuration and JSON observation schema."""
from dataclasses import asdict, dataclass
import math

@dataclass(frozen=True)
class DetectionConfig:
    model: str = "yolo11n.pt"
    confidence: float = .25
    every_n_frames: int = 1
    image_size: int = 640
    device: str = "auto"
    tracking: bool = True
    classes: tuple[int, ...] = (0, 24, 26, 28, 56)
    low_threshold: float = .1
    high_threshold: float = .25
    new_track_threshold: float = .25
    lost_seconds: float = 1.0
    match_threshold: float = .8

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu or cuda")
        for value in (self.confidence,self.low_threshold,self.high_threshold,self.new_track_threshold,self.match_threshold):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Confidence/association thresholds must lie in (0,1]")
        if self.low_threshold > self.high_threshold:
            raise ValueError("Low threshold exceeds high threshold")
        if not isinstance(self.every_n_frames,int) or self.every_n_frames < 1:
            raise ValueError("Sampling interval must be a positive integer")
        if not isinstance(self.image_size,int) or self.image_size < 32 or self.image_size % 32:
            raise ValueError("Image size must be a positive multiple of 32")
        if not math.isfinite(self.lost_seconds) or self.lost_seconds < 0:
            raise ValueError("Lost-track duration must be nonnegative")
        if not self.classes or any(not isinstance(c,int) or not 0<=c<80 for c in self.classes):
            raise ValueError("Select valid COCO class IDs")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Duplicate classes")

    def to_dict(self) -> dict:
        result=asdict(self)
        result['classes']=list(self.classes)
        return result

@dataclass(frozen=True)
class Observation:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float,float,float,float]
    track_id: str | None = None

    def __post_init__(self) -> None:
        if not 0<=self.class_id<80 or not self.label:
            raise ValueError("Invalid class")
        if not math.isfinite(self.confidence) or not 0<=self.confidence<=1:
            raise ValueError("Invalid confidence")
        box=self.bbox_xyxy
        if len(box)!=4 or not all(math.isfinite(x) for x in box) or not 0<=box[0]<box[2] or not 0<=box[1]<box[3]:
            raise ValueError("Invalid pixel bounding box")
        if self.track_id is not None and not self.track_id.startswith(f"c{self.class_id}-t"):
            raise ValueError("Track ID must be class-namespaced")

    def to_dict(self) -> dict:
        result=asdict(self)
        result['bbox_xyxy']=list(self.bbox_xyxy)
        result['track_state']='matched' if self.track_id else 'unassigned'
        return result
