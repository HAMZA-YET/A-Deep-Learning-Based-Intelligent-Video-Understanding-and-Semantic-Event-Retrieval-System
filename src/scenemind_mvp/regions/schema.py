from dataclasses import dataclass,asdict
import math

@dataclass(frozen=True)
class RegionConfig:
    sampling_fps: float=1.
    batch_size: int=16
    padding: float=.2
    min_crop_side: int=48
    detector_sizes: tuple[int,...]=(640,960)
    confidence: float=.1
    context_tiles: bool=True
    region_weight: float=.5
    aggregate_top_regions: int=2
    dedup_iou: float=.6
    track_match_iou: float=.5
    def __post_init__(self):
        if not math.isfinite(self.sampling_fps) or self.sampling_fps<=0:raise ValueError('Sampling FPS must be positive')
        if not isinstance(self.batch_size,int) or self.batch_size<1:raise ValueError('Batch size must be positive')
        if not math.isfinite(self.padding) or not 0<=self.padding<=2:raise ValueError('Invalid crop padding')
        if not isinstance(self.min_crop_side,int) or self.min_crop_side<1:raise ValueError('Invalid crop minimum')
        if not self.detector_sizes or any(not isinstance(s,int) or s<32 or s%32 for s in self.detector_sizes):raise ValueError('Invalid detector sizes')
        if not 0<self.confidence<=1 or not 0<self.dedup_iou<=1 or not 0<self.track_match_iou<=1:raise ValueError('Invalid detector/IoU thresholds')
        if not math.isfinite(self.region_weight) or not 0<=self.region_weight<=1:raise ValueError('Invalid fusion weight')
        if not isinstance(self.aggregate_top_regions,int) or self.aggregate_top_regions<1:raise ValueError('Invalid aggregation count')
    def to_dict(self):return asdict(self)
