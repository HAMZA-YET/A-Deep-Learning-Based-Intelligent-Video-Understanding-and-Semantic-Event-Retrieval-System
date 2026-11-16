"""Fixed engineering ranking weights, unrelated to benchmark labels."""
from dataclasses import asdict,dataclass
import math

@dataclass(frozen=True)
class HybridConfig:
    semantic_weight: float = .75
    object_weight: float = .1
    track_weight: float = .05
    association_weight: float = .05
    event_weight: float = .05
    missing_penalty: float = .1
    conflict_penalty: float = .2
    alignment_tolerance: float = .08
    top_k: int = 5
    min_gap_seconds: float = 4.
    strict_events: bool = False
    def __post_init__(self):
        for key,value in asdict(self).items():
            if key=='strict_events':continue
            if not math.isfinite(value) or value<0:raise ValueError('Ranking settings must be nonnegative finite values')
        if self.top_k<1 or not isinstance(self.top_k,int):raise ValueError('Top-K must be a positive integer')
        total=sum(getattr(self,name) for name in ('semantic_weight','object_weight','track_weight','association_weight','event_weight'))
        if abs(total-1)>1e-6:raise ValueError('Ranking component weights must sum to one')
    def to_dict(self):return asdict(self)
