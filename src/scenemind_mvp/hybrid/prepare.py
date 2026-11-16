"""Offline preparation reuses every existing Phase 1/2 component and its caches."""
from pathlib import Path
import hashlib,json,time
from ..config import Config
from ..model import Encoder
from ..pipeline import index_video
from ..detection.schema import DetectionConfig
from ..detection.pipeline import process_video
from ..association.core import AssociationConfig
from ..association.pipeline import process_associations
from ..temporal.core import TemporalConfig
from ..temporal.pipeline import process_timeline

def prepare_bundle(video: Path,data: Path,semantic_config: Config,detector_config: DetectionConfig,encoder_factory=None,association_config: AssociationConfig | None=None,temporal_config: TemporalConfig | None=None) -> dict:
    encoder=None
    def factory():
        nonlocal encoder
        if encoder is None:encoder=encoder_factory() if encoder_factory else Encoder(semantic_config,data/'models')
        return encoder
    started=time.perf_counter()
    semantic=index_video(video,data/'indexes',semantic_config,factory)
    detection=process_video(video,data/'detection',detector_config)
    association=process_associations(Path(detection['directory']),data/'associations',association_config or AssociationConfig())
    temporal=process_timeline(Path(association['directory']),data/'timelines',temporal_config or TemporalConfig())
    bundle={'schema':1,'source':str(video.resolve()),'source_sha256':semantic.manifest['identity']['video_sha256'],
        'semantic_directory':str(semantic.directory.resolve()),'detection_directory':detection['directory'],
        'association_directory':association['directory'],'temporal_directory':temporal['directory']}
    key=hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest()
    target=data/'hybrid-bundles'/key;target.mkdir(parents=True,exist_ok=True)
    path=target/'bundle.json'
    if not path.exists():path.write_text(json.dumps(bundle,indent=2),encoding='utf-8')
    return {'bundle':str(path.resolve()),'preparation_seconds':time.perf_counter()-started,
        'semantic_cache_hit':semantic.cache_hit,'detection_cache_hit':detection['cache_hit'],
        'association_cache_hit':association['cache_hit'],'temporal_cache_hit':temporal['cache_hit']}
