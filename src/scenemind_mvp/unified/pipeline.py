"""Cached query-independent detection, tracking, diagnostics and temporal inference."""
from __future__ import annotations
from pathlib import Path
import json,time,hashlib,tempfile
from ..pipeline import digest
from ..video import inspect_video
from ..detection.schema import DetectionConfig
from ..detection.pipeline import process_video
from ..association.pipeline import process_associations
from ..association.core import AssociationConfig
from ..association.diagnostics import diagnose
from ..temporal.pipeline import process_timeline
from ..temporal.core import TemporalConfig
from .actions import ActionModel,analyze_actions
from .cache import cache_lock,publish

def load_analysis(directory: Path) -> dict:
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if digest(Path(manifest['source']))!=manifest['source_sha256']:raise ValueError('Unified source changed')
    if digest(Path(manifest['detection_directory'])/'records.json')!=manifest['identity']['records_sha256']:raise ValueError('Unified detections changed')
    if digest(directory/'actions.json')!=manifest['actions_sha256']:raise ValueError('Unified temporal cache changed')
    records=json.loads((Path(manifest['detection_directory'])/'records.json').read_text(encoding='utf-8'))
    return {**manifest,'directory':str(directory.resolve()),'frames':records['frames'],'actions':json.loads((directory/'actions.json').read_text(encoding='utf-8')),'diagnostics':json.loads((directory/'diagnostics.json').read_text(encoding='utf-8'))}

def _analyze(source: Path,data: Path,config: dict,model_factory=None,detector_factory=None) -> dict:
    started=time.perf_counter();d=dict(config['detection']);d['classes']=tuple(d['classes']);det=process_video(source,data/'detection',DetectionConfig(**d),detector_factory=detector_factory)
    directory=Path(det['directory']);records=directory/'records.json';weights=data/'models/actions/SLOWFAST_8x8_R50_DETECTION.pyth'
    identity={'revision':'unified-ava-v2','source_sha256':digest(source),'records_sha256':digest(records),'config':config,'weights_sha256':digest(weights) if weights.is_file() else None}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();destination=data/'unified'/key
    if (destination/'manifest.json').exists():return load_analysis(destination)
    frames=json.loads(records.read_text(encoding='utf-8'))['frames'];metadata=det['manifest']['metadata'];diagnostics=diagnose(frames,metadata['fps'],d['every_n_frames'])
    associations=process_associations(directory,data/'associations',AssociationConfig());temporal=process_timeline(Path(associations['directory']),data/'timelines',TemporalConfig());events=json.loads((Path(temporal['directory'])/'timeline.json').read_text())['events']
    try:
        model=(model_factory or (lambda:ActionModel(data/'models/actions',d['device'])))()
        actions=analyze_actions(source,frames,metadata,config['actions'],model,events)
    except (RuntimeError,OSError,ImportError,ValueError) as error:
        actions={'model_available':False,'error':str(error),'observations':[],'intervals':[],'limits':['Action unknown; temporal model unavailable or inference failed. Detection and tracking remain usable.']}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='analysis-',dir=destination.parent) as temp:
        staging=Path(temp);(staging/'actions.json').write_text(json.dumps(actions,indent=2),encoding='utf-8');(staging/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
        manifest={'identity':identity,'source':str(source.resolve()),'source_sha256':identity['source_sha256'],'metadata':metadata,'detection_directory':str(directory.resolve()),'association_directory':associations['directory'],'temporal_directory':temporal['directory'],'processing_seconds':time.perf_counter()-started,'detection_performance':{k:v for k,v in det['manifest'].items() if k in ('processed_fps','processing_seconds','peak_gpu_allocated_mib','peak_gpu_reserved_mib','unique_person_track_ids','unique_track_ids','max_simultaneous_person_detections','detections_by_class')},'actions_sha256':digest(staging/'actions.json'),'limits':['All 80 COCO categories supported; arbitrary bags/objects are not recognized.','IDs are temporary tracklets, not distinct real-world people.','Model-supported actions are not independent ground truth.']}
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');publish(staging,destination,lambda p:_analysis_valid(p))
    return load_analysis(destination)


def _analysis_valid(path: Path) -> bool:
    try:load_analysis(path);return True
    except (OSError,ValueError,KeyError):return False

def analyze(source: Path,data: Path,config: dict,model_factory=None,detector_factory=None) -> dict:
    weights=data/'models/actions/SLOWFAST_8x8_R50_DETECTION.pyth'
    key=hashlib.sha256(json.dumps({'source':digest(source),'config':config,'weights':digest(weights) if weights.is_file() else None},sort_keys=True).encode()).hexdigest()
    with cache_lock(data/'unified',key):return _analyze(source,data,config,model_factory,detector_factory)
