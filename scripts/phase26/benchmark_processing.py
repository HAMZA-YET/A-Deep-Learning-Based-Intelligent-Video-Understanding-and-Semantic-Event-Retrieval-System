"""Fresh artifacts, cached weights, fixed configurations; no labels enter processing."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
import json,time,hashlib,contextlib,sys
import torch
from scenemind_mvp.config import Config
from scenemind_mvp.model import Encoder
from scenemind_mvp.pipeline import index_video,digest
from scenemind_mvp.detection.schema import DetectionConfig
from scenemind_mvp.detection.detector import Detector
from scenemind_mvp.detection.pipeline import process_video
from scenemind_mvp.association.core import AssociationConfig
from scenemind_mvp.association.pipeline import process_associations
from scenemind_mvp.temporal.core import TemporalConfig
from scenemind_mvp.temporal.pipeline import process_timeline
root=Path('benchmarks/phase26');manifest=json.loads((root/'manifest.json').read_text());protocol_bytes=(root/'protocol.json').read_bytes()
if any(digest(Path(path))!=value for path,value in json.loads(protocol_bytes)['config_hashes'].items()):raise ValueError('Frozen configurations changed before processing')
run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=Path('data/phase26/runs')/run_id;out.mkdir(parents=True)
raw=root/'raw'/run_id;raw.mkdir(parents=True)
config=Config.load(Path('configs/default.json'));d=json.loads(Path('configs/detection.json').read_text());d['classes']=tuple(d['classes']);detconfig=DetectionConfig(**d)
ac=AssociationConfig(**json.loads(Path('configs/association.json').read_text()));tc=TemporalConfig(**json.loads(Path('configs/temporal.json').read_text()))
def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()
def measure(fn):
    sync();started=time.perf_counter();value=fn();sync();return value,time.perf_counter()-started
with contextlib.redirect_stdout(sys.stderr):
    encoder,clip_load=measure(lambda:Encoder(config,Path('data/models')))
    detector,yolo_load=measure(lambda:Detector(detconfig,Path('data/models/yolo')))
report={'run_id':run_id,'protocol_sha256':hashlib.sha256(protocol_bytes).hexdigest(),'model_loading_seconds':{'openclip':clip_load,'yolo':yolo_load},'semantic_config':asdict(config),'detection_config':detconfig.to_dict(),'videos':[],'cache_policy':'Fresh isolated artifact roots; existing downloaded weights and OS filesystem caches. Both models stay resident. No accuracy labels enter processing.'}
all_started=time.perf_counter()
for video in manifest['videos']:
    source=Path(video['path']);assert digest(source)==video['sha256']
    started=time.perf_counter();stages={};peaks={}
    def stage(name,fn):
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        with contextlib.redirect_stdout(sys.stderr):value,seconds=measure(fn)
        stages[name]=seconds;peaks[name]={'allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'reserved_mib':torch.cuda.max_memory_reserved()/1024**2} if torch.cuda.is_available() else None
        return value
    semantic=stage('semantic_index',lambda:index_video(source,out/'indexes',config,lambda:encoder))
    detection=stage('detection_tracking_with_video_export',lambda:process_video(source,out/'detection',detconfig,detector_factory=lambda *_:detector))
    association=stage('association_with_video_export',lambda:process_associations(Path(detection['directory']),out/'associations',ac))
    temporal=stage('temporal_with_video_export',lambda:process_timeline(Path(association['directory']),out/'timelines',tc))
    assert not semantic.cache_hit and not any(x['cache_hit'] for x in (detection,association,temporal))
    bundle={'schema':1,'source':str(source.resolve()),'source_sha256':video['sha256'],'semantic_directory':str(semantic.directory.resolve()),'detection_directory':detection['directory'],'association_directory':association['directory'],'temporal_directory':temporal['directory']}
    bundle_path=out/f'{video["video_id"]}-bundle.json';bundle_path.write_text(json.dumps(bundle,indent=2),encoding='utf-8')
    quality=json.loads((Path(association['directory'])/'diagnostics.json').read_text());summary={k:v for k,v in quality.items() if k!='tracks'}
    item={'video_id':video['video_id'],'bundle':str(bundle_path.resolve()),'source_sha256':video['sha256'],'metadata':video['metadata'],'stage_wall_seconds':stages,'pipeline_total_seconds':time.perf_counter()-started,'gpu_peaks_per_stage':peaks,'semantic_manifest':semantic.manifest,'detection_manifest':detection['manifest'],'temporal_manifest':temporal['manifest'],'track_diagnostics':summary,'cache_hits':{'semantic':semantic.cache_hit,'detection':detection['cache_hit'],'association':association['cache_hit'],'temporal':temporal['cache_hit']}}
    report['videos'].append(item);(raw/'processing.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(video['video_id'],stages,flush=True)
report['all_video_pipeline_seconds']=time.perf_counter()-all_started
report['total_seconds_including_shared_model_loads']=report['all_video_pipeline_seconds']+clip_load+yolo_load
assert protocol_bytes==(root/'protocol.json').read_bytes()
assert all(digest(Path(p))==value for p,value in json.loads(protocol_bytes)['config_hashes'].items())
(raw/'processing.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(root/'latest-run.json').write_text(json.dumps({'run_id':run_id,'raw_directory':str(raw),'processing_file':str(raw/'processing.json')},indent=2),encoding='utf-8')
print('REPORT',raw/'processing.json',flush=True)
