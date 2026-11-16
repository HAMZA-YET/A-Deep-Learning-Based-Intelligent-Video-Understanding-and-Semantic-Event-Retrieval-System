from pathlib import Path
import json,sys,platform,importlib.metadata
from dataclasses import replace
from scenemind_mvp.pipeline import digest
from scenemind_mvp.detection.schema import DetectionConfig
import torch
root=Path('benchmarks/phase27');out=root/'protocol.json'
if out.exists():raise FileExistsError('Preserve existing frozen experiment')
base=json.loads(Path('configs/detection.json').read_text());base['classes']=tuple(base['classes']);base=DetectionConfig(**base)
profiles=[('baseline640',base),('640low',replace(base,confidence=.1)),('960',replace(base,image_size=960)),('960low',replace(base,image_size=960,confidence=.1)),('1280low',replace(base,image_size=1280,confidence=.1))]
files=[p for folder in ('src','tests','configs','benchmarks/phase26') for p in Path(folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in str(p) and '/raw/' not in p.as_posix() and '/summary/' not in p.as_posix() and '/recovery/' not in p.as_posix()]+[Path('app.py'),Path('pyproject.toml'),Path('benchmarks/missed-backpacks-v1.json'),Path('benchmarks/events-real-v1.json')]
snapshot={'files':{str(p):digest(p) for p in files},'interpreter':sys.executable,'platform':platform.platform(),'torch':torch.__version__,'ultralytics':importlib.metadata.version('ultralytics'),'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
a=torch.randn(100,100,device='cuda' if torch.cuda.is_available() else 'cpu');b=a@a
if torch.cuda.is_available():torch.cuda.synchronize()
snapshot['gpu_probe']={'shape':list(b.shape),'finite':bool(torch.isfinite(b).all())}
(root/'snapshot.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
protocol={'detectors':{name:c.to_dict() for name,c in profiles},'same_evaluation_frames':'All five frozen backpack frames plus every whole second of each manifest video; positive-only legacy scope separate from unlabelled coverage','repetitions':3,'warmup_per_profile':2,'timing':'Synchronized infer() only; predecoded images, no annotation/video export time. GPU allocator peaks after warmup; model loading separate.','spot_check_iou':.3,'human_detection_iou':.5,'model':'Existing pretrained YOLO11n; no new weights required','tracker_profiles':['configs/detection.json','configs/tracker-continuity.json','configs/tracker-low-init.json'],'full_video_inference_profiles':['baseline640','960low'],'tracking_candidates':'Same class-aware low-threshold YOLO candidate arrays per detector compared across profiles, no ID merging','selection_policy':'Development experiments only. Keep production baseline. No final choice from five positives or unsigned labels. New Phase26 clips stay closed to experimental detector/tracker tuning; their baseline artifacts may be audited only.','source_manifest_sha256':digest(Path('benchmarks/phase26/manifest.json')),'inputs':{'benchmarks/missed-backpacks-v1.json':digest(Path('benchmarks/missed-backpacks-v1.json')),'benchmarks/events-real-v1.json':digest(Path('benchmarks/events-real-v1.json'))}}
out.write_text(json.dumps(protocol,indent=2),encoding='utf-8');print(snapshot['gpu_probe'])
