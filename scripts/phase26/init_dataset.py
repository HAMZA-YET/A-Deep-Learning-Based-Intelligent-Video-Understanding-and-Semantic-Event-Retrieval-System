"""Freeze evaluation inputs and create unlabelled, model-blind human review files."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import json,hashlib,platform,subprocess,importlib.metadata,sys,html
from scenemind_mvp.video import inspect_video
from scenemind_mvp.pipeline import digest
root=Path('benchmarks/phase26');root.mkdir(exist_ok=True)
def save(path,value):
    if path.exists():raise FileExistsError(f'Preserving existing file: {path}')
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')
files=[p for base in ('src','configs','tests') for p in Path(base).rglob('*') if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in str(p) and 'phase26' not in str(p) and '/evaluation/' not in p.as_posix()]+[Path(p) for p in ('app.py','requirements.lock.txt','pyproject.toml')]
hashes={p.as_posix():digest(p) for p in sorted(files)}
import torch
probe=torch.randn(100,100,device='cuda' if torch.cuda.is_available() else 'cpu');result=probe@probe
if torch.cuda.is_available():torch.cuda.synchronize()
try:git=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip() or None
except OSError:git=None
snapshot={'created_utc':datetime.now(timezone.utc).isoformat(),'interpreter':sys.executable,'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),'git_head':git,'implementation_files':hashes,'implementation_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),'packages':{p:importlib.metadata.version(p) for p in ('torch','torchvision','open-clip-torch','opencv-python','faiss-cpu','streamlit','ultralytics','numpy')},'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/1024**2 if torch.cuda.is_available() else None,'gpu_probe_shape':list(result.shape),'gpu_probe_finite':bool(torch.isfinite(result).all())}
save(root/'implementation-before.json',snapshot)
queries=['Show a person walking.','Find someone carrying a black bag.','Show two people standing together.','Find a person picking up a bag.','Show someone leaving with a backpack.','Find all people near a bag.','Show a visible backpack.']
protocol={'schema':1,'status':'frozen_before_phase26_predictions','top_k':5,'min_gap_seconds':1.,'interval_convention':'[start,end)','timestamp_tolerance_seconds':.5,'temporal_iou_threshold':.3,'detection_iou_threshold':.5,'reviewer_required':'named human; predictions must not be used to create truth','query_relevance':'whole query must be satisfied in manually reviewed intervals; action positives require multi-frame evidence','precision_denominator':5,'hit_and_mrr_denominator':'reviewed positive queries only; negative and ambiguous cases reported separately','uncertain_intervals':'not positive or negative; exclude from accuracy','detector_scope':'exhaustively annotated frames/classes only; predictions outside scope excluded','tracking_scope':'continuous exhaustively reviewed identity sequences only; no identity inference from tracker IDs','structured_order':'existing temporal search, observed before candidate then start timestamp; no CLIP vector or score','queries':queries,'ranking_config_sha256':digest(Path('configs/hybrid.json')),'config_hashes':{str(p):digest(p) for p in Path('configs').glob('*.json')},'repetitions':5,'warmup_queries':1}
save(root/'protocol.json',protocol)
videos=[]
for key,path,attr,split in [('sabana_grande','data/samples/sabana_grande.mp4','data/samples/sabana_grande_attribution.json','development_reused'),('fitwalking','data/phase26/fitwalking.mp4','data/phase26/fitwalking_attribution.json','pending_human_review'),('winery','data/phase26/winery.mp4','data/phase26/winery_attribution.json','pending_human_review')]:
    p=Path(path);info=asdict(inspect_video(p));a=json.loads(Path(attr).read_text(encoding='utf-8'))
    record={'video_id':key,'path':path,'sha256':digest(p),'metadata':info,'attribution_file':attr,'license':a['license'],'license_url':a['license_url'],'source_page':a.get('source_page',a.get('source')),'creator':a['creator'],'split':split,'annotation_file':str(root/f'{key}-review.json')};videos.append(record)
    review={'schema':1,'video_id':key,'source_sha256':record['sha256'],'duration':info['duration'],'review_status':'pending','reviewer':None,'reviewed_utc':None,'independent_of_model_predictions':None,'queries':[{'query_id':f'q{i+1}','query':q,'verdict':'pending','relevant_intervals':None,'person_references':[],'object_categories':[],'notes':''} for i,q in enumerate(queries)],'events':[],'detection_frames':[{'frame_index':f,'timestamp':f/info['fps'],'reviewed':False,'exhaustive_classes':[],'boxes':[]} for f in sorted({0,info['frame_count']//2,info['frame_count']-1})],'tracking_sequences':[],'notes':'Blank ground truth. Human labels required. Do not copy model predictions.'}
    save(Path(record['annotation_file']),review)
save(root/'manifest.json',{'schema':1,'dataset_id':'scenemind-phase26-v1','videos':videos,'missing_coverage':['No independently reviewed pickup/carrying/departure positives yet','Only three short clips; not representative','Human review pending for all new annotations'],'acquisition_failures':['Pexels official pages and downloads returned HTTP403; no bypass attempted. Replaced by Wikimedia CC videos.']})
sections=[]
for v in videos:
    sections.append(f'<section><h2>{html.escape(v["video_id"])}</h2><p>{html.escape(v["license"])} | {html.escape(v["annotation_file"])}</p><video controls preload="metadata" width="640" src="{Path(v["path"]).resolve().as_uri()}"></video><p><input type="number" step="0.04" min="0" value="0"><button onclick="this.closest(\'section\').querySelector(\'video\').currentTime=Number(this.previousElementSibling.value)">Seek seconds</button><button onclick="let v=this.closest(\'section\').querySelector(\'video\');v.pause();v.currentTime+=0.04">Next frame</button><output></output></p></section>')
(root/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>SceneMind blind annotation review</title><style>body{font:18px sans-serif;margin:30px}section{margin:35px 0}video{max-height:520px;max-width:95%}</style><h1>Human review: source videos only</h1><p>Read ANNOTATION_GUIDELINES.md. Edit the matching review JSON; no predictions are shown. Labels remain pending until a named human signs them.</p>'+''.join(sections)+'<script>document.querySelectorAll("video").forEach(v=>v.ontimeupdate=()=>v.closest("section").querySelector("output").value=v.currentTime.toFixed(3)+" seconds")</script>',encoding='utf-8')
print(json.dumps({'snapshot':snapshot['implementation_sha256'],'videos':[(v['video_id'],v['metadata']['duration']) for v in videos]}))
