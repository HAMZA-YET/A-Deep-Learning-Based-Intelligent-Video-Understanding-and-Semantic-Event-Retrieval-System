from pathlib import Path
import json,time
from scenemind_mvp.ui.catalog import catalog
from scenemind_mvp.ui.service import prepare
ROOT=Path(__file__).resolve().parents[2]
S={'sampling_fps':1.,'batch_size':16,'model':'ViT-B-32','pretrained':'openai','device':'auto','max_results':20,'gap':1.,'cutoff':-1.,'strict':False,'padding':.2,'detector_sizes':[640,960],'detection_confidence':.1,'evidence_profile':'Baseline','full_hybrid':False}
for v in catalog(ROOT/'data',[]):
    if v['name'] in ('fitwalking.mp4','sabana_grande.mp4','winery.mp4','scenemind_demo.mp4'):
        start=time.perf_counter();path,kind=prepare(v,'Regions',S,ROOT,ROOT/'data');print(v['name'],str(path),time.perf_counter()-start,flush=True)

