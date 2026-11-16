"""Reproducible real multi-person analysis and optional unified rendering."""
from pathlib import Path
import json,time,sys,torch
from scenemind_mvp.ui.catalog import catalog
from scenemind_mvp.unified.pipeline import analyze
from scenemind_mvp.unified.actions import ActionModel
from scenemind_mvp.detection.detector import Detector
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/phase3';OUT.mkdir(exist_ok=True)
def main():
    config=json.loads((ROOT/'configs/unified.json').read_text());model=ActionModel(ROOT/'data/models/actions');detector=Detector(__import__('scenemind_mvp.detection.schema',fromlist=['DetectionConfig']).DetectionConfig(**{**config['detection'],'classes':tuple(config['detection']['classes'])}),ROOT/'data/models/yolo')
    x=torch.ones((100,100),device='cuda');y=x@x;torch.cuda.synchronize();runs=[]
    for prefix in ['011209','c242ea']:
        v=next(v for v in catalog(ROOT/'data',[]) if v['id'].startswith(prefix));started=time.perf_counter();a=analyze(Path(v['source']),ROOT/'data',config,model_factory=lambda:model,detector_factory=lambda c,p:detector)
        entry={'source':v['source'],'analysis_directory':a['directory'],'request_seconds':time.perf_counter()-started,'detection_performance':a['detection_performance'],'action_available':a['actions']['model_available'],'action_error':a['actions'].get('error'),'action_seconds':a['actions'].get('processing_seconds'),'intervals':a['actions']['intervals'],'person_tracklets':a['diagnostics']['person_tracklets']}
        if '--render' in sys.argv:
            from scenemind_mvp.unified.render import render
            from scenemind_mvp.unified.core import make_plan,match_summary
            entry['all_video']=render(a,make_plan('',a),ROOT/'data/unified-renders',{})
            entry['queries']={}
            for q in ['Show all walking people','Show all sitting people','Show all running people','Show Person 3','Person carrying a black bag']:
                plan=make_plan(q,a);summary=match_summary(plan,a);entry['queries'][q]=summary
                if q in ('Show all walking people','Show Person 3'):entry['queries'][q]['render']=render(a,plan,ROOT/'data/unified-renders',{'view':'Search Matches'})
        runs.append(entry);print(json.dumps(entry,indent=2),flush=True)
        (OUT/'actual-analysis.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'synchronized_cuda_value':y[0,0].item(),'runs':runs},indent=2))
if __name__=='__main__':main()
