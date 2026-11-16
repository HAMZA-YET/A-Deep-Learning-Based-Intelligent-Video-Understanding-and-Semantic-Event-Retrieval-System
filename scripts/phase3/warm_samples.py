from pathlib import Path
import json,time
from scenemind_mvp.ui.catalog import catalog
from scenemind_mvp.unified.pipeline import analyze
from scenemind_mvp.unified.actions import ActionModel
from scenemind_mvp.detection.detector import Detector
from scenemind_mvp.detection.schema import DetectionConfig
ROOT=Path(__file__).resolve().parents[2];config=json.loads((ROOT/'configs/unified.json').read_text());model=ActionModel(ROOT/'data/models/actions');d={**config['detection'],'classes':tuple(config['detection']['classes'])};detector=Detector(DetectionConfig(**d),ROOT/'data/models/yolo')
for v in catalog(ROOT/'data',[]):
    start=time.perf_counter();a=analyze(Path(v['source']),ROOT/'data',config,lambda:model,lambda c,p:detector);print(v['name'],a['directory'],time.perf_counter()-start,a['actions']['model_available'],flush=True)
    from scenemind_mvp.unified.render import render
    from scenemind_mvp.unified.core import make_plan
    print(render(a,make_plan('',a),ROOT/'data/unified-renders',{})['video'],flush=True)
