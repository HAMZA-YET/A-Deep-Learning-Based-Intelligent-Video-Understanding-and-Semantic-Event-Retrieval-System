"""Identical reviewed frame pool and queries; compare old hybrid with region variants."""
from pathlib import Path
from dataclasses import replace
import json,time,csv
import numpy as np
from scenemind_mvp.regions.pipeline import load_regions
from scenemind_mvp.regions.schema import RegionConfig
from scenemind_mvp.regions.engine import RegionEngine
from scenemind_mvp.hybrid.engine import HybridEngine
from scenemind_mvp.hybrid.schema import HybridConfig
from scenemind_mvp.hybrid.evidence import load_bundle
from scenemind_mvp.model import Encoder
from scenemind_mvp.config import Config
from scenemind_mvp.evaluation.metrics import retrieval_metrics
from scenemind_mvp.pipeline import digest
root=Path('benchmarks/regions');labels_path=Path('benchmarks/hybrid-real-v1.json');labels_bytes=labels_path.read_bytes();labels=json.loads(labels_bytes);report=json.loads((root/'comparison.json').read_text());encoder=Encoder(Config(),Path('data/models'));experiments=[];csvrows=[]
for example in report['experiments']:
    if example['video_id']!='sabana_grande':continue
    index=load_regions(Path(example['region_directory']));_,evidence=load_bundle(Path(index.manifest['bundle']));d=index.manifest['identity']['region_config'];d['detector_sizes']=tuple(d['detector_sizes']);config=RegionConfig(**d);hybrid=replace(HybridConfig(**json.loads(Path('configs/hybrid.json').read_text())),top_k=5,min_gap_seconds=1.)
    regional=RegionEngine(index,evidence,config,hybrid);full=HybridEngine(index.full,evidence,hybrid);cases=[]
    for case in labels['queries']:
        if case.get('relevant_timestamps') is None:continue
        methods={name:{'trials':[]} for name in ('clip_only','hybrid_full_frame','region','region_hybrid')}
        for repetition in range(5):
            started=time.perf_counter();vector=encoder.encode_text(case['query'])
            if encoder.device=='cuda':encoder.torch.cuda.synchronize()
            encoding=time.perf_counter()-started
            order=list(methods);order=order[repetition%4:]+order[:repetition%4]
            for name in order:
                response=full.search(case['query'],vector,'hybrid' if name=='hybrid_full_frame' else 'clip_only',labels['candidate_timestamps']) if name in ('clip_only','hybrid_full_frame') else regional.search(case['query'],vector,name,strict_actions=False,candidate_timestamps=labels['candidate_timestamps'])
                methods[name]['trials'].append(encoding+response['ranking_seconds'])
                if repetition==0:methods[name]['result']=response
        for name,m in methods.items():
            m['metrics']=retrieval_metrics([r['timestamp'] for r in m['result']['results']],[[t,t+.001] for t in case['relevant_timestamps']],'positive',5,0.);m['median_query_ms']=float(np.median(m['trials'])*1000)
            csvrows.append({'config':example['config_path'],'query':case['query'],'method':name,**m['metrics'],'median_query_ms':m['median_query_ms']})
        cases.append({'query':case['query'],'methods':methods})
    aggregates={name:{**{metric:float(np.mean([c['methods'][name]['metrics'][metric] for c in cases])) for metric in ('hit_at_1','hit_at_5','precision_at_5','mrr')},'median_query_ms':float(np.median([c['methods'][name]['median_query_ms'] for c in cases]))} for name in methods}
    experiments.append({'config':example['config_path'],'aggregates':aggregates,'cases':cases});print(example['config_path'],aggregates)
assert labels_path.read_bytes()==labels_bytes
result={'label_sha256':digest(labels_path),'scope':'Three correlated assistant-reviewed visual development queries, exact common ten-frame pool. Not independent event accuracy.','experiments':experiments}
(root/'legacy-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (root/'legacy-comparison.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
