"""Same encodings, videos, queries and candidate pools; weights never read evaluation labels."""
from pathlib import Path
from dataclasses import replace
import argparse,json,time,hashlib
import numpy as np
from scenemind_mvp.config import Config
from scenemind_mvp.model import Encoder
from scenemind_mvp.hybrid.schema import HybridConfig
from scenemind_mvp.hybrid.evidence import load_bundle
from scenemind_mvp.hybrid.engine import HybridEngine,MODELED
from scenemind_mvp.hybrid.query import parse_query

parser=argparse.ArgumentParser()
parser.add_argument('--demo-bundle',type=Path)
parser.add_argument('--real-bundle',type=Path)
parser.add_argument('--config',type=Path,default=Path('configs/hybrid.json'))
parser.add_argument('--output',type=Path,default=Path('docs/hybrid-evaluation.json'))
args=parser.parse_args()
config_bytes=args.config.read_bytes();settings=HybridConfig(**json.loads(config_bytes))
label_paths=[Path('data/demo/scenemind_demo_ground_truth.json'),Path('benchmarks/hybrid-real-v1.json')]
frozen={str(p):p.read_bytes() for p in label_paths}
encoder=Encoder(Config.load(Path('configs/default.json')),Path('data/models'))
if encoder.device=='cuda':encoder.torch.cuda.reset_peak_memory_stats()
cohorts=[]
for name,bundle_arg,label_path,gap in [('demo',args.demo_bundle,label_paths[0],4.),('real',args.real_bundle,label_paths[1],1.)]:
    bundle=bundle_arg or Path(json.loads(Path(f'docs/hybrid-{name}-preparation.json').read_text())['bundle'])
    index,evidence=load_bundle(bundle);labels=json.loads(frozen[str(label_path)])
    if name=='real' and labels['source_sha256']!=evidence.source_sha256:raise ValueError('Real evaluation source mismatch')
    config=replace(settings,min_gap_seconds=gap)
    engine=HybridEngine(index,evidence,config)
    allowed=labels.get('candidate_timestamps') if name=='real' else None
    comparisons=[]
    for case in labels['queries']:
        query=case['query'];start=time.perf_counter();vector=encoder.encode_text(query)
        if encoder.device=='cuda':encoder.torch.cuda.synchronize()
        encode_seconds=time.perf_counter()-start
        methods={}
        for mode in ('clip_only','hybrid'):
            result=engine.search(query,vector,mode,candidate_timestamps=allowed)
            result['query_seconds_with_shared_encoding']=encode_seconds+result['ranking_seconds']
            if name=='demo':
                relevance=lambda row:any(start<=row['timestamp']<end for start,end in case['intervals'])
                eligible=True
            else:
                times=case.get('relevant_timestamps');eligible=times is not None
                relevance=lambda row:times is not None and any(abs(row['timestamp']-t)<1e-5 for t in times)
            rows=result['results'];hits=[bool(relevance(row)) for row in rows] if eligible else None
            requested=parse_query(query)
            event_eligible=eligible and requested.action in MODELED
            supported=[bool(row['event_claim'] and relevance(row)) for row in rows] if event_eligible else None
            unsupported=0
            for row in rows:
                claim=row['event_claim']
                if claim:
                    events=row['evidence']['temporal_events']
                    justified=all(any(e['event_id']==identifier and e['action']==claim['action'] and e['status'] in ('candidate','observed') for e in events) for identifier in claim['event_ids'])
                    if not justified or requested.unsupported_constraints:unsupported+=1
            metrics={'visual_relevance_verified':eligible,'hit_at_1':bool(hits and hits[0]) if eligible else None,
                'hit_at_5':any(hits) if eligible else None,'precision_at_5':sum(hits)/5 if eligible else None,
                'returned_results':len(rows),'supported_event_hit_at_1':bool(supported and supported[0]) if event_eligible else None,
                'supported_event_hit_at_5':any(supported) if event_eligible else None,
                'event_claims':sum(row['event_claim'] is not None for row in rows),'unsupported_claim_audit_violations':unsupported,
                'explicit_uncertainty':result['status'] in ('uncertain','no_reliable_match')}
            methods[mode]={'metrics':metrics,'result':result}
        comparisons.append({'query':query,'label_note':case.get('reason'),'must_remain_uncertain':case.get('must_remain_uncertain',False),
            'text_encoding_seconds':encode_seconds,'methods':methods})
    aggregates={}
    for mode in ('clip_only','hybrid'):
        metrics=[row['methods'][mode]['metrics'] for row in comparisons]
        measured=[m for m in metrics if m['visual_relevance_verified']]
        event_measured=[m for m in metrics if m['supported_event_hit_at_1'] is not None]
        latencies=[row['methods'][mode]['result']['query_seconds_with_shared_encoding'] for row in comparisons]
        ranks=[row['methods'][mode]['result']['ranking_seconds'] for row in comparisons]
        aggregates[mode]={'verified_visual_queries':len(measured),'hit_at_1':sum(m['hit_at_1'] for m in measured)/len(measured) if measured else None,
            'hit_at_5':sum(m['hit_at_5'] for m in measured)/len(measured) if measured else None,
            'mean_precision_at_5':sum(m['precision_at_5'] for m in measured)/len(measured) if measured else None,
            'reviewed_event_queries':len(event_measured),
            'supported_event_hit_at_1':sum(m['supported_event_hit_at_1'] for m in event_measured)/len(event_measured) if event_measured else None,
            'supported_event_hit_at_5':sum(m['supported_event_hit_at_5'] for m in event_measured)/len(event_measured) if event_measured else None,
            'median_query_seconds':float(np.median(latencies)),'median_ranking_seconds':float(np.median(ranks)),
            'event_claims':sum(m['event_claims'] for m in metrics),'unsupported_claim_audit_violations':sum(m['unsupported_claim_audit_violations'] for m in metrics),
            'uncertain_cases_required':sum(row['must_remain_uncertain'] for row in comparisons),
            'required_cases_explicitly_uncertain':sum(row['must_remain_uncertain'] and row['methods'][mode]['metrics']['explicit_uncertainty'] for row in comparisons)}
    cohorts.append({'cohort':name,'bundle':str(bundle.resolve()),'candidate_timestamps':allowed,'min_gap_seconds':gap,
        'source_sha256':evidence.source_sha256,'aggregates':aggregates,'queries':comparisons})
assert args.config.read_bytes()==config_bytes
assert all(Path(path).read_bytes()==value for path,value in frozen.items())
report={'config_sha256':hashlib.sha256(config_bytes).hexdigest(),'weights':settings.to_dict(),
    'label_hashes':{path:hashlib.sha256(value).hexdigest() for path,value in frozen.items()},'labels_unchanged':True,
    'device':encoder.device,'cohorts':cohorts,
    'peak_gpu_allocated_mib':encoder.torch.cuda.max_memory_allocated()/1024**2 if encoder.device=='cuda' else None,
    'peak_gpu_reserved_mib':encoder.torch.cuda.max_memory_reserved()/1024**2 if encoder.device=='cuda' else None,
    'implementation_hashes':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in Path('src/scenemind_mvp/hybrid').glob('*.py')},
    'limitations':['Visual interval relevance is not physical action proof.',
        'Real set has 3 correlated visual queries on one small reviewed frame pool; other accuracy labels are unverified.',
        'Unsupported-claim audit checks API grounding, not real-world event precision. CLIP-only already makes no event claims.',
        'Supported-event hits assess attached structured candidate/geometry support; not carrying/contact accuracy.',
        'Single observations per query, shared text embedding time; model load excluded. Not a production latency benchmark.']}
args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'device':encoder.device,'cohorts':[{'cohort':c['cohort'],'aggregates':c['aggregates']} for c in cohorts]},indent=2))
