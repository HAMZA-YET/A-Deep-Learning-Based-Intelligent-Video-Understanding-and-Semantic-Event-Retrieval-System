"""Three-method repeated retrieval benchmark. Pending human labels stay null."""
from pathlib import Path
from dataclasses import replace
import argparse,json,time,csv,statistics,hashlib
from scenemind_mvp.config import Config
from scenemind_mvp.model import Encoder
from scenemind_mvp.hybrid.schema import HybridConfig
from scenemind_mvp.hybrid.evidence import load_bundle
from scenemind_mvp.hybrid.engine import HybridEngine
from scenemind_mvp.evaluation.retrieval import structured_search
from scenemind_mvp.evaluation.metrics import retrieval_metrics,validate_review,detection_metrics,tracking_metrics,temporal_case_metrics
from scenemind_mvp.temporal.evaluation import evaluate_cases
from scenemind_mvp.pipeline import digest
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('benchmarks/phase26/latest-run.json'));args=p.parse_args()
root=Path('benchmarks/phase26');pointer=json.loads(args.run.read_text());raw=Path(pointer['raw_directory']);processing=json.loads(Path(pointer['processing_file']).read_text());protocol=json.loads((root/'protocol.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
if (raw/'retrieval.json').exists():raise FileExistsError('Raw retrieval results are immutable. Run benchmark_processing.py for a new run before evaluating again.')
if processing['protocol_sha256']!=digest(root/'protocol.json'):raise ValueError('Protocol changed since processing')
if any(digest(Path(path))!=value for path,value in protocol['config_hashes'].items()):raise ValueError('Frozen model/ranking configuration changed')
settings=replace(HybridConfig(**json.loads(Path('configs/hybrid.json').read_text())),top_k=protocol['top_k'],min_gap_seconds=protocol['min_gap_seconds'])
frozen={str(path):digest(path) for path in [root/'protocol.json',root/'manifest.json',*root.glob('*-review.json'),Path('benchmarks/hybrid-real-v1.json'),Path('benchmarks/events-real-v1.json'),Path('benchmarks/missed-backpacks-v1.json'),Path('data/demo/scenemind_demo_ground_truth.json'),*Path('configs').glob('*.json')]}
encoder=Encoder(Config.load(Path('configs/default.json')),Path('data/models'));torch=encoder.torch
encoder.encode_text('warmup query')
if encoder.device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
cohorts=[]
for key in ('demo','real'):
    prep=json.loads(Path(f'docs/hybrid-{key}-preparation.json').read_text());labels=json.loads(Path('data/demo/scenemind_demo_ground_truth.json' if key=='demo' else 'benchmarks/hybrid-real-v1.json').read_text())
    cases=[]
    for i,c in enumerate(labels['queries']):
        intervals=c.get('intervals') if key=='demo' else [[t,t+.001] for t in c['relevant_timestamps']] if c.get('relevant_timestamps') is not None else None
        cases.append({'query_id':f'dev-{i+1}','query':c['query'],'verdict':'positive' if intervals else 'pending','relevant_intervals':intervals,'label_scope':'synthetic authored intervals' if key=='demo' else 'assistant-reviewed visual frame relevance only, not action truth'})
    cohorts.append({'cohort':f'development-{key}','bundle':prep['bundle'],'cases':cases,'candidate_timestamps':labels.get('candidate_timestamps'),'tolerance':protocol['timestamp_tolerance_seconds'] if key=='demo' else 0.,'annotation':None,'video_id':'demo' if key=='demo' else 'sabana_grande'})
for video in manifest['videos']:
    review=json.loads(Path(video['annotation_file']).read_text());signed=validate_review(review,video['sha256'],video['metadata']['duration']);
    if [q['query'] for q in review['queries']]!=protocol['queries']:raise ValueError('Predefined review queries changed')
    perf=next(v for v in processing['videos'] if v['video_id']==video['video_id'])
    cohorts.append({'cohort':'human-'+video['video_id'],'bundle':perf['bundle'],'cases':review['queries'] if signed else [{**q,'verdict':'pending','relevant_intervals':None} for q in review['queries']],'candidate_timestamps':None,'tolerance':protocol['timestamp_tolerance_seconds'],'annotation':review if signed else None,'video_id':video['video_id']})
report={'protocol_sha256':digest(root/'protocol.json'),'input_hashes':frozen,'device':encoder.device,'cohorts':[],'limitations':['Development accuracy is not independent human action accuracy.','Human-pending cases have null accuracy, not zero.','Five repeated warm queries per case, no production concurrency or cold filesystem guarantee.','Structured adapter uses existing event vocabulary; outside-vocabulary queries abstain.','Candidate temporal evidence is not proof of physical carrying/contact.']};csvrows=[]
for cohort in cohorts:
    index,evidence=load_bundle(Path(cohort['bundle']));engine=HybridEngine(index,evidence,settings)
    candidates=cohort['candidate_timestamps'] or index.search_index.timestamps.tolist()
    cases=[]
    for case in cohort['cases']:
        methods={m:{'trials':[]} for m in ('clip_only','structured','hybrid')}
        for repetition in range(protocol['repetitions']):
            start=time.perf_counter();vector=encoder.encode_text(case['query'])
            if encoder.device=='cuda':torch.cuda.synchronize()
            encoding=time.perf_counter()-start
            order=list(methods);order=order[repetition%3:]+order[:repetition%3]
            for mode in order:
                result=structured_search(case['query'],evidence,candidates,settings.top_k,settings.min_gap_seconds) if mode=='structured' else engine.search(case['query'],vector,mode,candidate_timestamps=candidates)
                elapsed=result['ranking_seconds']+(0 if mode=='structured' else encoding)
                row=methods[mode];row['trials'].append({'repetition':repetition,'encoding_seconds':0 if mode=='structured' else encoding,'ranking_seconds':result['ranking_seconds'],'query_seconds':elapsed})
                if repetition==0:row['result']=result
                else:assert [r['timestamp'] for r in row['result']['results']]==[r['timestamp'] for r in result['results']]
        for mode,row in methods.items():
            result=row['result'];timestamps=[r['timestamp'] for r in result['results']];metrics=retrieval_metrics(timestamps,case['relevant_intervals'],case['verdict'],settings.top_k,cohort['tolerance'])
            claims=[r for r in result['results'] if r.get('event_claim')];metrics.update({'event_claims':len(claims),'observed_event_claims':sum(r['event_claim']['status']=='observed' for r in claims),'candidate_event_claims':sum(r['event_claim']['status']=='candidate' for r in claims),'false_positive_event_claims':(len(claims) if case['verdict']=='negative' else sum(not any(a-cohort['tolerance']<=r['timestamp']<b+cohort['tolerance'] for a,b in case['relevant_intervals']) for r in claims)) if cohort['annotation'] is not None and case['verdict'] in ('positive','negative') else None,'explicit_uncertainty':result['status'] in ('uncertain','no_reliable_match'),'median_query_ms':statistics.median(t['query_seconds'] for t in row['trials'])*1000,'median_ranking_ms':statistics.median(t['ranking_seconds'] for t in row['trials'])*1000})
            row['metrics']=metrics;csvrows.append({'cohort':cohort['cohort'],'video_id':cohort['video_id'],'query_id':case['query_id'],'query':case['query'],'method':mode,**metrics})
        cases.append({'case':case,'methods':methods})
    aggregates={}
    for mode in ('clip_only','structured','hybrid'):
        ms=[c['methods'][mode]['metrics'] for c in cases];positives=[m for m in ms if m['hit_at_1'] is not None]
        aggregates[mode]={**{key:statistics.mean(float(m[key]) for m in positives) if positives else None for key in ('hit_at_1','hit_at_5','precision_at_5','mrr')},'positive_queries':len(positives),'queries':len(ms),'median_query_ms':statistics.median(m['median_query_ms'] for m in ms),'median_ranking_ms':statistics.median(m['median_ranking_ms'] for m in ms),'abstention_fraction':statistics.mean(m['abstained'] for m in ms),'explicit_uncertainty_fraction':statistics.mean(m['explicit_uncertainty'] for m in ms),'event_claims':sum(m['event_claims'] for m in ms),'false_positive_event_claims':sum(m['false_positive_event_claims'] for m in ms if m['false_positive_event_claims'] is not None) if any(m['false_positive_event_claims'] is not None for m in ms) else None}
    annotation=cohort['annotation'];component_metrics={'detection':detection_metrics(annotation['detection_frames'] if annotation else [],evidence.frames,protocol['detection_iou_threshold']),'tracking':tracking_metrics(annotation['tracking_sequences'] if annotation else [],evidence.frames,protocol['detection_iou_threshold']),'temporal':temporal_case_metrics(annotation['events'] if annotation else [],evidence.events,protocol['temporal_iou_threshold'])}
    if cohort['cohort']=='development-real':
        bundle=json.loads(Path(cohort['bundle']).read_text());timeline=json.loads((Path(bundle['temporal_directory'])/'timeline.json').read_text());component_metrics['legacy_development_temporal']=evaluate_cases(timeline,json.loads(Path('benchmarks/events-real-v1.json').read_text()))
    report['cohorts'].append({k:v for k,v in cohort.items() if k not in ('cases','annotation')}|{'annotation_signed':annotation is not None,'aggregates':aggregates,'cases':cases,'component_metrics':component_metrics})
    (raw/'retrieval.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(cohort['cohort'],json.dumps(aggregates),flush=True)
report['query_peak_gpu_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2 if encoder.device=='cuda' else None
report['query_peak_gpu_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2 if encoder.device=='cuda' else None
assert all(digest(Path(path))==value for path,value in frozen.items())
report['inputs_unchanged']=True
(raw/'retrieval.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
with (raw/'query-metrics.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
print('REPORT',raw/'retrieval.json')
