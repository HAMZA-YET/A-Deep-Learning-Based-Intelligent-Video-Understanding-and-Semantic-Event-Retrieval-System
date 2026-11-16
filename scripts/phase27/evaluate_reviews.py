"""Evaluate only named, source-independent human labels; absence is pending, not failure."""
from pathlib import Path
from dataclasses import replace
import argparse,json,time,csv,hashlib
from scenemind_mvp.pipeline import digest
from scenemind_mvp.hybrid.evidence import load_bundle,Evidence
from scenemind_mvp.hybrid.engine import HybridEngine
from scenemind_mvp.hybrid.schema import HybridConfig
from scenemind_mvp.evaluation.retrieval import structured_search
from scenemind_mvp.evaluation.metrics import validate_review,detection_metrics,tracking_metrics,temporal_case_metrics,retrieval_metrics
from scenemind_mvp.config import Config
from scenemind_mvp.model import Encoder
p=argparse.ArgumentParser();p.add_argument('--review-dir',type=Path,default=Path('benchmarks/phase27/human-reviewed'));args=p.parse_args()
root=Path('benchmarks/phase27');raw=Path(json.loads((root/'latest-run.json').read_text())['raw_directory']);manifest=json.loads(Path('benchmarks/phase26/manifest.json').read_text());protocol=json.loads(Path('benchmarks/phase26/protocol.json').read_text());processing=json.loads(Path(json.loads(Path('benchmarks/phase26/latest-run.json').read_text())['processing_file']).read_text());experiments=json.loads((raw/'tracking.json').read_text())['experiments']
if any(digest(Path(path))!=value for path,value in protocol['config_hashes'].items()):raise ValueError('Existing configurations changed')
settings=replace(HybridConfig(**json.loads(Path('configs/hybrid.json').read_text())),top_k=5,min_gap_seconds=1.)
report={'review_directory':str(args.review_dir.resolve()),'label_policy':'Only explicitly signed independent human annotations. New clips remain baseline-only, closed to experimental tuning. Caracas is development footage even if independently relabelled.','videos':[],'pending':[]};rows=[];encoder=None
for video in manifest['videos']:
    label_path=args.review_dir/f'{video["video_id"]}-review.json'
    if not label_path.exists():report['pending'].append({'video_id':video['video_id'],'reason':'No independently signed human review supplied'});continue
    label_bytes=label_path.read_bytes();review=json.loads(label_bytes)
    if not validate_review(review,video['sha256'],video['metadata']['duration']):report['pending'].append({'video_id':video['video_id'],'reason':'Review is not signed complete'});continue
    if [q['query'] for q in review['queries']]!=protocol['queries']:raise ValueError('Human query text differs from frozen queries')
    for event in review['events']:
        if event['verdict'] not in ('positive','negative','ambiguous') or not 0<=event['start_timestamp']<event['end_timestamp']<=video['metadata']['duration']:raise ValueError('Invalid signed event annotation')
    baseline=next(v for v in processing['videos'] if v['video_id']==video['video_id']);index,evidence=load_bundle(Path(baseline['bundle']));variants={'preserved_baseline':evidence}
    if video['video_id']=='sabana_grande':
        for experiment in experiments:
            directory=Path(experiment['directory']);frames=json.loads((directory/'records.json').read_text())['frames'];quality=json.loads((directory/'diagnostics.json').read_text());associations=json.loads((directory/'associations.json').read_text())['events'];events=json.loads((directory/'timeline.json').read_text())['events']
            variants[experiment['detector']+'/'+experiment['tracker']]=Evidence(frames,associations,events,{t['track_id']:t for t in quality['tracks']},video['sha256'],{'experiment_directory':str(directory)})
    item={'video_id':video['video_id'],'reviewer':review['reviewer'],'label_sha256':hashlib.sha256(label_bytes).hexdigest(),'variants':{}}
    if encoder is None:encoder=Encoder(Config.load(Path('configs/default.json')),Path('data/models'))
    vectors={}
    for case in review['queries']:
        started=time.perf_counter();vector=encoder.encode_text(case['query'])
        if encoder.device=='cuda':encoder.torch.cuda.synchronize()
        vectors[case['query']]=(vector,time.perf_counter()-started)
    for name,ev in variants.items():
        engine=HybridEngine(index,ev,settings);results=[]
        for case in review['queries']:
            vector,encoding=vectors[case['query']];methods={}
            for method in ('clip_only','structured','hybrid'):
                response=structured_search(case['query'],ev,index.search_index.timestamps.tolist()) if method=='structured' else engine.search(case['query'],vector,method)
                metrics=retrieval_metrics([r['timestamp'] for r in response['results']],case['relevant_intervals'],case['verdict'],5,.5)
                claims=[r for r in response['results'] if r.get('event_claim')];metrics.update(event_claims=len(claims),query_seconds=response['ranking_seconds']+(0 if method=='structured' else encoding),false_positive_event_claims=(len(claims) if case['verdict']=='negative' else sum(not any(a-.5<=r['timestamp']<b+.5 for a,b in case['relevant_intervals']) for r in claims)) if case['verdict']!='ambiguous' else None)
                methods[method]={'metrics':metrics,'result':response};rows.append({'video_id':video['video_id'],'variant':name,'query':case['query'],'method':method,**metrics})
            results.append({'case':case,'methods':methods})
        item['variants'][name]={'detection':detection_metrics(review['detection_frames'],ev.frames,.5),'tracking':tracking_metrics(review['tracking_sequences'],ev.frames,.5),'events':temporal_case_metrics(review['events'],ev.events,.3),'relationship_cases':temporal_case_metrics(review.get('relationships',[]),[{**a,'action':a['relationship'],'status':'unknown' if a['relationship']=='unknown' else 'candidate'} for a in ev.associations],.3),'retrieval':results}
    assert label_bytes==label_path.read_bytes();report['videos'].append(item)
report['status']='pending_human_review' if not report['videos'] else ('partial_human_review' if report['pending'] else 'human_labels_evaluated')
report['limitations']=['Manual identity labels are needed for ID-switch metrics.','Human labelling independence does not make development footage held out.','Case-level event matching is action/object/time scoped, not proof of contact or ownership.']
# Retain every evaluation invocation rather than overwriting observations.
from datetime import datetime,timezone
out=raw/'human-evaluations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if rows:
    with (out/'retrieval.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(root/'latest-human-evaluation.json').write_text(json.dumps({'results':str(out/'results.json'),'status':report['status']},indent=2),encoding='utf-8');print(json.dumps({'status':report['status'],'evaluated_videos':len(report['videos']),'pending':report['pending'],'output':str(out/'results.json')},indent=2))
