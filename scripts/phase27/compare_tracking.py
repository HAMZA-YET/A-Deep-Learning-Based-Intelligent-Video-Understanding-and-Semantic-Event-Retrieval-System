"""Replay existing class-separated ByteTrack on identical raw detections; never merge IDs."""
from pathlib import Path
from dataclasses import replace
import json,time,csv
from collections import Counter
import numpy as np,cv2,torch
from scenemind_mvp.detection.schema import DetectionConfig,Observation
from scenemind_mvp.detection.tracker import ClassTrackers
from scenemind_mvp.detection.detector import Detector
from scenemind_mvp.association.diagnostics import diagnose
from scenemind_mvp.association.core import associate,AssociationConfig
from scenemind_mvp.temporal.core import reason_events,TemporalConfig
from scenemind_mvp.temporal.evaluation import evaluate_cases
from scenemind_mvp.pipeline import digest
root=Path('benchmarks/phase27');pointer=json.loads((root/'latest-run.json').read_text());raw=Path(pointer['raw_directory']);protocol=json.loads((root/'protocol.json').read_text());video=next(v for v in json.loads(Path('benchmarks/phase26/manifest.json').read_text())['videos'] if v['video_id']=='sabana_grande');meta=video['metadata'];source=Path(video['path']);assert digest(source)==video['sha256']
report={'source_sha256':video['sha256'],'interpretation':'Output tracklets and coverage, not verified identity continuity or ID-switch accuracy. ByteTrack IDs never merged.','experiments':[]}
if (raw/'tracking-comparison.csv').exists():raise FileExistsError('Completed tracking results are preserved; start a new detector run for another experiment.')
rows=[]
for detector_name in protocol['full_video_inference_profiles']:
    d=dict(protocol['detectors'][detector_name]);d['classes']=tuple(d['classes']);detconfig=DetectionConfig(**d);directory=raw/('full-'+detector_name);directory.mkdir(exist_ok=True);candidate_path=directory/'candidates.json'
    if candidate_path.exists():candidate_data=json.loads(candidate_path.read_text());assert candidate_data['source_sha256']==video['sha256'] and candidate_data['config']==detconfig.to_dict()
    else:
        detector=Detector(detconfig,Path('data/models/yolo'));capture=cv2.VideoCapture(str(source));frames=[];index=0;inference=0.;started=time.perf_counter()
        while True:
            ok,image=capture.read()
            if not ok:break
            if torch.cuda.is_available():torch.cuda.synchronize()
            start=time.perf_counter();candidates=detector.infer(image)
            if torch.cuda.is_available():torch.cuda.synchronize()
            inference+=time.perf_counter()-start;frames.append({'frame_index':index,'timestamp':index/meta['fps'],'candidates':candidates.tolist()});index+=1
        capture.release();assert index==meta['frame_count']
        candidate_data={'source_sha256':video['sha256'],'config':detconfig.to_dict(),'class_names':{str(k):v for k,v in detector.names.items()},'frames':frames,'inference_seconds':inference,'decode_and_inference_seconds':time.perf_counter()-started};candidate_path.write_text(json.dumps(candidate_data),encoding='utf-8');del detector
    for profile_path in protocol['tracker_profiles']:
        profile=json.loads(Path(profile_path).read_text());settings=replace(detconfig,**{key:profile[key] for key in ('low_threshold','high_threshold','new_track_threshold','lost_seconds','match_threshold')});name=Path(profile_path).stem;tracker=ClassTrackers(settings,meta['fps']);frames=[];start=time.perf_counter()
        for item in candidate_data['frames']:
            candidates=np.asarray(item['candidates'],dtype=np.float32).reshape(-1,6);assignments,states=tracker.update(candidates,(meta['height'],meta['width']));observations=[]
            for i,r in enumerate(candidates):
                if r[4]<settings.confidence:continue
                box=np.clip(r[:4],[0,0,0,0],[meta['width'],meta['height'],meta['width'],meta['height']])
                if box[2]<=box[0] or box[3]<=box[1]:continue
                cls=int(r[5]);observations.append(Observation(cls,candidate_data['class_names'][str(cls)],float(r[4]),tuple(float(v) for v in box),assignments.get(i)).to_dict())
            frames.append({'frame_index':item['frame_index'],'timestamp':item['timestamp'],'detections':observations,'tracker_state':states})
        replay=time.perf_counter()-start;quality=diagnose(frames,meta['fps']);ac=AssociationConfig(**json.loads(Path('configs/association.json').read_text()));tc=TemporalConfig(**json.loads(Path('configs/temporal.json').read_text()));start=time.perf_counter();associations=associate(frames,quality,ac);timeline=reason_events(frames,associations,quality,tc);reason_seconds=time.perf_counter()-start;timeline['source_sha256']=video['sha256'];evaluation=evaluate_cases(timeline,json.loads(Path('benchmarks/events-real-v1.json').read_text()))
        target=directory/name;target.mkdir(exist_ok=True)
        for filename,value in [('records.json',{'frames':frames}),('diagnostics.json',quality),('associations.json',associations),('timeline.json',timeline),('legacy-event-evaluation.json',evaluation)]: (target/filename).write_text(json.dumps(value,indent=2),encoding='utf-8')
        bags=[t for t in quality['tracks'] if t['class_id'] in (24,26,28)];counts=Counter(e['status'] for e in timeline['events']);q={key:value for key,value in quality.items() if key!='tracks'}
        summary={'detector':detector_name,'tracker':name,'directory':str(target),'config':settings.to_dict(),'inference_seconds':candidate_data['inference_seconds'],'tracking_replay_seconds':replay,'association_temporal_seconds':reason_seconds,'person_tracklets':q['person_tracklets'],'median_person_span_seconds':q['median_person_span_seconds'],'median_person_coverage':q['median_person_coverage'],'person_tracklets_with_gaps':q['person_tracklets_with_gaps'],'short_person_tracklets_under_half_second':q['short_person_tracklets_under_half_second'],'bag_tracklets':len(bags),'bag_observations':sum(q['detections_by_class'].get(label,0) for label in ('backpack','handbag','suitcase')),'median_bag_span_seconds':float(np.median([b['span_seconds'] for b in bags])) if bags else None,'temporal_status_counts':dict(counts),'legacy_event_tp':evaluation['true_positives'],'legacy_event_fn':evaluation['false_negatives'],'legacy_event_fp':evaluation['false_positives'],'legacy_event_tn':evaluation['true_negatives'],'identity_switches':None,'false_associations':None}
        report['experiments'].append(summary);rows.append({key:value for key,value in summary.items() if not isinstance(value,dict)})
        (raw/'tracking.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
        if detector_name=='baseline640' and name=='detection':
            assert q['person_tracklets']==101 and q['detections_by_class'].get('backpack')==22 and q['detections_by_class'].get('handbag')==1,'Baseline replay disagrees with preserved run'
with (raw/'tracking-comparison.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
