"""Model-only experiment: identical decoded development frames, no changed ground truth."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
import json,time,csv,gc
from collections import Counter
import cv2,numpy as np,torch
from scenemind_mvp.detection.detector import Detector
from scenemind_mvp.detection.schema import DetectionConfig
from scenemind_mvp.pipeline import digest
from scenemind_mvp.association.core import box_iou
root=Path('benchmarks/phase27');protocol=json.loads((root/'protocol.json').read_text());manifest=json.loads(Path('benchmarks/phase26/manifest.json').read_text());video=next(v for v in manifest['videos'] if v['video_id']=='sabana_grande');source=Path(video['path']);assert digest(source)==video['sha256']
bench=json.loads(Path('benchmarks/missed-backpacks-v1.json').read_text());assert digest(source)==bench['source_sha256']
run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');raw=root/'raw'/run_id;raw.mkdir(parents=True)
indices=sorted(set(range(0,video['metadata']['frame_count'],round(video['metadata']['fps'])))|{c['frame_index'] for c in bench['cases']});images={};capture=cv2.VideoCapture(str(source))
for index in indices:
    capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=capture.read()
    if not ok:raise RuntimeError('Failed exact development frame decode')
    images[index]=image
capture.release()
def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()
report={'run_id':run_id,'protocol_sha256':digest(root/'protocol.json'),'source_sha256':video['sha256'],'frame_indices':indices,'label_scope':'Only five approximate developer-reviewed positive backpack boxes; all other predictions unverified. False positives and precision are null.','profiles':{},'gpu_cleanup':'Python garbage collection and empty_cache before loading each profile; no previous model intentionally remains resident'}
rows=[]
for name,config in protocol['detectors'].items():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    d=dict(config);d['classes']=tuple(d['classes']);settings=DetectionConfig(**d);start=time.perf_counter();detector=Detector(settings,Path('data/models/yolo'));sync();load=time.perf_counter()-start
    assert detector.names[24]=='backpack' and detector.names[26]=='handbag'
    for _ in range(protocol['warmup_per_profile']):detector.infer(images[indices[0]])
    sync()
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    predictions={};trials=[]
    for index,image in images.items():
        for repeat in range(protocol['repetitions']):
            sync();start=time.perf_counter();candidates=detector.infer(image);sync();seconds=time.perf_counter()-start;trials.append({'frame_index':index,'repetition':repeat,'inference_seconds':seconds})
            if repeat==0:predictions[str(index)]={'raw_candidates':candidates.tolist(),'visible_predictions':candidates[candidates[:,4]>=settings.confidence].tolist()}
    spots=[]
    for case in bench['cases']:
        predicted=predictions[str(case['frame_index'])]['visible_predictions'];scores=[(box_iou(case['bbox_xyxy'],r[:4]),r) for r in predicted if int(r[5])==case['class_id']];best=max(scores,key=lambda pair:pair[0],default=(0.,None));spots.append({'frame_index':case['frame_index'],'timestamp':case['timestamp'],'gt_box':case['bbox_xyxy'],'matched':best[0]>=bench['matching_iou'],'best_class_iou':best[0],'best_prediction':best[1]})
    latencies=np.array([t['inference_seconds'] for t in trials]);counts=Counter(detector.names[int(r[5])] for p in predictions.values() for r in p['visible_predictions'])
    summary={'config':config,'model_load_seconds':load,'class_mapping':{str(k):detector.names[k] for k in (0,24,26,28,56)},'median_inference_ms':float(np.median(latencies)*1000),'p95_inference_ms':float(np.percentile(latencies,95)*1000),'pooled_inference_fps':len(trials)/float(latencies.sum()),'matched_reviewed_boxes':sum(s['matched'] for s in spots),'missed_reviewed_boxes':sum(not s['matched'] for s in spots),'false_positives':None,'precision':None,'unverified_predictions_by_class':dict(counts),'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/1024**2 if torch.cuda.is_available() else None}
    report['profiles'][name]={'summary':summary,'trials':trials,'spot_check':spots,'predictions':predictions};rows.append({'profile':name,**{k:v for k,v in summary.items() if not isinstance(v,dict)}})
    # Full source-frame overlays: green developer GT, yellow predictions; these are NOT the human review interface.
    directory=raw/'visuals'/name;directory.mkdir(parents=True)
    for case in bench['cases']:
        image=images[case['frame_index']].copy();x1,y1,x2,y2=map(int,case['bbox_xyxy']);cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2);cv2.putText(image,'reviewed backpack box',(x1,max(15,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
        for r in predictions[str(case['frame_index'])]['visible_predictions']:
            if int(r[5]) not in (24,26,28):continue
            a,b,c,d=map(round,r[:4]);cv2.rectangle(image,(a,b),(c,d),(0,210,255),2);cv2.putText(image,f'{detector.names[int(r[5])]} {r[4]:.3f}',(a,max(15,b-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,210,255),1)
        cv2.putText(image,f'{name} frame={case["frame_index"]}: green=legacy review yellow=model bags; unlabelled predictions not FP',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        cv2.imwrite(str(directory/f'{case["frame_index"]}.jpg'),image)
    (raw/'detectors.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(name,json.dumps(summary),flush=True)
    detector.model.to('cpu')
    del detector
    gc.collect()
with (raw/'detector-comparison.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert all(digest(Path(p))==sha for p,sha in protocol['inputs'].items())
(root/'latest-run.json').write_text(json.dumps({'run_id':run_id,'raw_directory':str(raw)},indent=2),encoding='utf-8')
print('REPORT',raw/'detectors.json')
