"""Actual region/full-frame comparisons; unsigned accuracy remains unmeasurable."""
from pathlib import Path
from dataclasses import replace
import json,time,gc,csv
import numpy as np
from PIL import Image
import cv2
from scenemind_mvp.config import Config
from scenemind_mvp.model import Encoder
from scenemind_mvp.pipeline import digest
from scenemind_mvp.hybrid.evidence import load_bundle
from scenemind_mvp.hybrid.schema import HybridConfig
from scenemind_mvp.regions.schema import RegionConfig
from scenemind_mvp.regions.pipeline import build_regions
from scenemind_mvp.regions.engine import RegionEngine
from scenemind_mvp.regions.crops import padded_box,square_view,center_crop_source_fov
from scenemind_mvp.association.core import box_iou
from scenemind_mvp.evaluation.metrics import retrieval_metrics
root=Path('benchmarks/regions')
if (root/'comparison.json').exists():
    from datetime import datetime,timezone
    import shutil
    shutil.copy2(root/'comparison.json',root/('comparison-previous-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'))
protocol=json.loads((root/'protocol.json').read_text());weights={str(p):digest(p) for p in [Path('configs/hybrid.json'),*(Path(p) for p in protocol['region_configs']),Path('benchmarks/missed-backpacks-v1.json'),Path('benchmarks/hybrid-real-v1.json')]}
encoder=Encoder(Config.load(Path('configs/default.json')),Path('data/models'));torch=encoder.torch
bundles={'sabana_grande':Path(json.loads(Path('docs/hybrid-real-preparation.json').read_text())['bundle'])}
processing=json.loads(Path(json.loads(Path('benchmarks/phase26/latest-run.json').read_text())['processing_file']).read_text())
for row in processing['videos']:
    if row['video_id']!='sabana_grande':bundles[row['video_id']]=Path(row['bundle'])
queries=['Show a visible backpack.','Show a small person in the background.','Show a large person in the foreground.','Show a black bag.','Show a person walking.','Show a person running.','Find all people near a bag.','Find a person picking up a bag.','Show someone leaving with a backpack.']
report={'config_hashes':weights,'device':encoder.device,'label_policy':'Predicted region areas are scale proxies, not true foreground/background identities. New human accuracy remains pending; legacy visual relevance is development only.','experiments':[]};timingrows=[];cached={}
for video_id,bundle in bundles.items():
    index,evidence=load_bundle(bundle)
    profiles=protocol['region_configs'] if video_id=='sabana_grande' else ['configs/regions.json']
    for config_path in profiles:
        config_data=json.loads(Path(config_path).read_text(encoding='utf-8-sig'));config_data['detector_sizes']=tuple(config_data['detector_sizes']);config=RegionConfig(**config_data)
        gc.collect()
        if encoder.device=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();regions,hit=build_regions(bundle,Path('data/regions'),config,encoder)
        if encoder.device=='cuda':torch.cuda.synchronize()
        index_wall=time.perf_counter()-started;allocated=torch.cuda.max_memory_allocated()/1024**2 if encoder.device=='cuda' else None;reserved=torch.cuda.max_memory_reserved()/1024**2 if encoder.device=='cuda' else None
        engine=RegionEngine(regions,evidence,config,HybridConfig(**json.loads(Path('configs/hybrid.json').read_text())));cases=[]
        if encoder.device=='cuda':torch.cuda.reset_peak_memory_stats()
        for text in queries:
            methods={mode:{'trials':[]} for mode in ('clip_only','region','region_hybrid')}
            encoder.encode_text(text) # warm query; not timed
            for repetition in range(protocol['repetitions']):
                started=time.perf_counter();vector=encoder.encode_text(text)
                if encoder.device=='cuda':torch.cuda.synchronize()
                encoding=time.perf_counter()-started;order=list(methods);order=order[repetition%3:]+order[:repetition%3]
                for mode in order:
                    result=engine.search(text,vector,mode,strict_actions=False);latency=encoding+result['ranking_seconds'];methods[mode]['trials'].append({'encoding_seconds':encoding,'ranking_seconds':result['ranking_seconds'],'query_seconds':latency})
                    if repetition==0:methods[mode]['result']=result
            for mode,m in methods.items():
                m['median_query_ms']=float(np.median([t['query_seconds'] for t in m['trials']])*1000);timingrows.append({'video':video_id,'config':config_path,'query':text,'mode':mode,'median_query_ms':m['median_query_ms'],'first_timestamp':m['result']['results'][0]['timestamp'] if m['result']['results'] else None,'event_claims':sum(r['event_claim'] is not None for r in m['result']['results'])})
            strict={q:engine.search(q,encoder.encode_text(q),strict_actions=True) for q in ('Show a person walking.','Show a person running.','Find a person picking up a bag.')} if text==queries[0] else None
            cases.append({'query':text,'methods':methods,'strict_action_checks':strict})
        item={'video_id':video_id,'config_path':config_path,'region_directory':str(regions.directory.resolve()),'cache_hit':hit,'index_wall_seconds':index_wall,'index_manifest':regions.manifest,'peak_gpu_allocated_mib':allocated,'peak_gpu_reserved_mib':reserved,'cases':cases}
        item['query_peak_gpu_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2 if encoder.device=='cuda' else None
        item['query_peak_gpu_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2 if encoder.device=='cuda' else None
        if video_id=='sabana_grande':
            legacy=json.loads(Path('benchmarks/hybrid-real-v1.json').read_text());development=[]
            for label in legacy['queries']:
                if label.get('relevant_timestamps') is None:continue
                v=encoder.encode_text(label['query']);methods={}
                for method in ('clip_only','region','region_hybrid'):
                    response=engine.search(label['query'],v,method,strict_actions=False,candidate_timestamps=legacy['candidate_timestamps'])
                    metrics=retrieval_metrics([r['timestamp'] for r in response['results']],[[t,t+.001] for t in label['relevant_timestamps']],'positive',5,0.)
                    methods[method]={'metrics':metrics,'result':response}
                development.append({'query':label['query'],'methods':methods})
            item['legacy_visual_development_comparison']=development
            truth=json.loads(Path('benchmarks/missed-backpacks-v1.json').read_text());coverage=[]
            for c in truth['cases']:
                same=[r for r in regions.records if r['source_frame_index']==c['frame_index']];det=[r for r in same if r['class_id']==24];iou=max((box_iou(c['bbox_xyxy'],r['bbox_xyxy']) for r in det),default=0.)
                coverage.append({'timestamp':c['timestamp'],'sampled':bool(same),'detected_backpack_match':iou>=.3,'best_class_iou':iou,'context_regions':sum(r['kind']=='context' for r in same)})
            item['legacy_spot_coverage']=coverage
        report['experiments'].append(item);cached[(video_id,config.sampling_fps,tuple(config.detector_sizes))]=regions
        (root/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(video_id,config_path,regions.manifest['region_count'],index_wall,allocated,flush=True)
# Diagnostic only: the old reviewed boxes are NEVER production proposals.
truth=json.loads(Path('benchmarks/missed-backpacks-v1.json').read_text());source=Path(json.loads(bundles['sabana_grande'].read_text())['source']);cap=cv2.VideoCapture(str(source));oracle=[];vector=encoder.encode_text('a backpack')[0]
for c in truth['cases']:
    cap.set(cv2.CAP_PROP_POS_FRAMES,c['frame_index']);ok,frame=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));box=padded_box(c['bbox_xyxy'],im.width,im.height,.2,48);crop=im.crop(box);low=im.resize((224,126),Image.Resampling.BILINEAR);scaled=[int(box[0]*224/im.width),int(box[1]*126/im.height),max(1,int(box[2]*224/im.width)),max(1,int(box[3]*126/im.height))];lowcrop=low.crop(scaled)
    embeddings=encoder.encode_images([im,square_view(crop),square_view(lowcrop),square_view(im)]);scores=(embeddings@vector).tolist();oracle.append({'timestamp':c['timestamp'],'source_box':c['bbox_xyxy'],'native_crop_pixels':list(crop.size),'downscaled_crop_pixels':list(lowcrop.size),'full_frame_cosine':scores[0],'native_oracle_crop_cosine':scores[1],'downscaled_oracle_crop_cosine':scores[2],'letterboxed_full_frame_cosine':scores[3],'scope':'Ground-truth-box oracle diagnostic ONLY; not production retrieval or action accuracy'})
cap.release()
resize=next(t for t in encoder.preprocess.transforms if t.__class__.__name__=='Resize');center=next(t for t in encoder.preprocess.transforms if t.__class__.__name__=='CenterCrop')
for row in oracle:
    resized=resize(Image.new('RGB',(960,540)));size=center.size if isinstance(center.size,(list,tuple)) else (center.size,center.size);fov=center_crop_source_fov((960,540),resized.size,(size[1],size[0]));a,b,c,d=row['source_box'];intersection=max(0,min(c,fov[2])-max(a,fov[0]))*max(0,min(d,fov[3])-max(b,fov[1]));row['actual_clip_source_fov']=fov;row['bbox_fraction_in_actual_clip_fov']=intersection/((c-a)*(d-b))
report['resolution_oracle_diagnostic']=oracle
report['actual_clip_preprocess']=repr(encoder.preprocess)
report['inputs_unchanged']=all(digest(Path(p))==h for p,h in weights.items());assert report['inputs_unchanged']
(root/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
with (root/'query-times.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(timingrows[0]));w.writeheader();w.writerows(timingrows)
print('Saved benchmarks/regions/comparison.json')
