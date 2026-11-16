"""Reproduce actual localization, latency, GPU and selected-target video checks."""
from pathlib import Path
import json,time,csv
import torch
from PIL import Image,ImageDraw
from scenemind_mvp.ui.catalog import catalog
from scenemind_mvp.ui.service import search,encoder_for
from scenemind_mvp.highlighting.render import thumbnail,highlighted_video
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'data';OUT=ROOT/'docs/highlighting';OUT.mkdir(exist_ok=True)
QUERIES=['A person walking.','A person wearing a white shirt.','A person wearing a black shirt.','A person carrying a black bag.','Two people walking together.']
SETTINGS={'sampling_fps':1.,'batch_size':16,'model':'ViT-B-32','pretrained':'openai','device':'auto','max_results':5,'gap':1.,'cutoff':-1.,'strict':False,'padding':.2,'detector_sizes':[640,960],'detection_confidence':.1,'evidence_profile':'Baseline','full_hybrid':False}

def main():
    torch.cuda.synchronize();x=torch.ones((100,100),device='cuda');y=x@x;torch.cuda.synchronize()
    gpu={'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'matmul_shape':list(y.shape),'matmul_value':y[0,0].item()}
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();encoder_for(SETTINGS,DATA);torch.cuda.synchronize();model_seconds=time.perf_counter()-start
    videos=catalog(DATA,[]);runs=[];flat=[];images=[]
    sources=[('mixkit_walker','011209'),('mixkit_students','c242ea'),('caracas','118516'),('mixkit_lake_scene_only','31f674')]
    for name,prefix in sources:
        video=next(v for v in videos if v['id'].startswith(prefix))
        for query in QUERIES:
            mode='Regions' if video['regions'] else 'CLIP-only';torch.cuda.synchronize();start=time.perf_counter()
            result=search(video,mode,query,SETTINGS,ROOT,DATA);torch.cuda.synchronize();seconds=time.perf_counter()-start
            entry={'video':name,'query':query,'mode':mode,'search_seconds':seconds,'result':result};runs.append(entry)
            row=result['results'][0] if result['results'] else None
            targets=[t for t in row['localization']['candidates'] if t['id'] in row['localization']['primary_target_ids']] if row else []
            rendered=None
            if targets:
                path=thumbnail(Path(video['source']),row['localization'],targets,DATA/'highlights');entry['thumbnail']=str(path)
                image=Image.open(path).convert('RGB');images.append((name+' / '+query,image))
                rendered=highlighted_video(Path(video['source']),row['localization'],targets,DATA/'highlights');entry['render']=rendered
                cached=highlighted_video(Path(video['source']),row['localization'],targets,DATA/'highlights');assert cached['cache_hit'];entry['cached_render_seconds']=cached['request_seconds']
            flat.append({'video':name,'query':query,'timestamp':row['timestamp'] if row else None,'full_frame_cosine':row['clip_cosine'] if row else None,'target_labels':' + '.join(t['label'] for t in targets),'region_cosines':';'.join(str(t['region_cosine']) for t in targets),'search_seconds':seconds,'highlighted_frames':rendered['manifest']['highlighted_frames'] if rendered else 0,'render_seconds':rendered['request_seconds'] if rendered else None,'cache_hit':rendered['cache_hit'] if rendered else None,'verified_query_match':False})
            print(name,query,flat[-1],flush=True)
            (OUT/'raw-results.json').write_text(json.dumps({'gpu':gpu,'model_load_seconds':model_seconds,'runs':runs},indent=2),encoding='utf-8')
    report={'gpu':gpu,'model_load_seconds':model_seconds,'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'peak_gpu_reserved_mb':torch.cuda.max_memory_reserved()/2**20,'rows':flat,'review_status':'Unreviewed model outputs; no localization/clothing/action accuracy claimed.'}
    (OUT/'measurements.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUT/'measurements.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    sheet=Image.new('RGB',(640,220*((len(images)+1)//2)),'white');draw=ImageDraw.Draw(sheet)
    for i,(label,image) in enumerate(images):
        x=(i%2)*320;y=(i//2)*220;sheet.paste(image,(x,y+35));draw.text((x+3,y+3),label,fill='black')
    sheet.save(OUT/'target-review.jpg')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
