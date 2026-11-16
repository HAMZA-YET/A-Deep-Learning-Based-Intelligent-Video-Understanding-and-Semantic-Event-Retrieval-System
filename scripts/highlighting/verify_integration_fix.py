"""Actual cache-miss reproduction and fixed Mixkit localization/rendering."""
from pathlib import Path
from copy import deepcopy
import json,time,types,sys
import torch,cv2
from PIL import Image,ImageDraw
from scenemind_mvp.ui.catalog import catalog,region_choice
from scenemind_mvp.ui.service import search
from scenemind_mvp.highlighting.render import highlighted_video,thumbnail,draw_frame
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/highlighting-fix';DATA=ROOT/'data'
SETTINGS={'sampling_fps':1.,'batch_size':16,'model':'ViT-B-32','pretrained':'openai','device':'auto','max_results':5,'gap':1.,'cutoff':-1.,'strict':False,'padding':.2,'detector_sizes':[640,960],'detection_confidence':.1,'evidence_profile':'Baseline','full_hybrid':False}
QUERIES=['a person walking','a person wearing a white shirt','a person wearing a black shirt']
def main():
    x=torch.ones((100,100),device='cuda');y=x@x;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    vs=catalog(DATA,[]);walker=next(v for v in vs if v['id'].startswith('011209'))
    # Execute the archived code before the fix against a real unmatched ROI configuration.
    old=types.ModuleType('scenemind_mvp.ui._beforefix');sys.modules[old.__name__]=old
    path=OUT/'before-service.py';exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),old.__dict__)
    missing={**SETTINGS,'padding':.25};before=old.search({**walker,'regions':[]},'CLIP-only',QUERIES[0],missing,ROOT,DATA)
    assert not any(r['localization']['candidates'] for r in before['results'])
    (OUT/'reproduced-cache-miss.json').write_text(json.dumps(before,indent=2))
    start=time.perf_counter();after=search({**walker,'regions':[]},'CLIP-only',QUERIES[0],missing,ROOT,DATA)
    assert after['results'][0]['localization']['primary_target_ids']
    runs=[{'name':'walker-cache-miss-fixed','mode':'CLIP-only','query':QUERIES[0],'seconds':time.perf_counter()-start,'result':after}];pictures=[]
    for prefix,name in [('011209','mixkit-walker'),('c242ea','mixkit-students')]:
        video=next(v for v in vs if v['id'].startswith(prefix))
        for mode in ['CLIP-only','Regions','Regions + Evidence']:
            for q in QUERIES:
                video=next(v for v in catalog(DATA,[]) if v['id'].startswith(prefix))
                start=time.perf_counter();result=search(video,mode,q,SETTINGS,ROOT,DATA);torch.cuda.synchronize()
                entry={'name':name,'mode':mode,'query':q,'seconds':time.perf_counter()-start,'result':result};runs.append(entry)
                row=result['results'][0];loc=row['localization'];targets=[c for c in loc['candidates'] if c['id'] in loc['primary_target_ids']]
                if mode=='CLIP-only' and targets:
                    entry['thumbnail']=str(thumbnail(Path(video['source']),loc,targets,DATA/'highlights'))
                    entry['render']=highlighted_video(Path(video['source']),loc,targets,DATA/'highlights')
                    again=highlighted_video(Path(video['source']),loc,targets,DATA/'highlights');assert again['cache_hit']
                    cap=cv2.VideoCapture(entry['render']['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,targets[0]['source_frame_index']);ok,frame=cap.read();cap.release();assert ok
                    annotated=OUT/(name+'-'+q.replace(' ','-')+'.jpg');Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(annotated);entry['annotated_frame']=str(annotated)
                    pictures.append((name+' / '+q,Image.open(annotated).copy()))
                print(name,mode,q,row['timestamp'],[(t['label'],t['bbox_xyxy'],t['highlight_continuity']) for t in targets],flush=True)
                (OUT/'actual-results.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(0),'cuda_value':y[0,0].item(),'runs':runs,'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/2**20},indent=2))
    sheet=Image.new('RGB',(960,580*len(pictures)),'white');d=ImageDraw.Draw(sheet)
    for i,(text,im) in enumerate(pictures):
        im.thumbnail((960,540));sheet.paste(im,(0,i*580+30));d.text((6,i*580+5),text,fill='black')
    sheet.save(OUT/'annotated-review.jpg')
if __name__=='__main__':main()
