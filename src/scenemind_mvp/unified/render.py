"""One synchronized CFR output with all observed targets and evidence-gated labels."""
from __future__ import annotations
from pathlib import Path
import json,time,hashlib,tempfile,subprocess
import cv2
from ..pipeline import digest
from ..detection.pipeline import find_ffmpeg,validated_timestamp
from ..highlighting.render import project_box
from .core import matched_indices,action_at,label,make_plan
from .cache import cache_lock,publish

DEFAULTS={'view':'Show All','boxes':True,'ids':True,'actions':True,'objects':True,'hide_nonmatches':False,'max_edge':960}

def draw(image,frame: dict,analysis: dict,plan: dict,options: dict):
    h,w=image.shape[:2];source=(analysis['metadata']['width'],analysis['metadata']['height']);matched=set(matched_indices(frame,plan,analysis));people=objects=0;shown=[]
    for i,detection in enumerate(frame['detections']):
        if detection['class_id']==0:people+=1
        else:objects+=1
        active=i in matched
        if options['view']=='Search Matches' and not active and options['hide_nonmatches']:continue
        color=(40,200,90) if detection['class_id']==0 else (30,190,245)
        if options['view']=='Search Matches' and not active:color=(150,145,140)
        if options['view']=='Search Matches' and active and plan.get('colors'):color=(30,180,245)
        box=project_box(detection['bbox_xyxy'],source,(w,h));x1,y1,x2,y2=box
        if options['boxes']:cv2.rectangle(image,(x1,y1),(x2,y2),color,2 if active else 1)
        text=label(detection,options['ids'],options['objects'])
        if detection['class_id']==0 and options['actions']:
            event=action_at(analysis['actions']['intervals'],detection.get('track_id'),frame['timestamp'])
            action=(event['action'].title()+' (model)') if event else 'Action unknown'
            text=(text+' | '+action).strip(' |')
        if active and plan.get('colors') and detection['class_id']==0:text+=' | '+plan['colors'][0].title()+' cue (uncertain)'
        if text:
            scale=.45;size,_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1);tx=max(0,min(x1,w-size[0]-3));ty=max(18,min(y1-5,h-4))
            cv2.rectangle(image,(tx,max(0,ty-size[1]-5)),(min(w,tx+size[0]+4),ty+3),(28,32,40),-1);cv2.putText(image,text,(tx+2,ty),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
        shown.append({'track_id':detection.get('track_id'),'class_id':detection['class_id'],'bbox_xyxy':list(box),'match':active,'evidence_status':'uncertain_clothing_candidate' if active and plan.get('colors') else 'model_or_detection'})
    cv2.rectangle(image,(0,0,min(w,690),24),(28,32,40),-1);cv2.putText(image,f'People in frame: {people} | Objects: {objects} | Temporary tracks; actions are model estimates',(7,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    return image,shown

def _render(analysis: dict,plan: dict,root: Path,options: dict) -> dict:
    started=time.perf_counter();options={**DEFAULTS,**options};source=Path(analysis['source']);metadata=analysis['metadata']
    if digest(source)!=analysis['source_sha256']:raise ValueError('Unified playback source changed')
    effective=make_plan('',analysis) if options['view']=='Show All' else plan
    identity={'revision':'unified-player-v1','analysis':analysis['identity'],'actions':analysis['actions_sha256'],'options':options,'plan':effective}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();destination=root/key
    valid=_valid_cache(destination,identity,metadata)
    if valid:return _result(destination,valid,True,started)
    if destination.exists():
        # A prior partial/corrupt directory is preserved; recover at a new immutable sibling.
        return render(analysis,plan,root,{**options,'recovery':hashlib.sha256(json.dumps([destination.name,[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(destination.iterdir())]],sort_keys=True).encode()).hexdigest()})
    root.mkdir(parents=True,exist_ok=True);scale=min(1.,options['max_edge']/max(metadata['width'],metadata['height']));width=max(2,int(metadata['width']*scale)//2*2);height=max(2,int(metadata['height']*scale)//2*2);by_frame={f['frame_index']:f for f in analysis['frames']};audit=[];max_people=max_objects=max_matches=0
    with tempfile.TemporaryDirectory(prefix='overlay-',dir=root) as temporary:
        staging=Path(temporary);raw=staging/'render.avi';cap=cv2.VideoCapture(str(source));writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'MJPG'),metadata['fps'],(width,height));index=0
        if not writer.isOpened():raise RuntimeError('Cannot create unified annotated video')
        try:
            while True:
                ok,image=cap.read()
                if not ok:break
                timestamp=validated_timestamp(index,cap.get(cv2.CAP_PROP_POS_MSEC)/1000,metadata['fps']);frame=by_frame.get(index,{'frame_index':index,'timestamp':timestamp,'detections':[]})
                image=cv2.resize(image,(width,height));image,shown=draw(image,frame,analysis,effective,options);writer.write(image)
                max_people=max(max_people,sum(d['class_id']==0 for d in shown));max_objects=max(max_objects,sum(d['class_id']!=0 for d in shown));max_matches=max(max_matches,sum(d['match'] and d['class_id']==0 for d in shown));audit.append({'frame_index':index,'timestamp':timestamp,'targets':shown});index+=1
        finally:cap.release();writer.release()
        if index!=metadata['frame_count']:raise ValueError('Incomplete unified source decode')
        output=staging/'annotated.mp4';subprocess.run([find_ffmpeg(),'-v','error','-n','-i',str(raw),'-i',str(source),'-map','0:v:0','-map','1:a:0?','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-c:a','aac','-t',str(metadata['duration']),'-movflags','+faststart',str(output)],check=True);raw.unlink()
        check=cv2.VideoCapture(str(output));count=round(check.get(cv2.CAP_PROP_FRAME_COUNT));check.release()
        if count!=index:raise ValueError('Unified output timestamp/frame count mismatch')
        manifest={'identity':identity,'frames':index,'fps':metadata['fps'],'output_size':[width,height],'max_visible_people':max_people,'max_visible_objects':max_objects,'max_simultaneous_person_matches':max_matches,'render_seconds':time.perf_counter()-started,'notes':['Only actual observed source detections are drawn. No lost-track coordinates are predicted.','All action labels are model estimates, not verified ground truth.']}
        (staging/'frame-audit.json').write_text(json.dumps(audit),encoding='utf-8')
        manifest['video_sha256']=digest(output);manifest['audit_sha256']=digest(staging/'frame-audit.json')
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        published=publish(staging,destination,lambda p:_valid_cache(p,identity,metadata))
        if not published:return _result(destination,_valid_cache(destination,identity,metadata),True,started)
    return {'video':str((destination/'annotated.mp4').resolve()),'directory':str(destination.resolve()),'cache_hit':False,'request_seconds':time.perf_counter()-started,'manifest':manifest}


def _valid_cache(path: Path,identity: dict,metadata: dict) -> dict|None:
    try:
        manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'));video=path/'annotated.mp4';audit=path/'frame-audit.json'
        if json.dumps(manifest['identity'],sort_keys=True)!=json.dumps(identity,sort_keys=True) or not video.is_file() or not audit.is_file() or video.stat().st_size==0 or manifest['frames']!=metadata['frame_count']:return None
        if len(json.loads(audit.read_text(encoding='utf-8')))!=metadata['frame_count']:return None
        if manifest.get('video_sha256') and digest(video)!=manifest['video_sha256']:return None
        if manifest.get('audit_sha256') and digest(audit)!=manifest['audit_sha256']:return None
        cap=cv2.VideoCapture(str(video))
        try:
            valid=cap.isOpened() and round(cap.get(cv2.CAP_PROP_FRAME_COUNT))==metadata['frame_count'] and abs(cap.get(cv2.CAP_PROP_FPS)-metadata['fps'])<.01 and [round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]==manifest['output_size']
        finally:cap.release()
        return manifest if valid else None
    except (OSError,ValueError,KeyError):return None

def _result(destination,manifest,hit,started):return {'video':str((destination/'annotated.mp4').resolve()),'directory':str(destination.resolve()),'cache_hit':hit,'request_seconds':time.perf_counter()-started,'manifest':manifest}

def render(analysis: dict,plan: dict,root: Path,options: dict) -> dict:
    # Same identity as _render, including the neutral Show All plan, so repeated
    # queries share one lock and one valid completed render.
    options={**DEFAULTS,**options};effective=make_plan('',analysis) if options['view']=='Show All' else plan
    identity={'revision':'unified-player-v1','analysis':analysis['identity'],'actions':analysis['actions_sha256'],'options':options,'plan':effective}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    with cache_lock(root,key):
        result=_render(analysis,plan,root,options)
        result['cache_publication']='validated-key-lock-v2'
        return result
