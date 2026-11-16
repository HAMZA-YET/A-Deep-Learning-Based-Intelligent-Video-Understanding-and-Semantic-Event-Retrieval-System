"""Draw only validated source detections; terminate traces at missing/ambiguous evidence."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,math,tempfile,subprocess,time
from functools import lru_cache
import cv2
from PIL import Image,ImageDraw,ImageFont,ImageOps
from ..pipeline import digest
from ..video import inspect_video
from ..hybrid.evidence import load_bundle
from ..detection.pipeline import find_ffmpeg,validated_timestamp
from ..association.core import box_iou
from .targets import valid_box

@lru_cache(maxsize=64)
def _verified(path: str,size: int,mtime: int,sha: str):
    if digest(Path(path))!=sha:raise ValueError('Highlight source changed')
    return inspect_video(Path(path))
def verified_source(source: Path,sha: str):
    stat=source.stat();return _verified(str(source.resolve()),stat.st_size,stat.st_mtime_ns,sha)

def project_box(box: list[float],source_size: tuple[int,int],content_size: tuple[int,int],offset: tuple[int,int]=(0,0)) -> tuple[int,int,int,int]:
    if not valid_box(box,*source_size):raise ValueError('Invalid detection coordinates')
    sx=content_size[0]/source_size[0];sy=content_size[1]/source_size[1]
    return tuple(round(v*(sx if i%2==0 else sy)+offset[i%2]) for i,v in enumerate(box))

def draw_frame(image: Image.Image,targets: list[dict],size: tuple[int,int]|None=None,letterbox: bool=False) -> Image.Image:
    source_size=image.size
    if size:
        if letterbox:
            content=ImageOps.contain(image.convert('RGB'),size);offset=((size[0]-content.width)//2,(size[1]-content.height)//2);canvas=Image.new('RGB',size,'#ECF1F7');canvas.paste(content,offset);content_size=content.size
        else:canvas=image.convert('RGB').resize(size,Image.Resampling.LANCZOS);content_size=size;offset=(0,0)
    else:canvas=image.convert('RGB').copy();content_size=source_size;offset=(0,0)
    draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=12 if canvas.width<=420 else 17)
    for i,target in enumerate(targets):
        x1,y1,x2,y2=project_box(target['bbox_xyxy'],source_size,content_size,offset);color='#F59E0B' if target.get('relationship') else ('#0D9488' if i==0 else '#2563EB')
        draw.rectangle((x1,y1,x2,y2),outline=color,width=3)
        label=target['label'];bounds=draw.textbbox((0,0),label,font=font);w=bounds[2]+8;h=bounds[3]-bounds[1]+6;x=max(0,min(x1,canvas.width-w));y=max(0,min(y1-h,canvas.height-h))
        draw.rectangle((x,y,x+w,y+h),fill=color);draw.text((x+4,y+2-bounds[1]),label,font=font,fill='white')
    return canvas

def validate_regions(localization: dict,targets: list[dict]) -> dict:
    manifest_path=Path(localization['region_manifest']);m=json.loads(manifest_path.read_text(encoding='utf-8'));records_path=manifest_path.parent/'regions.json'
    if m['identity']['source_sha256']!=localization['source_sha256'] or digest(records_path)!=m['regions_sha256']:raise ValueError('Highlight region provenance mismatch')
    records=json.loads(records_path.read_text());source=localization['source_sha256']
    for target in targets:
        if target.get('track_scope')!=source:raise ValueError('Cross-video target is forbidden')
        if 'region_id' not in target:
            _,evidence=load_bundle(Path(localization['evidence_bundle']))
            event=next((a for a in evidence.associations if a['event_id']==target.get('association_id')),None)
            if event is None or not any(o['frame_index']==target['source_frame_index'] and o.get('object_bbox_xyxy')==target['bbox_xyxy'] for o in event['supporting_observations']):raise ValueError('Associated target has no recorded box')
        if 'region_id' in target:

            record=records[target['region_id']]
            for field in ('class_id','bbox_xyxy','source_frame_index','track_id'):
                if record.get(field)!=target.get(field):raise ValueError('Target is not an indexed detection')
            if record['kind']!='detection':raise ValueError('A context tile is not a localized object')
    return m

def thumbnail(source: Path,localization: dict,targets: list[dict],output_root: Path) -> Path:
    if not targets:raise ValueError('No specific target localized')
    m=validate_regions(localization,targets);info=verified_source(source,localization['source_sha256']);frame_index=targets[0]['source_frame_index']
    if any(t['source_frame_index']!=frame_index for t in targets):raise ValueError('Targets must share one frame')
    key=hashlib.sha256(json.dumps({'revision':'target-thumb-v1','source':localization['source_sha256'],'regions':m['regions_sha256'],'targets':render_specs(targets),'size':[320,180]},sort_keys=True).encode()).hexdigest();directory=output_root/'thumbnails';directory.mkdir(parents=True,exist_ok=True);path=directory/(key+'.png')
    if path.is_file():return path
    cap=cv2.VideoCapture(str(source))
    try:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index):raise ValueError('Cannot seek target frame')
        ok,frame=cap.read()
        if not ok:raise ValueError('Target source frame could not be decoded')
        image=draw_frame(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),targets,(320,180),True)
        with tempfile.NamedTemporaryFile(suffix='.png',dir=directory,delete=False) as temporary:temp=Path(temporary.name)
        try:image.save(temp);temp.replace(path)
        finally:temp.unlink(missing_ok=True)
    finally:cap.release()
    return path

def render_specs(targets: list[dict]) -> list[dict]:
    return [{key:t.get(key) for key in ('id','region_id','association_id','class_id','bbox_xyxy','source_frame_index','track_id','track_scope','label','relationship')} for t in targets]

def track_segment(frames: list[dict],target: dict,stride: int=1) -> tuple[dict[int,dict],dict]:
    anchor=target['source_frame_index'];identifier=target.get('track_id');class_id=target['class_id'];by_frame={f['frame_index']:f for f in frames}
    anchor_view={'bbox_xyxy':target['bbox_xyxy'],'label':target['label'],'relationship':target.get('relationship'),'track_id':identifier}
    if not identifier:return {anchor:anchor_view},{'tracking':'untracked','frames':1,'reason':'No confirmed track reference; anchor detection only.'}
    def observation(index):
        rows=[d for d in by_frame.get(index,{}).get('detections',[]) if d['class_id']==class_id and d.get('track_id')==identifier]
        return rows[0] if len(rows)==1 else None
    seed=observation(anchor)
    if seed is None or box_iou(seed['bbox_xyxy'],target['bbox_xyxy'])<.5:return {anchor:anchor_view},{'tracking':'unconfirmed','frames':1,'reason':'Anchor could not be bound uniquely to the source tracker.'}
    trace={anchor:anchor_view};reasons={}
    for direction in (-1,1):
        previous=seed;index=anchor+direction*stride
        while index>=0:
            current=observation(index)
            if current is None:reasons[str(direction)]='Missing or ambiguous track observation; stopped without interpolation.';break
            a=previous['bbox_xyxy'];b=current['bbox_xyxy'];diagonal=math.hypot(a[2]-a[0],a[3]-a[1]);step=math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)/max(1.,diagonal)
            ratio=(b[2]-b[0])*(b[3]-b[1])/max(1.,(a[2]-a[0])*(a[3]-a[1]))
            if step>.75 or not .25<=ratio<=4:reasons[str(direction)]='Large box discontinuity; stopped rather than guessing identity.';break
            trace[index]={**anchor_view,'bbox_xyxy':b};previous=current;index+=direction*stride
    return trace,{'tracking':'observed_tracklet','frames':len(trace),'first_frame':min(trace),'last_frame':max(trace),'stops':reasons,'note':'Temporary track reference, not verified identity continuity. Gaps are never bridged.'}

def highlighted_video(source: Path,localization: dict,targets: list[dict],output_root: Path,max_edge: int=960) -> dict:
    if not targets:raise ValueError('No specific target localized')
    started=time.perf_counter();m=validate_regions(localization,targets);_,evidence=load_bundle(Path(localization['evidence_bundle']));sha=localization['source_sha256'];info=verified_source(source,sha)
    if evidence.source_sha256!=sha:raise ValueError('Highlight evidence is from another source')
    detection=Path(evidence.provenance['detection_directory']);d=json.loads((detection/'manifest.json').read_text());stride=d['identity']['config']['every_n_frames']
    key_input={'revision':'selected-target-cfr-v1','source':sha,'records_sha256':digest(detection/'records.json'),'regions_sha256':m['regions_sha256'],'targets':render_specs(targets),'max_edge':max_edge,'tracking_policy':'stop-at-first-missing-ambiguous-or-jump-v1'}
    key=hashlib.sha256(json.dumps(key_input,sort_keys=True).encode()).hexdigest();destination=output_root/'videos'/key
    if (destination/'manifest.json').is_file() and (destination/'highlighted.mp4').is_file():
        existing=json.loads((destination/'manifest.json').read_text());return {'directory':str(destination.resolve()),'video':str((destination/'highlighted.mp4').resolve()),'cache_hit':True,'manifest':existing,'request_seconds':time.perf_counter()-started}
    traces=[];track_info=[]
    for target in targets:
        if 'association_id' in target:
            event=next((a for a in evidence.associations if a['event_id']==target['association_id']),None)
            if event is None or not any(o['frame_index']==target['source_frame_index'] and box_iou(o['object_bbox_xyxy'],target['bbox_xyxy'])>.95 for o in event['supporting_observations']):raise ValueError('Associated box has no source evidence')
        trace,summary=track_segment(evidence.frames,target,stride);traces.append(trace);track_info.append({'target_id':target['id'],**summary})
    scale=min(1.,max_edge/max(info.width,info.height));width=max(2,int(info.width*scale)//2*2);height=max(2,int(info.height*scale)//2*2)
    destination.parent.mkdir(parents=True,exist_ok=True);audit=[];highlighted=0
    with tempfile.TemporaryDirectory(prefix='render-',dir=destination.parent) as temporary:
        staging=Path(temporary);raw=staging/'render.avi';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'MJPG'),info.fps,(width,height))
        if not writer.isOpened():raise RuntimeError('Cannot initialize highlight video writer')
        capture=cv2.VideoCapture(str(source));index=0
        try:
            while True:
                ok,frame=capture.read()
                if not ok:break
                timestamp=validated_timestamp(index,capture.get(cv2.CAP_PROP_POS_MSEC)/1000,info.fps);boxes=[trace[index] for trace in traces if index in trace];highlighted+=bool(boxes)
                image=draw_frame(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),boxes,(width,height));draw=ImageDraw.Draw(image);draw.rectangle((0,0,min(width,640),24),fill='#14243A');draw.text((7,5),'Selected visual candidate - attributes/actions/relationships unverified',fill='white')
                writer.write(cv2.cvtColor(__import__('numpy').asarray(image),cv2.COLOR_RGB2BGR));audit.append({'frame_index':index,'timestamp':timestamp,'visible_target_ids':[targets[j]['id'] for j,trace in enumerate(traces) if index in trace]});index+=1
        finally:capture.release();writer.release()
        if index!=info.frame_count:raise ValueError('Highlight source failed full decode')
        output=staging/'highlighted.mp4'
        subprocess.run([find_ffmpeg(),'-v','error','-n','-i',str(raw),'-i',str(source),'-map','0:v:0','-map','1:a:0?','-c:v','libx264','-pix_fmt','yuv420p','-preset','fast','-crf','23','-c:a','aac','-t',str(info.duration),'-movflags','+faststart',str(output)],check=True)
        raw.unlink();out_info=inspect_video(output)
        if out_info.frame_count!=info.frame_count:raise ValueError('Highlight output frame count mismatch')
        manifest={'identity':key_input,'source':str(source.resolve()),'source_frames':info.frame_count,'fps':info.fps,'output_size':[width,height],'highlighted_frames':highlighted,'track_segments':track_info,'render_seconds':time.perf_counter()-started,'notes':['Only selected detections are drawn. No other people are implied query matches.','No interpolation, lost-track boxes, guessed identity merges, clothing verification or action recognition.','Resized copies preserve source timestamps; original source video is unchanged.']}
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');(staging/'frame-audit.json').write_text(json.dumps(audit),encoding='utf-8')
        if not destination.exists():staging.rename(destination)
    return {'directory':str(destination.resolve()),'video':str((destination/'highlighted.mp4').resolve()),'cache_hit':False,'manifest':manifest,'request_seconds':time.perf_counter()-started}
