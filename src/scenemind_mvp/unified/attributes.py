"""Localized native-resolution torso color cues, explicitly not clothing verification."""
from __future__ import annotations
from pathlib import Path
import json,time,hashlib,tempfile
import cv2,numpy as np
from ..pipeline import digest
from ..highlighting.targets import valid_box
from .cache import cache_lock,publish
POLICY={'revision':'torso-hsv-cues-v2-masked','interval_seconds':1.,'min_roi_width':15,'min_roi_height':20,'min_fraction':.3,'black_fraction':.5,'white_fraction':.45,'dominance_margin':.1,'min_visible_fraction':.35}
COLORS=('green','black','white','red','blue','yellow','brown','gray')

def pixel_fractions(image,visible=None) -> dict:
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);h,s,v=cv2.split(hsv)
    masks={'green':(h>=35)&(h<=85)&(s>=65)&(v>=45),'black':v<=65,'white':(v>=180)&(s<=70),'red':((h<=10)|(h>=170))&(s>=90)&(v>=50),'blue':(h>=90)&(h<=130)&(s>=65)&(v>=45),'yellow':(h>=20)&(h<35)&(s>=80)&(v>=65),'brown':(h>=5)&(h<25)&(s>=70)&(v>=45)&(v<180),'gray':(s<=50)&(v>65)&(v<180)}
    if visible is None:visible=np.ones(image.shape[:2],dtype=bool)
    count=int(visible.sum())
    return {color:float((mask&visible).sum()/count) for color,mask in masks.items()} if count else {}

def cue_intervals(observations: list[dict]) -> list[dict]:
    groups={}
    for o in observations:
        if o['track_id']:groups.setdefault(o['track_id'],[]).append(o)
    intervals=[]
    for track,rows in groups.items():
        for left,right in zip(rows,rows[1:]):
            if right['timestamp']-left['timestamp']>1.25 or left['reasons'] or right['reasons']:continue
            for color in set(left['candidate_colors'])&set(right['candidate_colors']):
                previous=next((i for i in reversed(intervals) if i['track_id']==track and i['color']==color and abs(i['end']-left['timestamp'])<1e-6),None)
                if previous:previous['end']=right['timestamp'];previous['supporting_observations'].append(right)
                else:intervals.append({'track_id':track,'color':color,'start':left['timestamp'],'end':right['timestamp'],'status':'uncertain_torso_color_candidate','verified':False,'supporting_observations':[left,right],'note':'Torso pixels may include objects, skin or background. This does not verify a shirt or ownership.'})
    return intervals

def _valid(path: Path,identity: dict):
    try:
        result=json.loads((path/'attributes.json').read_text(encoding='utf-8'))
        return result if result['identity']==identity else None
    except (OSError,ValueError,KeyError):return None

def analyze_attributes(analysis: dict,root: Path) -> dict:
    identity={'source':analysis['source_sha256'],'detections':analysis['identity']['records_sha256'],'policy':POLICY};key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();destination=root/key
    with cache_lock(root,key):
        cached=_valid(destination,identity)
        if cached:return cached
        if destination.exists():raise ValueError('Incomplete attribute cache was preserved: '+str(destination))
        source=Path(analysis['source'])
        if digest(source)!=analysis['source_sha256']:raise ValueError('Color source changed')
        root.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(source));observations=[];next_time=0.;started=time.perf_counter();width=analysis['metadata']['width'];height=analysis['metadata']['height']
        with tempfile.TemporaryDirectory(prefix='attributes-',dir=root) as temporary:
            staging=Path(temporary);(staging/'previews').mkdir()
            try:
                for frame in analysis['frames']:
                    if frame['timestamp']+1e-6<next_time:continue
                    next_time=(int(frame['timestamp'])+1.)
                    cap.set(cv2.CAP_PROP_POS_FRAMES,frame['frame_index']);ok,image=cap.read()
                    if not ok:raise ValueError('Clothing source frame failed decoding')
                    bags=[d['bbox_xyxy'] for d in frame['detections'] if d['class_id'] in (24,26,28)]
                    for d in frame['detections']:
                        if d['class_id']!=0 or not d.get('track_id') or not valid_box(d['bbox_xyxy'],width,height):continue
                        x1,y1,x2,y2=d['bbox_xyxy'];w=x2-x1;h=y2-y1;torso=[round(x1+.2*w),round(y1+.2*h),round(x1+.8*w),round(y1+.55*h)];tx1,ty1,tx2,ty2=torso;roi=image[ty1:ty2,tx1:tx2];reasons=[]
                        if roi.shape[1]<POLICY['min_roi_width'] or roi.shape[0]<POLICY['min_roi_height']:reasons.append('native_torso_crop_too_small')
                        area=max(1,(tx2-tx1)*(ty2-ty1))
                        visible=np.ones(roi.shape[:2],dtype=bool)
                        for b in bags:
                            bx1=max(tx1,round(b[0]));by1=max(ty1,round(b[1]));bx2=min(tx2,round(b[2]));by2=min(ty2,round(b[3]))
                            if bx2>bx1 and by2>by1:visible[by1-ty1:by2-ty1,bx1-tx1:bx2-tx1]=False
                        visible_fraction=float(visible.mean()) if visible.size else 0.
                        if visible_fraction<POLICY['min_visible_fraction'] or visible.sum()<POLICY['min_roi_width']*POLICY['min_roi_height']:reasons.append('insufficient_unmasked_torso_pixels')
                        fractions=pixel_fractions(roi,visible) if roi.size else {};candidate=[]
                        if fractions and not reasons:
                            for color,value in fractions.items():
                                threshold=POLICY.get(color+'_fraction',POLICY['min_fraction'])
                                if value>=threshold and value-max(v for c,v in fractions.items() if c!=color)>=POLICY['dominance_margin']:candidate.append(color)
                        name=f"{d['track_id']}-{frame['frame_index']}.jpg";preview=staging/'previews'/name
                        person=image[round(y1):round(y2),round(x1):round(x2)].copy()
                        if person.size:
                            cv2.rectangle(person,(tx1-round(x1),ty1-round(y1)),(tx2-round(x1),ty2-round(y1)),(0,190,255),2);cv2.imwrite(str(preview),person)
                        observations.append({'track_id':d['track_id'],'timestamp':frame['timestamp'],'frame_index':frame['frame_index'],'person_bbox_xyxy':d['bbox_xyxy'],'torso_analysis_bbox_xyxy':torso,'fractions':fractions,'visible_torso_fraction':visible_fraction,'mask_policy':'Exclude detected object boxes, without inferring ownership or physical occlusion.','candidate_colors':candidate,'reasons':reasons,'preview':str((destination/'previews'/name).resolve()),'verified':False})
            finally:cap.release()
            result={'identity':identity,'observations':observations,'intervals':cue_intervals(observations),'processing_seconds':time.perf_counter()-started,'status':'uncalibrated_localized_color_cues','verified':False,'limits':['Not shirt segmentation or validated color recognition.','Detected object boxes are excluded; insufficient visible torso pixels remain unknown.','Never use these cues as proof of carrying or ownership.']}
            (staging/'attributes.json').write_text(json.dumps(result,indent=2),encoding='utf-8');publish(staging,destination,lambda p:_valid(p,identity))
        return result
