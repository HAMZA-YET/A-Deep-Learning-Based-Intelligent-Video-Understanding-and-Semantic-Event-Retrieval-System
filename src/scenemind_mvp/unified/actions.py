"""Person-ROI SlowFast/AVA inference on actual multi-frame clips."""
from __future__ import annotations
from pathlib import Path
import time,threading
import cv2,numpy as np
from ..pipeline import digest
AVA={'running':10,'sitting':11,'standing':12,'walking':14,'carrying':17,'pickup':36}
POSTURES=('running','sitting','standing','walking')

class ActionModel:
    def __init__(self,cache: Path,device: str='auto'):
        import torch
        from pytorchvideo.models.hub import slowfast_r50_detection
        self.torch=torch;self.device='cuda' if device!='cpu' and torch.cuda.is_available() else 'cpu';self.lock=threading.RLock()
        self.weights=cache/'SLOWFAST_8x8_R50_DETECTION.pyth'
        if not self.weights.is_file():raise RuntimeError('Missing AVA checkpoint. Run scripts/phase3/download_model.py; detections remain usable with Action unknown.')
        started=time.perf_counter();self.model=slowfast_r50_detection(pretrained=False)
        self.model.load_state_dict(torch.load(self.weights,map_location='cpu',weights_only=True)['model_state']);self.model.eval().to(self.device)
        self.load_seconds=time.perf_counter()-started
    def predict(self,frames: list[np.ndarray],boxes: list[list[float]],config: dict) -> tuple[np.ndarray,tuple[int,int]]:
        torch=self.torch;h,w=frames[0].shape[:2];scale=min(config['short_side']/min(h,w),config['max_long_side']/max(h,w));nw,nh=round(w*scale),round(h*scale)
        clip=np.stack([cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2RGB),(nw,nh)) for f in frames])
        tensor=torch.from_numpy(clip).permute(3,0,1,2).float().div_(255);tensor=(tensor-.45)/.225
        slow=tensor[:,torch.linspace(0,tensor.shape[1]-1,8).long()];paths=[slow.unsqueeze(0).to(self.device),tensor.unsqueeze(0).to(self.device)]
        rois=np.asarray([[0,b[0]*nw/w,b[1]*nh/h,b[2]*nw/w,b[3]*nh/h] for b in boxes],dtype=np.float32)
        with self.lock,torch.inference_mode(),torch.autocast(device_type=self.device,enabled=self.device=='cuda'):
            prediction=self.model(paths,torch.from_numpy(rois).to(self.device)).float().cpu().numpy()
        if prediction.shape!=(len(boxes),80) or not np.isfinite(prediction).all() or np.any((prediction<0)|(prediction>1)):raise ValueError('Invalid AVA output shape/range')
        return prediction,(nw,nh)

def supported_intervals(observations: list[dict],config: dict,relation_events: list[dict]) -> list[dict]:
    """Adjacent temporal predictions only; no single-window action confirmation."""
    by_track={}
    for o in observations:by_track.setdefault((o['track_id'],o.get('candidate_action')),[]).append(o)
    intervals=[]
    for (track,_),rows in by_track.items():
        rows.sort(key=lambda o:o['timestamp'])
        for left,right in zip(rows,rows[1:]):
            action=left.get('candidate_action')
            if not action or right.get('candidate_action')!=action or right['timestamp']-left['timestamp']>config['interval_seconds']*1.25:continue
            if left['reasons'] or right['reasons']:continue
            if action in ('carrying','pickup'):
                required='possible_carrying' if action=='carrying' else 'possible_pickup'
                related=[e for e in relation_events if e.get('action')==required and e.get('status') in ('candidate','observed') and track in e.get('person_track_ids',[]) and e.get('object_track_id') and e['start_timestamp']<=left['timestamp'] and e['end_timestamp']>=right['timestamp'] and not any(any(flag in reason for flag in ('insufficient','unconfirmed','ambiguous','multiple','weak','low_track','continuity_not')) for reason in e.get('reason_codes',[]))]
                if not related:continue
            score=min(left['scores'][action],right['scores'][action])
            event={'person_track_id':track,'action':action,'start':left['timestamp'],'end':right['timestamp'],'status':'model_supported','score':score,'score_type':'Uncalibrated AVA model score, not validated accuracy','supporting_frames':[left['frame_index'],right['frame_index']],'supporting_clip_ranges':[left['clip_frame_range'],right['clip_frame_range']],'verified':False}
            if action in ('carrying','pickup'):
                event['object_evidence']=[{'class':e.get('object_class'),'track_id':e['object_track_id'],'event_action':e['action']} for e in related]
                event['relationship_status']='candidate_not_contact_or_ownership_proof'
            previous=next((e for e in reversed(intervals) if e['person_track_id']==track and e['action']==action and abs(e['end']-event['start'])<1e-6),None)
            if previous:previous['end']=event['end'];previous['score']=min(previous['score'],score);previous['supporting_frames'].append(right['frame_index']);previous['supporting_clip_ranges'].append(right['clip_frame_range'])
            else:intervals.append(event)
    return intervals

def analyze_actions(source: Path,frames: list[dict],metadata: dict,config: dict,model: ActionModel,relation_events: list[dict]) -> dict:
    fps=metadata['fps'];total=metadata['frame_count'];half=config['clip_seconds']/2;by_frame={f['frame_index']:f for f in frames};observations=[];cap=cv2.VideoCapture(str(source));started=time.perf_counter();forward_seconds=0.
    try:
        for center in np.arange(half,metadata['duration']-half,config['interval_seconds']):
            index=round(center*fps);indices=np.linspace(round((center-half)*fps),round((center+half)*fps),config['num_frames']).round().astype(int)
            if min(indices)<0 or max(indices)>=total or len(set(indices))<config['num_frames']:continue
            people=[d for d in by_frame.get(index,{}).get('detections',[]) if d['class_id']==0 and d.get('track_id')]
            if not people:continue
            clip=[]
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,frame=cap.read()
                if not ok:raise ValueError('Temporal clip frame failed decoding')
                clip.append(frame)
            t=time.perf_counter();scores,size=model.predict(clip,[p['bbox_xyxy'] for p in people],config);forward_seconds+=time.perf_counter()-t
            for person,vector in zip(people,scores):
                identifier=person['track_id'];coverage=sum(sum(d.get('track_id')==identifier and d['class_id']==0 for d in by_frame.get(int(i),{}).get('detections',[]))==1 for i in indices)/len(indices)
                values={name:float(vector[label-1]) for name,label in AVA.items()};order=sorted(POSTURES,key=lambda a:values[a],reverse=True);best=order[0];reasons=[]
                if coverage<config['min_coverage']:reasons.append('insufficient_unique_track_coverage')
                if (person['bbox_xyxy'][3]-person['bbox_xyxy'][1])*size[1]/metadata['height']<config['min_roi_height']:reasons.append('person_roi_too_small_for_action')
                if values[best]<config['threshold'] or values[best]-values[order[1]]<config['posture_margin']:reasons.append('weak_or_conflicting_posture_scores')
                observations.append({'track_id':identifier,'frame_index':index,'timestamp':index/fps,'bbox_xyxy':person['bbox_xyxy'],'scores':values,'top_ava_scores':[{'label_id':int(k)+1,'score':float(vector[k])} for k in np.argsort(vector)[-5:][::-1]],'coverage':coverage,'candidate_action':None if reasons else best,'reasons':reasons,'clip_frame_range':[int(min(indices)),int(max(indices))],'clip_sample_indices':indices.tolist(),'verified':False})
                for interaction in ('carrying','pickup'):
                    if values[interaction]>=config['threshold']:
                        observations.append({'track_id':identifier,'frame_index':index,'timestamp':index/fps,'scores':values,'coverage':coverage,'candidate_action':interaction,'reasons':[r for r in reasons if r!='weak_or_conflicting_posture_scores'],'clip_frame_range':[int(min(indices)),int(max(indices))],'verified':False})
    finally:cap.release()
    return {'observations':observations,'intervals':supported_intervals(observations,config,relation_events),'processing_seconds':time.perf_counter()-started,'forward_seconds':forward_seconds,'model_available':True,'model':'SlowFast R50 AVA v2.2','weights_sha256':digest(model.weights),'model_load_seconds':model.load_seconds,'device':model.device,'limits':['Model-supported is not independently verified.','No physical contact, ownership or gait speed ground truth.','Small or incomplete person clips remain unknown.']}
