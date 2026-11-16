"""Conservative candidate relationships from boxes, not ownership or physical contact."""
from dataclasses import asdict,dataclass
import math
import numpy as np

@dataclass(frozen=True)
class AssociationConfig:
    near_distance: float = .35
    max_gap_seconds: float = .2
    object_chain_iou: float = .65
    min_observations: int = 5
    min_duration_seconds: float = .5
    min_track_coverage: float = .8
    min_detection_confidence: float = .4
    max_relative_std: float = .12
    min_motion: float = .1

    def __post_init__(self):
        for name in ('near_distance','max_gap_seconds','min_duration_seconds','max_relative_std','min_motion'):
            value=getattr(self,name)
            if not math.isfinite(value) or value<0: raise ValueError('Invalid nonnegative association setting')
        for name in ('object_chain_iou','min_track_coverage','min_detection_confidence'):
            if not 0<getattr(self,name)<=1: raise ValueError('Invalid fractional setting')
        if not isinstance(self.min_observations,int) or self.min_observations<2: raise ValueError('Temporal evidence requires at least two observations')

    def to_dict(self): return asdict(self)

def box_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.

def spatial_evidence(person,object_box):
    p=person; b=object_box
    height=max(1,p[3]-p[1])
    dx=max(0,p[0]-b[2],b[0]-p[2]);dy=max(0,p[1]-b[3],b[1]-p[3])
    intersection=max(0,min(p[2],b[2])-max(p[0],b[0]))*max(0,min(p[3],b[3])-max(p[1],b[1]))
    fraction=intersection/max(1,(b[2]-b[0])*(b[3]-b[1]))
    pc=[(p[0]+p[2])/2,(p[1]+p[3])/2];bc=[(b[0]+b[2])/2,(b[1]+b[3])/2]
    return {'edge_distance_in_person_heights':math.hypot(dx,dy)/height,'object_fraction_inside_person_box':fraction,
        'box_iou':box_iou(p,b),'relative_offset':[(bc[0]-pc[0])/height,(bc[1]-pc[1])/height],
        'object_vertical_fraction':(bc[1]-p[1])/height,'person_center':pc,'object_center':bc,'person_height':height}


def associate(frames: list[dict], diagnostics: dict, config: AssociationConfig) -> dict:
    """Retain all spatial candidates; break evidence at gaps/ID changes; never merge person IDs."""
    qualities={t['track_id']:t for t in diagnostics['tracks']}
    histories={};events=[];frame_rows=[];chains={};counter=0
    def finish(key):
        event=histories.pop(key)
        evidence=event['supporting_observations'];last=evidence[-1]
        reasons=sorted({reason for row in evidence for reason in row['uncertainty_reasons']})
        duration=event['end_timestamp']-event['start_timestamp']
        person=qualities.get(event['person_track_id'],{})
        obj=qualities.get(event['object_track_id'],{})
        relative=np.array([row['spatial']['relative_offset'] for row in evidence if row['spatial']])
        relative_std=float(np.linalg.norm(relative.std(axis=0))) if len(relative) else None
        movement=0.
        if len(evidence)>1 and evidence[0]['spatial'] and evidence[-1]['spatial']:
            first=evidence[0]['spatial'];end=evidence[-1]['spatial']
            movement=float(np.linalg.norm(np.array(end['person_center'])-first['person_center'])/first['person_height'])
        stable_zone=all(row['spatial'] and .15<=row['spatial']['object_vertical_fraction']<=.85 and
            (row['spatial']['object_fraction_inside_person_box']>=.3 or row['spatial']['edge_distance_in_person_heights']<=.1)
            for row in evidence)
        confidence=min(min(row['person_detection_confidence'] or 0,row['object_detection_confidence']) for row in evidence)
        temporal=(len(evidence)>=config.min_observations and duration>=config.min_duration_seconds)
        tracks_reliable=(person.get('coverage',0)>=config.min_track_coverage and obj.get('coverage',0)>=config.min_track_coverage)
        ambiguous=any('multiple_nearby_people' in row['uncertainty_reasons'] or 'duplicate_object_boxes' in row['uncertainty_reasons'] for row in evidence)
        possible=(not reasons and temporal and tracks_reliable and confidence>=config.min_detection_confidence and
                  relative_std is not None and relative_std<=config.max_relative_std and movement>=config.min_motion and stable_zone)
        if possible: relation='possibly_carrying'
        elif ambiguous or event['person_track_id'] is None: relation='unknown'
        else: relation='nearby'
        if not temporal: reasons.append('insufficient_temporal_support')
        if movement<config.min_motion: reasons.append('insufficient_motion_evidence')
        if relative_std is None or relative_std>config.max_relative_std: reasons.append('relative_geometry_not_stable')
        if not stable_zone: reasons.append('outside_candidate_body_region')
        if confidence<config.min_detection_confidence: reasons.append('weak_detection_confidence')
        if not tracks_reliable: reasons.append('track_continuity_not_established')
        score=confidence*min(1,len(evidence)/config.min_observations)*(0.5 if ambiguous else 1.)
        event.update({'relationship':relation,'state':'candidate' if possible else 'uncertain',
            'evidence_score':score,'score_type':'uncalibrated heuristic, not probability',
            'uncertainty_reasons':sorted(set(reasons)),
            'temporal_summary':{'observations':len(evidence),'duration_seconds':duration,
                'relative_offset_std':relative_std,'person_displacement_in_heights':movement,
                'person_track_coverage':person.get('coverage'),'object_track_coverage':obj.get('coverage')},
            'disclaimer':'No ownership, physical contact or reliable carrying claim is established.'})
        events.append(event)
    previous_timestamp=-1.
    for frame in frames:
        if not math.isfinite(frame["timestamp"]) or frame["timestamp"]<=previous_timestamp:
            raise ValueError("Association frames require strictly increasing nonnegative timestamps")
        previous_timestamp=frame["timestamp"]
        timestamp=frame['timestamp'];index=frame['frame_index']
        for key in list(histories):
            if timestamp-histories[key]['end_timestamp']>config.max_gap_seconds+1e-6: finish(key)
        persons=[(i,d) for i,d in enumerate(frame['detections']) if d['class_id']==0]
        objects=[(i,d) for i,d in enumerate(frame['detections']) if d['class_id'] in (24,26,28)]
        output=[]
        # Untracked objects can have short geometric evidence chains, not invented track IDs.
        previous={key:value for key,value in chains.items() if timestamp-value['timestamp']<=config.max_gap_seconds+1e-6}
        proposals={}
        for oi,obj in objects:
            if obj['track_id']: continue
            proposals[oi]=[key for key,prior in previous.items() if prior['class_id']==obj['class_id'] and
                box_iou(prior['bbox'],obj['bbox_xyxy'])>=config.object_chain_iou]
        claimed={}
        for choices in proposals.values():
            for choice in choices: claimed[choice]=claimed.get(choice,0)+1
        next_chains={}
        for oi,obj in objects:
            if obj['track_id']: object_key='track:'+obj['track_id']
            else:
                choices=proposals.get(oi,[])
                if len(choices)==1 and claimed[choices[0]]==1: object_key=choices[0]
                else:
                    counter+=1;object_key=f"untracked-evidence-{counter}"
                next_chains[object_key]={'class_id':obj['class_id'],'bbox':obj['bbox_xyxy'],'timestamp':timestamp}
            candidates=[]
            for pi,person in persons:
                geometry=spatial_evidence(person['bbox_xyxy'],obj['bbox_xyxy'])
                if geometry['edge_distance_in_person_heights']<=config.near_distance:
                    candidates.append((pi,person,geometry))
            duplicate=any(other_i!=oi and other['class_id']==obj['class_id'] and
                box_iou(other['bbox_xyxy'],obj['bbox_xyxy'])>.5 for other_i,other in objects)
            if not candidates: candidates=[(None,None,None)]
            ambiguous=len(candidates)>1
            for pi,person,geometry in candidates:
                person_id=person['track_id'] if person else None
                reasons=[]
                if ambiguous: reasons.append('multiple_nearby_people')
                if duplicate: reasons.append('duplicate_object_boxes')
                if person is None: reasons.append('no_nearby_person_detected')
                elif not person_id: reasons.append('person_track_unconfirmed')
                if not obj['track_id']: reasons.append('object_track_unconfirmed')
                key=(person_id or ('no-person' if person is None else f'person-observation-{index}-{pi}'),object_key)
                supporting={'frame_index':index,'timestamp':timestamp,'object_detection_index':oi,
                    'person_detection_index':pi,'person_bbox_xyxy':person['bbox_xyxy'] if person else None,
                    'object_bbox_xyxy':obj['bbox_xyxy'],'person_detection_confidence':person['confidence'] if person else None,
                    'object_detection_confidence':obj['confidence'],'nearby_person_candidates':len(candidates) if person else 0,
                    'spatial':geometry,'uncertainty_reasons':reasons}
                if key not in histories:
                    histories[key]={'person_track_id':person_id,'object_class':obj['label'],
                        'object_class_id':obj['class_id'],'object_track_id':obj['track_id'],
                        'object_evidence_chain_id':object_key,'start_timestamp':timestamp,'end_timestamp':timestamp,
                        'supporting_observations':[]}
                histories[key]['supporting_observations'].append(supporting)
                histories[key]['end_timestamp']=timestamp
                output.append({'person_track_id':person_id,'object_track_id':obj['track_id'],
                    'object_evidence_chain_id':object_key,'object_class':obj['label'],**supporting})
        chains={**previous,**next_chains}
        frame_rows.append({'frame_index':index,'timestamp':timestamp,'candidates':output})
    for key in list(histories): finish(key)
    for number,event in enumerate(events,1): event['event_id']=f'candidate-{number}'
    counts={relationship:sum(event['relationship']==relationship for event in events)
            for relationship in ('nearby','possibly_carrying','unknown')}
    return {'schema':1,'config':config.to_dict(),'events':events,'frames':frame_rows,'summary':{
        'candidate_events':len(events),'relationships':counts,
        'object_observations':sum(sum(d['class_id'] in (24,26,28) for d in f['detections']) for f in frames),
        'candidate_pair_observations':sum(len(row['candidates']) for row in frame_rows),
        'objects_without_nearby_person':sum(c['person_detection_index'] is None for f in frame_rows for c in f['candidates'])}}
