"""Temporal rules over measured box trajectories; never contact/ownership conclusions."""
from dataclasses import asdict,dataclass
import math
import numpy as np
from ..association.core import spatial_evidence,box_iou

ACTIONS=('approaching','moving_together','stationary_near_object','possible_pickup','possible_carrying','possible_departure','unknown')

@dataclass(frozen=True)
class TemporalConfig:
    min_observations: int = 6
    min_duration_seconds: float = .6
    max_gap_seconds: float = .2
    min_track_coverage: float = .75
    min_detection_confidence: float = .35
    motion_min: float = .12
    stationary_max: float = .08
    relative_std_max: float = .12
    distance_change_min: float = .25
    near_distance: float = .35
    pickup_lift_min: float = .15
    phase_min_seconds: float = .3

    def __post_init__(self):
        if not isinstance(self.min_observations,int) or self.min_observations<3: raise ValueError('At least three observations required')
        for key,value in asdict(self).items():
            if not math.isfinite(value) or value<=0: raise ValueError('Temporal settings must be positive finite values')
        if self.min_track_coverage>1 or self.min_detection_confidence>1: raise ValueError('Fractions exceed one')

    def to_dict(self): return asdict(self)

def measurements(rows: list[dict]) -> dict:
    valid=[r for r in rows if r.get('person_bbox_xyxy') and r.get('object_bbox_xyxy')]
    if len(valid)<2: return {'observations':len(valid),'duration_seconds':0.}
    times=np.array([r['timestamp'] for r in valid]);relative=[];persons=[];objects=[];edges=[];vertical=[]
    scale=float(np.median([r['person_bbox_xyxy'][3]-r['person_bbox_xyxy'][1] for r in valid]))
    for row in valid:
        spatial=spatial_evidence(row['person_bbox_xyxy'],row['object_bbox_xyxy'])
        persons.append(spatial['person_center']);objects.append(spatial['object_center'])
        relative.append(spatial['relative_offset']);edges.append(spatial['edge_distance_in_person_heights'])
        vertical.append(spatial['object_vertical_fraction'])
    persons=np.asarray(persons)/scale;objects=np.asarray(objects)/scale
    pv=persons[-1]-persons[0];ov=objects[-1]-objects[0]
    distance=np.linalg.norm(persons-objects,axis=1)
    dt=times[-1]-times[0]
    changes=np.diff(distance)
    cosine=float(np.dot(pv,ov)/(np.linalg.norm(pv)*np.linalg.norm(ov))) if np.linalg.norm(pv)*np.linalg.norm(ov)>1e-9 else None
    return {'observations':len(valid),'duration_seconds':float(dt),'scale_person_height_pixels':scale,
        'person_displacement':float(np.linalg.norm(pv)),'object_displacement':float(np.linalg.norm(ov)),
        'person_excursion':float(np.linalg.norm(persons-persons[0],axis=1).max()),
        'object_excursion':float(np.linalg.norm(objects-objects[0],axis=1).max()),
        'person_speed_in_heights_per_second':float(np.linalg.norm(pv)/dt) if dt>0 else None,
        'object_speed_in_heights_per_second':float(np.linalg.norm(ov)/dt) if dt>0 else None,
        'motion_direction_cosine':cosine,'relative_offset_std':float(np.linalg.norm(np.std(relative,axis=0))),
        'distance_change':float(distance[-1]-distance[0]),'closing_step_fraction':float(np.mean(changes<-.005)),
        'separating_step_fraction':float(np.mean(changes>.005)),
        'mean_edge_distance':float(np.mean(edges)),'first_edge_distance':float(edges[0]),
        'mean_object_vertical_fraction':float(np.mean(vertical)),'coordinate_scope':'image-plane; no depth/contact estimate'}

def _together(m,config):
    return (m.get('person_displacement',0)>=config.motion_min and m.get('object_displacement',0)>=config.motion_min
        and m.get('motion_direction_cosine') is not None and m['motion_direction_cosine']>=.85
        and m['relative_offset_std']<=config.relative_std_max)

def _analyze(rows,person_id,object_id,object_class,qualities,warnings,config,association_ids,carrying_seed=False):
    m=measurements(rows);reasons=set(warnings)
    if len(rows)<config.min_observations or rows[-1]['timestamp']-rows[0]['timestamp']+1e-6<config.min_duration_seconds:
        reasons.add('insufficient_multi_frame_evidence')
    if not person_id: reasons.add('person_track_unconfirmed')
    if not object_id: reasons.add('object_track_unconfirmed')
    for identifier in (person_id,object_id):
        if identifier and qualities.get(identifier,{}).get('coverage',0)<config.min_track_coverage: reasons.add('low_track_observation_coverage')
    for row in rows:
        reasons.update(row.get('uncertainty_reasons',[]))
        if min(row.get('person_detection_confidence') or 0,row['object_detection_confidence'])<config.min_detection_confidence:
            reasons.add('weak_detection_confidence')
    outcomes=[]
    if not reasons:
        if m['person_excursion']<=config.stationary_max and m['object_excursion']<=config.stationary_max and m['mean_edge_distance']<=config.near_distance:
            outcomes.append(('stationary_near_object','observed',['sustained_image_plane_stationarity']))
        if _together(m,config):
            outcomes.append(('moving_together','observed',['compatible_multi_frame_image_plane_motion']))
            if carrying_seed and .15<=m['mean_object_vertical_fraction']<=.85:
                outcomes.append(('possible_carrying','candidate',['sustained_comotion','prior_candidate_association','body_region_consistency','contact_not_verified']))
        if m['object_excursion']<=config.stationary_max and m['person_displacement']>=config.motion_min:
            if m['distance_change']<=-config.distance_change_min and m['closing_step_fraction']>=.7:
                outcomes.append(('approaching','observed',['sustained_image_plane_distance_decrease']))
            if m['distance_change']>=config.distance_change_min and m['separating_step_fraction']>=.7 and m['first_edge_distance']<=config.near_distance:
                outcomes.append(('possible_departure','candidate',['visible_person_object_separation_increases','object_remains_observed','departure_semantics_uncertain']))
        third=len(rows)//3
        if third>=3:
            before=rows[:third];after=rows[-third:]
            middle=rows[third:-third]
            b=measurements(before);a=measurements(after);mid=measurements(middle)
            ys=np.array([(r['object_bbox_xyxy'][1]+r['object_bbox_xyxy'][3])/2 for r in middle])
            upward_fraction=float(np.mean(np.diff(ys)<-.005*m['scale_person_height_pixels'])) if len(ys)>1 else 0.
            before_y=np.median([(r['object_bbox_xyxy'][1]+r['object_bbox_xyxy'][3])/2 for r in before])
            after_y=np.median([(r['object_bbox_xyxy'][1]+r['object_bbox_xyxy'][3])/2 for r in after])
            lift=float((before_y-after_y)/m['scale_person_height_pixels'])
            m['pickup_phase_evidence']={'before':b,'middle':mid,'after':a,'upward_displacement':lift,'upward_step_fraction':upward_fraction}
            if (b['duration_seconds']>=config.phase_min_seconds and a['duration_seconds']>=config.phase_min_seconds
                and mid['duration_seconds']>=config.phase_min_seconds and upward_fraction>=.7
                and b['object_excursion']<=config.stationary_max and b['mean_edge_distance']<=config.near_distance
                and lift>=config.pickup_lift_min and _together(a,config)):
                outcomes.append(('possible_pickup','candidate',['object_stationary_before','upward_transition','subsequent_person_object_comotion','contact_not_verified']))
    if not outcomes:
        outcomes=[('unknown','unknown',sorted(reasons or {'no_supported_temporal_pattern'}))]
    minimum_confidence=min(min(row.get('person_detection_confidence') or 0,row['object_detection_confidence']) for row in rows)
    return [{'action':action,'status':status,'start_timestamp':rows[0]['timestamp'],'end_timestamp':rows[-1]['timestamp'],
        'person_track_ids':[person_id] if person_id else [],'object_class':object_class,'object_track_id':object_id,
        'reason_codes':codes,'motion_evidence':m,'supporting_observations':list(rows),
        'object_observations':[{'frame_index':r['frame_index'],'timestamp':r['timestamp'],'bbox_xyxy':r['object_bbox_xyxy'],
                               'confidence':r['object_detection_confidence'],'track_id':object_id} for r in rows],
        'source_association_event_ids':association_ids,'evidence_strength':minimum_confidence,
        'score_type':'minimum supporting detector score, not an event probability',
        'uncertainty':['2D projection and camera motion may mislead; no physical-contact/ownership conclusion.',
                       'Raw track IDs are tracklets; no distinct-person count or cross-ID identity merge.']}
        for action,status,codes in outcomes]

def reason_events(detection_frames: list[dict],association: dict,diagnostics: dict,config: TemporalConfig) -> dict:
    """Tracked pairs use full intersecting histories; untracked/ambiguous evidence remains unknown."""
    qualities={t['track_id']:t for t in diagnostics['tracks']}
    events=[];pairs={};previous=-1.
    for frame in detection_frames:
        if not math.isfinite(frame['timestamp']) or frame['timestamp']<=previous: raise ValueError('Timestamps must increase')
        previous=frame['timestamp']
    object_fragmentation=[];previous_objects=[]
    for frame in detection_frames:
        current=[d for d in frame['detections'] if d['class_id'] in (24,26,28) and d['track_id']]
        old_ids={d['track_id'] for d in previous_objects};new_ids={d['track_id'] for d in current}
        for old in previous_objects:
            for new in current:
                if old['track_id'] not in new_ids and new['track_id'] not in old_ids and old['class_id']==new['class_id'] and box_iou(old['bbox_xyxy'],new['bbox_xyxy'])>=.5:
                    object_fragmentation.append({'timestamp':frame['timestamp'],'old':old['track_id'],'new':new['track_id']})
        previous_objects=current
    for candidate in association['events']:
        pid=candidate['person_track_id'];oid=candidate['object_track_id']
        if not pid or not oid:
            warnings=set(candidate.get('uncertainty_reasons',[]))
            # Association uncertainty is not sufficient evidence for motion conclusions.
            events.extend(_analyze(candidate['supporting_observations'],pid,oid,candidate['object_class'],qualities,
                warnings,config,[candidate['event_id']]))
        else:
            pair=pairs.setdefault((pid,oid),{'label':candidate['object_class'],'ids':[],'carrying':False})
            pair['ids'].append(candidate['event_id']);pair['carrying']|=candidate['relationship']=='possibly_carrying'
    fragmentation=diagnostics.get('high_overlap_id_change_candidates',[])
    for (pid,oid),seed in pairs.items():
        rows=[]
        def flush():
            if not rows: return
            warnings=set()
            if any((item['previous_track_id']==pid or item['new_track_id']==pid) and
                rows[0]['timestamp']-config.max_gap_seconds<=item['timestamp']<=rows[-1]['timestamp']+config.max_gap_seconds for item in fragmentation):
                warnings.add('possible_person_track_fragmentation')
            if any((item['old']==oid or item['new']==oid) and rows[0]['timestamp']-config.max_gap_seconds<=item['timestamp']<=rows[-1]['timestamp']+config.max_gap_seconds for item in object_fragmentation):
                warnings.add('possible_object_track_fragmentation')
            events.extend(_analyze(rows,pid,oid,seed['label'],qualities,warnings,config,seed['ids'],seed['carrying']))
            rows.clear()
        for frame in detection_frames:
            persons=[d for d in frame['detections'] if d['track_id']==pid]
            objects=[d for d in frame['detections'] if d['track_id']==oid]
            if len(persons)!=1 or len(objects)!=1: continue
            person,obj=persons[0],objects[0]
            geometry=spatial_evidence(person['bbox_xyxy'],obj['bbox_xyxy'])
            if geometry['edge_distance_in_person_heights']>1.5: continue
            if rows and frame['timestamp']-rows[-1]['timestamp']>config.max_gap_seconds+1e-6: flush()
            competitors=[d for d in frame['detections'] if d['class_id']==0 and d['track_id']!=pid and
                spatial_evidence(d['bbox_xyxy'],obj['bbox_xyxy'])['edge_distance_in_person_heights']<=config.near_distance]
            warnings=[]
            if competitors:warnings.append('multiple_nearby_people')
            if any(d is not obj and d['class_id']==obj['class_id'] and box_iou(d['bbox_xyxy'],obj['bbox_xyxy'])>.5 for d in frame['detections']):
                warnings.append('duplicate_object_boxes')
            rows.append({'frame_index':frame['frame_index'],'timestamp':frame['timestamp'],
                'person_bbox_xyxy':person['bbox_xyxy'],'object_bbox_xyxy':obj['bbox_xyxy'],
                'person_detection_confidence':person['confidence'],'object_detection_confidence':obj['confidence'],
                'uncertainty_reasons':warnings})
        flush()
    for i,event in enumerate(events,1):event['event_id']=f'temporal-{i}'
    return {'schema':1,'config':config.to_dict(),'events':events,
        'summary':{'events':len(events),'actions':{a:sum(e['action']==a for e in events) for a in ACTIONS},
                   'statuses':{s:sum(e['status']==s for e in events) for s in ('observed','candidate','unknown')},
                   'distinct_people':None},
        'scope':'Observed means measured image-plane geometry, not physical-world interaction or identity.'}

def search_events(timeline: dict,action: str | None=None,object_category: str | None=None,
                  timestamp: float | None=None,status: str | None=None) -> list[dict]:
    if action is not None and action not in ACTIONS: raise ValueError('Unknown action filter')
    if timestamp is not None and (not math.isfinite(timestamp) or timestamp<0): raise ValueError('Invalid timestamp filter')
    if status is not None and status not in ('observed','candidate','unknown'): raise ValueError('Invalid status filter')
    return [e for e in timeline['events'] if (action is None or e['action']==action)
        and (object_category is None or e['object_class'].lower()==object_category.lower())
        and (timestamp is None or e['start_timestamp']-1e-6<=timestamp<=e['end_timestamp']+1e-6)
        and (status is None or e['status']==status)]


def playback_bounds(event: dict, duration: float) -> tuple[int,int | None]:
    """Whole-second bounds that include point events and never truncate the final frame."""
    start_value=event['start_timestamp'];end_value=event['end_timestamp']
    if not all(math.isfinite(x) for x in (start_value,end_value,duration)) or not 0<=start_value<=end_value<duration:
        raise ValueError('Event interval must lie within the video')
    start=max(0,math.floor(start_value-.5));end=math.ceil(end_value+.5)
    return start,None if end>=duration else max(start+1,end)
