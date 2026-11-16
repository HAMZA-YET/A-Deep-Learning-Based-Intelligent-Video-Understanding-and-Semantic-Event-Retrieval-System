"""Localize using individual region embeddings, never a full-frame score."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
from ..association.core import box_iou

PALETTE=('white','black','red','blue','green','gray','yellow','brown')
CLASSES={0:'Person',24:'Backpack',26:'Handbag',28:'Suitcase',56:'Chair'}

def attribute_profile(text: str,query: dict) -> dict|None:
    colors=query.get('colors',[])
    if len(colors)!=1:return None
    color='gray' if colors[0]=='grey' else colors[0]
    if color not in PALETTE:return None
    if re.search(r'\b(?:shirt|t-shirt|tee|top|jacket)\b',text.lower()):return {'requested':color,'kind':'shirt','classes':[0]}
    if query.get('object_classes'):return {'requested':color,'kind':'bag','classes':list(query['object_classes'])}
    return None

def target_label(record: dict) -> str:
    label=CLASSES.get(record['class_id'],'Object')
    identifier=record.get('track_id')
    if record['class_id']==0 and identifier and re.fullmatch(r'c0-t\d+',identifier):label+=' '+identifier.split('-t')[1]
    return label

def valid_box(box,width,height) -> bool:
    return len(box)==4 and all(np.isfinite(v) for v in box) and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height

def localize(result: dict,index,vector: np.ndarray,evidence,attribute_vectors: dict[str,np.ndarray]|None=None) -> dict:
    """Attach independently scored region candidates; does not change frame ranking."""
    query=result['query'];profile=attribute_profile(result.get('query_text',query.get('text','')),query)
    meta=index.full.manifest['metadata'];width,height=meta['width'],meta['height'];scores=index.vectors@vector[0];by_time={}
    identity=index.manifest['identity'];source=identity['source_sha256'];asset=index.directory.name
    if result.get('video_id',source)!=source or evidence.source_sha256!=source:raise ValueError('Target source mismatch')
    requested_person=bool(query.get('minimum_people')) or query.get('action') in ('walking','running','standing','sitting')
    expected={0} if requested_person else set(query.get('object_classes',[]))
    for record,score in zip(index.records,scores):
        if record.get('kind')!='detection' or record.get('class_id') not in CLASSES or not valid_box(record['bbox_xyxy'],width,height):continue
        if record.get('track_scope')!=source:continue
        by_time.setdefault(record['timestamp'],[]).append((record,float(score)))
    for row in result['results']:
        candidates=[]
        # Decoder timestamps are serialized through NumPy/JSON and may differ in low bits.
        # Only join the same source frame; never borrow detections from a nearby moment.
        fps=meta.get('fps');tolerance=min(1e-5,.25/fps) if fps else 1e-5
        matching_times=[t for t in by_time if abs(t-row['timestamp'])<=tolerance]
        matched_time=matching_times[0] if len(matching_times)==1 else None
        observations=by_time.get(matched_time,[])
        for record,score in observations:
            if expected and record['class_id'] not in expected:continue
            if not expected:continue # a scene query has no defensible object category to select
            cue=None;cue_supported=True
            if profile and record['class_id'] in profile['classes']:
                values={color:float(index.vectors[record['region_id']]@v[0]) for color,v in (attribute_vectors or {}).items()}
                desired=values.get(profile['requested']);other=max((v for k,v in values.items() if k!=profile['requested']),default=None)
                margin=desired-other if desired is not None and other is not None else None
                cue_supported=margin is not None and margin>=.01
                cue={'requested':profile['requested'],'kind':profile['kind'],'cosines':values,'margin':margin,'candidate_cue':cue_supported,'status':'unverified','note':'Uncalibrated region-text color contrast; not validated clothing/object color recognition.'}
            target={'id':asset+':'+str(record['region_id']),'region_id':record['region_id'],'class_id':record['class_id'],'label':target_label(record),'bbox_xyxy':record['bbox_xyxy'],'timestamp':record['timestamp'],'source_frame_index':record['source_frame_index'],'track_id':record.get('track_id'),'track_scope':source,'track_reference_status':record.get('track_reference_status','unconfirmed'),'region_cosine':score,'detection_confidence':record.get('confidence'),'attribute_evidence':cue,'auto_eligible':score>0 and cue_supported,'localization_status':'visual_candidate','query_fulfilled':False,'region_preview':str(index.preview(record['region_id'])),'associated_objects':[]}
            related_events=[e for e in evidence.events if target['track_id'] and (e.get('person_track_id')==target['track_id'] or target['track_id'] in e.get('person_track_ids',[])) and e.get('start',e.get('start_timestamp',float('inf')))<=target['timestamp']<=e.get('end',e.get('end_timestamp',-1))]
            target['temporal_evidence']={'requested_action':query.get('action'),'status':'unknown' if query.get('action') else 'not_requested','records':related_events,'reason':'Region similarity is not temporal evidence. Walking/running are not modelled. Candidate events are not independently verified.'}
            candidates.append(target)
        candidates.sort(key=lambda c:c['region_cosine'],reverse=True)
        # Prefer a temporally usable source-bound candidate for playback. This is track
        # reliability, not evidence that the query's action/attribute occurred. Keep all
        # detections inspectable, including small/untracked people.
        from .render import track_segment
        stride=1
        if evidence.frames:
            indices=sorted({f['frame_index'] for f in evidence.frames})
            gaps=[b-a for a,b in zip(indices,indices[1:]) if b>a]
            stride=min(gaps,default=1)
        for candidate in candidates:
            trace,summary=track_segment(evidence.frames,candidate,stride)
            duration=(max(trace)-min(trace))/fps if trace and fps else 0.
            candidate['highlight_continuity']={'observed_frames':len(trace),'span_seconds':duration,'status':summary['tracking'],'usable_video_segment':len(trace)>=3 and duration>=.5}
        reliable=[c for c in candidates if c['auto_eligible'] and c['highlight_continuity']['usable_video_segment']]
        selection_pool=reliable if len(reliable)>=max(1,query.get('minimum_people',1)) else candidates
        # A marginal low-confidence proposal must not outrank a clear person solely
        # through a tiny crop-cosine difference. This affects target recommendation,
        # never the original frame search ranking or a claim about the action.
        for candidate in candidates:
            confidence=candidate.get('detection_confidence')
            candidate['localization_rank_score']=candidate['region_cosine']*(float(confidence) if confidence is not None else 1.)
            candidate['localization_rank_score_type']='Detector-quality-weighted region cosine; uncalibrated ranking, not probability'
        selection_pool=sorted(selection_pool,key=lambda c:c['localization_rank_score'],reverse=True)
        # Overlapping duplicate proposals are alternatives, not two distinct people.
        primary=[]
        for candidate in selection_pool:
            if not candidate['auto_eligible']:continue
            if any(box_iou(candidate['bbox_xyxy'],old['bbox_xyxy'])>.5 for old in primary):continue
            primary.append(candidate)
            if len(primary)>=max(1,query.get('minimum_people',1)):break
        if query.get('minimum_people',0)>len(primary):primary=[]
        # No localization for unsupported identity requests.
        if any('identity' in warning for warning in query.get('unsupported_constraints',[])):primary=[]
        for target in candidates:
            if target['class_id']!=0 or not target['track_id']:continue
            for association in evidence.associations:
                if association.get('person_track_id')!=target['track_id']:continue
                for observation in association.get('supporting_observations',[]):
                    if observation['frame_index']!=target['source_frame_index']:continue
                    box=observation.get('object_bbox_xyxy');class_id=association.get('object_class_id',{'backpack':24,'handbag':26,'suitcase':28}.get(association.get('object_class')))
                    if class_id not in (24,26,28) or not box or not valid_box(box,width,height):continue
                    related={'id':'association:'+association['event_id'],'class_id':class_id,'label':CLASSES[class_id]+' (candidate)','bbox_xyxy':box,'track_id':association.get('object_track_id'),'track_scope':source,'source_frame_index':target['source_frame_index'],'timestamp':target['timestamp'],'relationship':'unverified','association_id':association['event_id'],'competing_people':sorted({a['person_track_id'] for a in evidence.associations if a.get('person_track_id') and any(o['frame_index']==observation['frame_index'] and o.get('object_detection_index')==observation.get('object_detection_index') for o in a.get('supporting_observations',[]))})}
                    if not any(box_iou(box,old['bbox_xyxy'])>.8 for old in target['associated_objects']):target['associated_objects'].append(related)
        row['localization']={'candidates':candidates,'primary_target_ids':[c['id'] for c in primary],'state':'localized_visual_candidate' if primary else 'no_specific_target','message':'Localized visual candidate; requested attributes, actions and relationships remain unverified.' if primary else 'No specific target localized.','timestamp_mapping':{'result_timestamp':row['timestamp'],'detection_timestamp':matched_time,'tolerance_seconds':tolerance,'matched_detections':len(observations)},'selection_basis':'Individual region-text cosine, class constraint and unvalidated attribute contrast; not full-frame similarity.','attribute_profile':profile,'evidence_bundle':str(index.manifest['bundle']),'source_sha256':source,'region_manifest':str(index.directory/'manifest.json')}
    return result
