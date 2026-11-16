"""Hybrid rank scores are separate from evidence gates and never event probabilities."""
import bisect
import time
from .schema import HybridConfig
from .query import parse_query,accepts_timestamp
from .evidence import Evidence
from ..association.core import box_iou

MODELED={'approaching','moving_together','stationary_near_object','possible_pickup','possible_carrying','possible_departure'}
CLASS_IDS={'person':0,'backpack':24,'handbag':26,'suitcase':28,'chair':56}

def _object_match(query,item):
    class_id=item.get('class_id',item.get('object_class_id',CLASS_IDS.get(item.get('object_class'))))
    return not query.object_classes or class_id in query.object_classes

def _row_at(event,frame_index):
    return [row for row in event.get('supporting_observations',[]) if row['frame_index']==frame_index]

class HybridEngine:
    def __init__(self,index,evidence: Evidence,config: HybridConfig):
        if index.manifest['identity']['video_sha256']!=evidence.source_sha256:raise ValueError('Semantic/structured source mismatch')
        self.index=index;self.evidence=evidence;self.config=config
        self.times=[frame['timestamp'] for frame in evidence.frames]
        self.sample_tolerance=.5/index.manifest['identity']['config']['sampling_fps']

    def _frame(self,timestamp):
        if not self.times:return None
        i=bisect.bisect_left(self.times,timestamp)
        options=[j for j in (i-1,i) if 0<=j<len(self.times)]
        nearest=min(options,key=lambda j:abs(self.times[j]-timestamp))
        return self.evidence.frames[nearest] if abs(self.times[nearest]-timestamp)<=self.config.alignment_tolerance else None

    def _evidence(self,timestamp,query):
        frame=self._frame(timestamp);reasons=list(query.unsupported_constraints)
        detections=frame['detections'] if frame else []
        persons=[d for d in detections if d['class_id']==0]
        ids=sorted({d['track_id'] for d in persons if d['track_id']})
        reliable=[identifier for identifier in ids if self.evidence.qualities.get(identifier,{}).get('coverage',0)>=.75]
        objects=[d for d in detections if d['class_id']!=0 and _object_match(query,d)]
        associations=[];events=[]
        if frame:
            associations=[a for a in self.evidence.associations if _object_match(query,a) and _row_at(a,frame['frame_index'])]
            events=[e for e in self.evidence.events if _object_match(query,e) and _row_at(e,frame['frame_index'])]
        if query.person_track_ids:
            associations=[a for a in associations if a['person_track_id'] in query.person_track_ids]
            events=[e for e in events if set(e['person_track_ids']) & set(query.person_track_ids)]
        current_object_ids={d['track_id'] for d in objects if d['track_id']}
        requested=[e for e in events if e['action']==query.action and e['status'] in ('observed','candidate')
                   and e['object_track_id'] is not None and e['object_track_id'] in current_object_ids]
        supported_ids=sorted({p for e in requested for p in e['person_track_ids']})
        spatial_ids=sorted({a['person_track_id'] for a in associations if a['person_track_id']})
        if query.minimum_people and len(reliable)<query.minimum_people:reasons.append('requested_people_count_or_track_continuity_unconfirmed')
        if query.object_classes and not objects:reasons.append('requested_object_not_detected_at_sample')
        if query.action in MODELED and not requested:reasons.append('requested_temporal_event_not_supported')
        if not frame:reasons.append('no_timestamp_aligned_structured_frame')
        if any(d['track_id'] is None for d in objects):reasons.append('object_track_unconfirmed')
        duplicate_people=any(box_iou(a['bbox_xyxy'],b['bbox_xyxy'])>.7 for i,a in enumerate(persons) for b in persons[i+1:])
        if duplicate_people:reasons.append('overlapping_person_detections_count_uncertain')
        ambiguous=any('multiple_nearby_people' in a.get('uncertainty_reasons',[]) for a in associations)
        if ambiguous:reasons.append('multiple_spatial_candidates_no_exclusive_assignment')
        if any(self.evidence.qualities.get(p,{}).get('gap_count',0)>0 for p in ids):reasons.append('tracklet_has_observation_gaps_not_identity_proof')
        incompatible=[]
        # Stationarity cannot support the requested motion pattern. This is not a physical-world negative label.
        if query.action in ('walking','possible_pickup') and not requested:
            stationary=[e for e in events if e['action']=='stationary_near_object' and e['status']=='observed']
            stationary_ids={p for e in stationary for p in e['person_track_ids']}
            if ids and set(ids)<=stationary_ids:
                incompatible=[e['event_id'] for e in stationary]
                reasons.append('requested_motion_incompatible_with_observed_stationarity')
        observation={'detected_person_track_ids':ids,'spatial_candidate_person_track_ids':spatial_ids,
            'supported_action_person_track_ids':supported_ids,'unconfirmed_person_detections':sum(d['track_id'] is None for d in persons),
            'objects':objects,'associations':[{'event_id':a['event_id'],'relationship':a['relationship'],'person_track_id':a['person_track_id'],
                'object_track_id':a['object_track_id'],'start_timestamp':a['start_timestamp'],'end_timestamp':a['end_timestamp'],
                'observations':_row_at(a,frame['frame_index'])} for a in associations],
            'temporal_events':[{'event_id':e['event_id'],'action':e['action'],'status':e['status'],
                'person_track_ids':e['person_track_ids'],'object_track_id':e['object_track_id'],
                'start_timestamp':e['start_timestamp'],'end_timestamp':e['end_timestamp'],'reason_codes':e['reason_codes']} for e in events],
            'structured_timestamp':frame['timestamp'] if frame else None,'incompatible_event_ids':incompatible}
        return observation,requested,reliable,sorted(set(reasons)),ambiguous,duplicate_people

    def _claim_gate(self,query,evidence,requested,reliable,ambiguous,duplicate):
        reasons=[]
        blocked=bool(query.unsupported_constraints or duplicate or
            (query.minimum_people and len(reliable)<query.minimum_people) or
            (query.object_classes and not evidence['objects']))
        if ambiguous and query.action in ('possible_pickup','possible_carrying','possible_departure'):
            blocked=True;reasons.append('competing_association_candidates_prevent_interaction_claim')
        if query.person_track_ids and query.action in MODELED and not set(query.person_track_ids)<=set(evidence['supported_action_person_track_ids']):
            blocked=True;reasons.append('requested_track_ids_not_all_supported_for_event')
        if requested and not set(evidence['supported_action_person_track_ids'])<=set(reliable):
            blocked=True;reasons.append('event_person_track_continuity_unconfirmed')
        return not blocked,reasons

    def search(self,text: str,vector,mode: str='hybrid',candidate_timestamps=None) -> dict:
        if mode not in ('hybrid','clip_only'):raise ValueError('Unknown retrieval mode')
        started=time.perf_counter();query=parse_query(text);c=self.config
        semantic=self.index.search_index.search(vector,len(self.index.search_index.timestamps),0,-1)
        allowed=None if candidate_timestamps is None else list(candidate_timestamps)
        results=[]
        for candidate in semantic:
            timestamp=candidate['timestamp']
            if not accepts_timestamp(query,timestamp,self.sample_tolerance):continue
            if allowed is not None and not any(abs(timestamp-t)<1e-5 for t in allowed):continue
            if mode=='clip_only':
                results.append({'timestamp':timestamp,'frame_index':candidate['frame_index'],'preview':str(self.index.preview(candidate['frame_index'])),
                    'preview_timestamp':timestamp,'clip_cosine':candidate['score'],'rank_score':candidate['score'],
                    'score_type':'CLIP cosine ranking score, not probability','state':'visual_match_only','event_claim':None,
                    'description':'CLIP visual candidate only; action, identity and object relationships are not verified.',
                    'score_components':{'clip_cosine':candidate['score']},'evidence':{},
                    'uncertainty':['clip_only_does_not_verify_query_constraints']+list(query.unsupported_constraints),
                    'track_scope':self.evidence.source_sha256})
                continue
            evidence,requested,reliable,reasons,ambiguous,duplicate=self._evidence(timestamp,query)
            if query.person_track_ids and not set(query.person_track_ids)<=set(evidence['detected_person_track_ids']):continue
            components={'semantic':(candidate['score']+1)/2,
                'object':float(bool(query.object_classes and evidence['objects'])),
                'track':min(1,len(reliable)/max(1,query.minimum_people)) if query.minimum_people else 0.,
                'association':(.5 if ambiguous else 1.) if query.action=='nearby' and evidence['associations'] else 0.,
                'event':max((1. if event['status']=='observed' else .5 for event in requested),default=0.)}
            if 'negation_not_supported' in query.unsupported_constraints:
                components['event']=0.;components['association']=0.
            missing=any(reason in reasons for reason in ('requested_people_count_or_track_continuity_unconfirmed','requested_object_not_detected_at_sample','requested_temporal_event_not_supported'))
            conflict=bool(evidence['incompatible_event_ids'])
            score=sum(components[name]*getattr(c,name+'_weight') for name in components)-c.missing_penalty*missing-c.conflict_penalty*conflict
            event_claim=None;state='uncertain_visual_match'
            can_claim,gate_reasons=self._claim_gate(query,evidence,requested,reliable,ambiguous,duplicate)
            reasons.extend(gate_reasons)
            if mode=='hybrid' and requested and can_claim:
                state='supported_image_plane_observation' if all(e['status']=='observed' for e in requested) else 'candidate_event_evidence'
                event_claim={'action':query.action,'status':'observed' if state=='supported_image_plane_observation' else 'candidate',
                    'event_ids':[e['event_id'] for e in requested],'scope':'image-plane geometry or possible interaction; no ownership/contact claim'}
            elif mode=='hybrid' and query.action=='nearby' and evidence['associations']:
                state='uncertain_spatial_candidates'
            elif mode=='hybrid' and query.action is None and evidence['objects'] and can_claim:
                state='detector_object_observation'
            if mode=='hybrid' and c.strict_events and query.action in MODELED and event_claim is None:continue
            if mode=='clip_only':score=candidate['score'];state='visual_match_only';event_claim=None
            notes=list(reasons)
            if query.at is not None and abs(timestamp-query.at)>c.alignment_tolerance:notes.append('nearest_sample_timestamp_not_exact_requested_time')
            if event_claim:
                description=('Image-plane event evidence' if event_claim['status']=='observed' else 'Possible event candidate')+': '+query.action+'. Physical contact and ownership are not established.'
            elif state=='detector_object_observation':description='Detector-labelled object observation; classification may be wrong.'
            elif state=='uncertain_spatial_candidates':description='Image-near person/object candidates; all competing tracklets retained, relationship uncertain.'
            else:description='Visual candidate only. The requested action/attributes are not established by structured evidence.'
            results.append({'timestamp':timestamp,'frame_index':candidate['frame_index'],
                'preview':str(self.index.preview(candidate['frame_index'])),'preview_timestamp':timestamp,
                'clip_cosine':candidate['score'],'rank_score':float(score),'score_type':'ranking score, not probability',
                'state':state,'event_claim':event_claim,'description':description,'score_components':components,
                'missing_evidence_penalty_applied':bool(missing and mode=='hybrid'),
                'incompatible_motion_penalty_applied':bool(conflict and mode=='hybrid'),
                'evidence':evidence,'uncertainty':notes,'track_scope':self.evidence.source_sha256})
        results.sort(key=lambda row:row['rank_score'],reverse=True)
        selected=[]
        for row in results:
            if any(abs(row['timestamp']-r['timestamp'])<c.min_gap_seconds for r in selected):continue
            selected.append(row)
            if len(selected)==c.top_k:break
        all_spatial=[];all_supported=[];all_candidate=[];track_records=[]
        # Complete structured intervals are reported separately, so short intervals between sampled frames are not lost.
        for a in self.evidence.associations if query.action=='nearby' and mode=='hybrid' else []:
            if not _object_match(query,a):continue
            valid=any(accepts_timestamp(query,o['timestamp'],0.) for o in a['supporting_observations'])
            if not valid or (query.person_track_ids and a['person_track_id'] not in query.person_track_ids):continue
            if a['person_track_id']:all_spatial.append(a['person_track_id'])
            track_records.append({'source':'association','event_id':a['event_id'],'person_track_id':a['person_track_id'],
                'object_track_id':a['object_track_id'],'object_class':a['object_class'],'relationship':a['relationship'],
                'start_timestamp':a['start_timestamp'],'end_timestamp':a['end_timestamp'],
                'state':'uncertain','reason':'Spatial candidate, not a verified requested action or exclusive ownership',
                'supporting_observations':[o for o in a['supporting_observations'] if accepts_timestamp(query,o['timestamp'],0.)]})
        for e in self.evidence.events if mode=='hybrid' else []:
            if e['action']!=query.action or e['status'] not in ('observed','candidate') or not _object_match(query,e):continue
            if not any(accepts_timestamp(query,o['timestamp'],0.) for o in e['supporting_observations']):continue
            ids=e['person_track_ids']
            if query.person_track_ids:ids=[p for p in ids if p in query.person_track_ids]
            admissible=False
            for observation in e['supporting_observations']:
                if not accepts_timestamp(query,observation['timestamp'],0.):continue
                moment,requested,reliable,reasons,ambiguous,duplicate=self._evidence(observation['timestamp'],query)
                allowed,_=self._claim_gate(query,moment,requested,reliable,ambiguous,duplicate)
                if allowed and any(item['event_id']==e['event_id'] for item in requested):admissible=True;break
            if admissible:
                if e['status']=='observed':all_supported.extend(ids)
                else:all_candidate.extend(ids)
            track_records.append({'source':'temporal','event_id':e['event_id'],'person_track_ids':ids,'status':e['status'],
                'admissible_for_query':admissible,
                'start_timestamp':e['start_timestamp'],'end_timestamp':e['end_timestamp'],
                'unverified_query_constraints':list(query.unsupported_constraints),
                'supporting_observations':[o for o in e['supporting_observations'] if accepts_timestamp(query,o['timestamp'],0.)]})
        claims=sum(row['event_claim'] is not None for row in selected)
        status='supported_evidence' if claims else ('uncertain' if selected else 'no_reliable_match')
        return {'implementation':'hybrid-evidence-v1','query':query.to_dict(),'mode':mode,'config':c.to_dict(),'status':status,
            'message':('Structured event evidence is attached; candidate labels are not physical-event proof.' if claims else
                'No reliable requested event is established; results are explicitly uncertain/visual candidates.' if selected else
                'No reliable match within these constraints. Missing evidence does not prove absence.'),
            'results':selected,'all_supported_action_track_ids':sorted(set(all_supported)),
            'all_candidate_action_track_ids':sorted(set(all_candidate)),'all_spatial_candidate_track_ids':sorted(set(all_spatial)),
            'complete_structured_track_intervals':track_records if mode=='hybrid' else [],
            'track_scope':self.evidence.source_sha256,'ranking_seconds':time.perf_counter()-started,
            'identity_note':'IDs are temporary tracklets. Fragmented IDs are not merged or counted as distinct people.'}
