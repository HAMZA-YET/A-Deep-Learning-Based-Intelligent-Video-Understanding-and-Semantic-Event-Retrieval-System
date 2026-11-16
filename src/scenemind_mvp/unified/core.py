"""Evidence-gated multi-target plans, independent of frame-level CLIP ranking."""
from __future__ import annotations
import re
from ..regions.engine import parse_region_query
from ..hybrid.query import accepts_timestamp
ACTION_NAMES={'walking':'walking','running':'running','sitting':'sitting','standing':'standing','possible_pickup':'pickup','possible_carrying':'carrying'}

def label(d: dict,ids: bool=True,objects: bool=True) -> str:
    name='Person' if d['class_id']==0 else d['label'].title()
    if d['class_id']!=0 and not objects:return ''
    track=d.get('track_id')
    if ids:return name+' '+track.split('-t')[-1] if track else name+' (untracked)'
    return name

def action_at(intervals: list[dict],track: str|None,timestamp: float,requested: str|None=None) -> dict|None:
    matches=[e for e in intervals if track and e['person_track_id']==track and e['start']-1e-6<=timestamp<=e['end']+1e-6 and e['status']=='model_supported']
    if requested:matches=[e for e in matches if e['action']==requested]
    else:
        postures=[e for e in matches if e['action'] in ('walking','running','standing','sitting')]
        if postures:matches=postures
    actions={e['action'] for e in matches}
    return max(matches,key=lambda e:e['score']) if len(actions)==1 else None

def make_plan(text: str,analysis: dict,inspect: str|None=None) -> dict:
    parsed=parse_region_query(text) if text.strip() else None
    plan={'query':text,'mode':'all' if parsed is None and not inspect else 'search','action':None,'classes':[],'track_ids':[],'reasons':[],'intervals':[],'minimum_people':0,'colors':[],'garment':None,'attribute_intervals':[],'explanations':[],'time':parsed.to_dict() if parsed else None}
    if inspect:plan['track_ids']=[inspect]
    elif parsed:
        plan['track_ids']=list(parsed.person_track_ids)
        explicit=re.search(r'\bperson\s+(\d+)\b',text.lower())
        if explicit:plan['track_ids']=['c0-t'+explicit.group(1)]
        plan['classes']=list(parsed.object_classes) if not parsed.minimum_people else [0]
        plan['minimum_people']=parsed.minimum_people
        plan['action']=ACTION_NAMES.get(parsed.action)
        if parsed.colors:
            plan['colors']=list(dict.fromkeys('gray' if c=='grey' else c for c in parsed.colors))
            plan['garment']='shirt' if re.search(r'\b(?:shirt|t-shirt|top|blouse)\b',text.lower()) else None
            plan['explanations'].append('Localized torso-color cues are uncertain candidates, not verified clothing attributes.')
            attributes=analysis.get('attributes')
            if len(plan['colors'])!=1:plan['reasons'].append('multiple_colors_not_supported')
            elif plan['garment'] is None:plan['reasons'].append('requested_color_scope_not_supported')
            elif attributes is None or attributes.get('error'):plan['reasons'].append('localized_clothing_analysis_unavailable')
            else:
                plan['attribute_intervals']=[i for i in attributes['intervals'] if i['color']==plan['colors'][0] and (not plan['track_ids'] or i['track_id'] in plan['track_ids'])]
                if not plan['attribute_intervals']:plan['reasons'].append('no_repeated_localized_color_candidate')
        if parsed.action and not plan['action']:plan['reasons'].append('requested_event_not_supported_by_action_model')
        if any('negation' in s or 'identity' in s for s in parsed.unsupported_constraints):plan['reasons'].append('unsupported_constraint')
        if not (plan['track_ids'] or plan['classes'] or plan['action']):plan['reasons'].append('scene_similarity_has_no_structured_target')
    if plan['action']:
        plan['classes']=[0]
        plan['intervals']=[e for e in analysis['actions']['intervals'] if e['action']==plan['action'] and (not plan['track_ids'] or e['person_track_id'] in plan['track_ids'])]
        if parsed and parsed.object_classes and plan['action'] in ('carrying','pickup'):
            names={24:'backpack',26:'handbag',28:'suitcase',56:'chair'}
            requested={names.get(c) for c in parsed.object_classes}
            plan['intervals']=[e for e in plan['intervals'] if any(o['class'] in requested for o in e.get('object_evidence',[]))]
        if not plan['intervals']:plan['reasons'].append('no_supported_temporal_match')
    if plan.get('colors') and plan['action'] and not plan['reasons']:
        intersections=[]
        for event in plan['intervals']:
            for attribute in plan['attribute_intervals']:
                start=max(event['start'],attribute['start']);end=min(event['end'],attribute['end'])
                if event['person_track_id']==attribute['track_id'] and start<end:
                    intersections.append({**event,'start':start,'end':end,'query_match_status':'uncertain_clothing_candidate','color_cue':attribute['color']})
        plan['intervals']=intersections
        if not intersections:plan['reasons'].append('no_temporal_and_clothing_evidence_overlap')
    if plan['reasons']:plan['intervals']=[]
    return plan

def matched_indices(frame: dict,plan: dict,analysis: dict) -> list[int]:
    if plan['mode']=='all':return list(range(len(frame['detections'])))
    if plan['reasons']:return []
    if plan.get('time'):
        from ..hybrid.query import ParsedQuery
        p=ParsedQuery(**plan['time'])
        if not accepts_timestamp(p,frame['timestamp'],sample_tolerance=.5/analysis['metadata']['fps']):return []
    indexes=[]
    for i,d in enumerate(frame['detections']):
        if plan['track_ids'] and d.get('track_id') not in plan['track_ids']:continue
        if plan['classes'] and d['class_id'] not in plan['classes']:continue
        if plan['action']:
            event=action_at(plan['intervals'],d.get('track_id'),frame['timestamp'],plan['action'])
            if not event or event['action']!=plan['action']:continue
        if plan.get('colors') and not any(e['track_id']==d.get('track_id') and e['start']-1e-6<=frame['timestamp']<=e['end']+1e-6 for e in plan['attribute_intervals']):continue
        indexes.append(i)
    if plan['minimum_people']>1 and sum(frame['detections'][i]['class_id']==0 for i in indexes)<plan['minimum_people']:return []
    return indexes

def match_summary(plan: dict,analysis: dict) -> dict:
    ids=set();samples=[];frames=0
    for frame in analysis['frames']:
        indexes=matched_indices(frame,plan,analysis)
        if indexes:
            frames+=1
            if len(samples)<20:samples.append({'timestamp':frame['timestamp'],'frame_index':frame['frame_index'],'track_ids':[frame['detections'][i].get('track_id') for i in indexes]})
            ids.update(frame['detections'][i]['track_id'] for i in indexes if frame['detections'][i].get('track_id'))
    return {'plan':plan,'track_ids':sorted(ids),'matched_observed_frames':frames,'sample_moments':samples,'status':('uncertain_clothing_candidates' if plan.get('colors') else 'supported_model_candidates') if frames else 'unknown_or_no_match','verified_track_ids':[],'uncertain_track_ids':sorted(ids) if plan.get('colors') else [],'verified':False}
