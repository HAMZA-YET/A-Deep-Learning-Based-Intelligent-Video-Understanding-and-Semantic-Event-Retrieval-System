"""Evaluation adapter over the existing temporal search, without semantic scores."""
from __future__ import annotations
import time
from ..temporal.core import search_events
from ..hybrid.query import parse_query,accepts_timestamp
from ..hybrid.engine import MODELED,CLASS_IDS

def structured_search(query_text: str, evidence, candidate_timestamps: list[float], top_k: int=5, min_gap: float=1.) -> dict:
    started=time.perf_counter();query=parse_query(query_text)
    if query.unsupported_constraints or query.action not in MODELED:
        return {'results':[],'status':'no_reliable_match','ranking_seconds':time.perf_counter()-started,'reason':'Query is outside existing temporal event vocabulary or has unverified attributes.'}
    events=search_events({'events':evidence.events},action=query.action)
    events=[e for e in events if e['status'] in ('observed','candidate') and (not query.object_classes or CLASS_IDS.get(e['object_class']) in query.object_classes) and (not query.person_track_ids or set(query.person_track_ids)<=set(e['person_track_ids'])) and len(e['person_track_ids'])>=query.minimum_people]
    events.sort(key=lambda e:(e['status']!='observed',e['start_timestamp'],e['event_id']))
    rows=[]
    for event in events:
        for timestamp in sorted(candidate_timestamps):
            if not event['start_timestamp']<=timestamp<=event['end_timestamp'] or not accepts_timestamp(query,timestamp,0.):continue
            if any(abs(timestamp-r['timestamp'])<min_gap for r in rows):continue
            rows.append({'timestamp':timestamp,'state':'candidate_event_evidence' if event['status']=='candidate' else 'supported_image_plane_observation','event_claim':{'action':event['action'],'status':event['status'],'event_ids':[event['event_id']],'scope':'Existing temporal record; not physical contact/ownership proof'},'source_event':event,'rank_score':None})
            if len(rows)==top_k:break
        if len(rows)==top_k:break
    return {'results':rows,'status':'supported_evidence' if rows else 'no_reliable_match','ranking_seconds':time.perf_counter()-started,'note':'Status then chronological ordering, not learned confidence; unknown event records abstain.'}
