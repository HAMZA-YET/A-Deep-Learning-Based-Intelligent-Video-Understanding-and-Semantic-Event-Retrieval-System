"""Region scores rank visual candidates; they never manufacture temporal evidence."""
from __future__ import annotations
from dataclasses import replace
import time,re
import numpy as np
from ..search import normalize
from ..hybrid.schema import HybridConfig
from ..hybrid.engine import HybridEngine
from ..hybrid.query import parse_query
from .schema import RegionConfig
from .pipeline import RegionIndex

def parse_region_query(text: str):
    query=parse_query(text)
    if re.search(r'\b(?:run|running|runs|jogging|jog)\b',text.lower()):
        query=replace(query,action='running',unsupported_constraints=tuple(sorted(set(query.unsupported_constraints+('action_not_modelled:running',)))))
    return query

class RegionEngine:
    def __init__(self,regions: RegionIndex,evidence,config: RegionConfig,hybrid: HybridConfig|None=None):
        self.regions=regions;self.config=config;self.evidence=evidence;self.hybrid_config=hybrid or HybridConfig()
        if regions.manifest['identity']['source_sha256']!=evidence.source_sha256:raise ValueError('Region/evidence source mismatch')
        self.base=HybridEngine(regions.full,evidence,replace(self.hybrid_config,top_k=len(regions.full.search_index.timestamps),min_gap_seconds=0.,strict_events=False))
    def search(self,text: str,vector: np.ndarray,mode: str='region_hybrid',top_k: int=5,min_gap: float=1.,strict_actions: bool=True,candidate_timestamps: list[float]|None=None) -> dict:
        if mode not in ('clip_only','region','region_hybrid'):raise ValueError('Unknown region search mode')
        if not isinstance(top_k,int) or top_k<1 or not np.isfinite(min_gap) or min_gap<0:raise ValueError('Invalid result parameters')
        started=time.perf_counter();query=parse_region_query(text);vector=normalize(vector)
        if vector.shape!=(1,self.regions.manifest['dimension']):raise ValueError('Query/region dimension mismatch')
        base=self.base.search(text,vector,'hybrid' if mode=='region_hybrid' else 'clip_only',candidate_timestamps)
        scores=self.regions.vectors@vector[0] if len(self.regions.records) else np.empty(0);grouped={}
        for record,score in zip(self.regions.records,scores):
            # Person crops can contain a missed bag. Context views are never object detections.
            if query.object_classes and record['class_id'] not in (*query.object_classes,0,None):continue
            if query.minimum_people and not query.object_classes and record['class_id'] not in (0,None):continue
            if query.person_track_ids and record.get('track_id') not in query.person_track_ids:continue
            grouped.setdefault(record['timestamp'],[]).append((float(score),record))
        rows=[]
        for original in base['results']:
            row=dict(original);matches=sorted(grouped.get(row['timestamp'],[]),key=lambda x:x[0],reverse=True);best=matches[:self.config.aggregate_top_regions];mean=float(np.mean([s for s,_ in best])) if best else row['clip_cosine'];full=row['clip_cosine']
            fused=(1-self.config.region_weight)*full+self.config.region_weight*max(full,mean) if mode!='clip_only' else full
            row['rank_score']=(row['rank_score']+self.hybrid_config.semantic_weight*(fused-full)/2) if mode=='region_hybrid' else fused
            row['semantic_sample_index']=row['frame_index'];row['source_frame_index']=best[0][1].get('source_frame_index') if best else None
            row['full_frame_cosine']=full;row['region_aggregate_cosine']=mean if best else None;row['fused_visual_score']=fused
            row['all_proposal_track_ids']=sorted({r['track_id'] for _,r in matches if r.get('track_id')})
            row['regions']=[{'cosine':s,**r,'preview':str(self.regions.preview(r['region_id']))} for s,r in best] if mode!='clip_only' else []
            # A frame-level event about another person cannot become a claim about this crop.
            source_claim=row.get('event_claim');row['frame_event_evidence']=source_claim
            supported=set(row.get('evidence',{}).get('supported_action_person_track_ids',[]));region_ids={r['track_id'] for _,r in best if r.get('track_id')}
            if source_claim and not (region_ids & supported):
                row['event_claim']=None;row['uncertainty']=row.get('uncertainty',[])+['temporal_event_not_bound_to_retrieved_region']
            if query.action and (mode!='region_hybrid' or row.get('event_claim') is None):
                row['event_claim']=None;row['state']='unknown_action_visual_candidate';row['uncertainty']=row.get('uncertainty',[])+['requested_action_requires_supported_multi_frame_evidence']
                if strict_actions:continue
            if row.get('event_claim') is None:
                row['description']='Visual region/full-frame candidate only. Actions and person-object relationships are not established.'
            row['score_type']='Ranking score, not probability';rows.append(row)
        rows.sort(key=lambda r:r['rank_score'],reverse=True);selected=[]
        for row in rows:
            if any(abs(row['timestamp']-r['timestamp'])<min_gap for r in selected):continue
            selected.append(row)
            if len(selected)==top_k:break
        return {'implementation':'region-native-square-v1','query':query.to_dict(),'mode':mode,'strict_actions':strict_actions,'status':'supported_temporal_evidence' if any(r.get('event_claim') for r in selected) else ('uncertain_visual_candidates' if selected else 'no_supported_match'),'results':selected,'ranking_seconds':time.perf_counter()-started,'config':self.config.to_dict(),'identity_note':'IDs are source-frame matched temporary tracklets; unknown/ambiguous references are not merged.','limits':['Crop similarity is not action,relationship,color or identity proof.','Walking versus running is not modelled and requires additional verified temporal evidence; strict mode abstains.','Foreground/background depth is not verified by crop area.']}
