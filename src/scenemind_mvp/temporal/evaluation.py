"""Scoped case-level evaluation; unknown/unsupported metrics stay explicit."""
from ..association.core import box_iou

def evaluate_cases(timeline: dict,benchmark: dict) -> dict:
    if timeline['source_sha256']!=benchmark['source_sha256']:raise ValueError('Benchmark source mismatch')
    policy=benchmark['matching_policy'];rows=[];tp=fp=fn=tn=0;ambiguous_covered=ambiguous_abstained=0
    for case in benchmark['cases']:
        matching=[]
        for event in timeline['events']:
            observations=[r for r in event['supporting_observations'] if r.get('person_bbox_xyxy')]
            if not observations:continue
            nearest=min(observations,key=lambda r:abs(r['timestamp']-case['anchor_timestamp']))
            if abs(nearest['timestamp']-case['anchor_timestamp'])>policy['max_anchor_time_distance']:continue
            if min(box_iou(nearest['person_bbox_xyxy'],case['person_bbox_xyxy']),box_iou(nearest['object_bbox_xyxy'],case['object_bbox_xyxy']))<policy['min_box_iou']:continue
            matching.append(event)
        asserted=[e for e in matching if e['status'] in policy['accepted_statuses']]
        row={'case_id':case['case_id'],'kind':case['kind'],'action':case['action'],'evidence_covered':bool(matching),
             'matching_event_ids':[e['event_id'] for e in matching],'asserted_actions':[e['action'] for e in asserted]}
        if case['kind']=='ambiguous':
            if matching:
                ambiguous_covered+=1
                abstained=not asserted and all(e['status']=='unknown' for e in matching)
                ambiguous_abstained+=abstained;row['remained_unknown']=abstained
            else:row['remained_unknown']=None
        else:
            hits=[]
            for event in asserted:
                if event['action']!=case['action']:continue
                intersection=max(0,min(event['end_timestamp'],case['end_timestamp'])-max(event['start_timestamp'],case['start_timestamp']))
                union=max(event['end_timestamp'],case['end_timestamp'])-min(event['start_timestamp'],case['start_timestamp'])
                if union>0 and intersection/union>=policy['min_temporal_iou']:hits.append(event)
            predicted=bool(hits);row['predicted']=predicted
            if case['kind']=='positive':
                tp+=predicted;fn+=not predicted;row['outcome']='TP' if predicted else 'FN'
            else:
                fp+=predicted;tn+=not predicted;row['outcome']='FP' if predicted else 'TN'
        rows.append(row)
    return {'scope':'Selected developer-reviewed real-video cases, not global event precision or an independent benchmark',
        'true_positives':tp,'false_positives':fp,'false_negatives':fn,'true_negatives':tn,
        'case_precision':tp/(tp+fp) if tp+fp else None,'case_recall':tp/(tp+fn) if tp+fn else None,
        'ambiguous_cases':sum(c['kind']=='ambiguous' for c in benchmark['cases']),
        'ambiguous_cases_covered':ambiguous_covered,'covered_ambiguous_cases_remaining_unknown':ambiguous_abstained,
        'cases':rows,'unmeasurable':['Global event precision: non-exhaustive case labels',
            'Positive pickup/carrying/departure/approaching/stationary recall: no positive reviewed examples',
            'ID-switch accuracy: no identity ground truth']}
