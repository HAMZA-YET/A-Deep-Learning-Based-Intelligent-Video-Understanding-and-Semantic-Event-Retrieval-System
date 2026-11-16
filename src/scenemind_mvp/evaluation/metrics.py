"""Explicit denominators and label eligibility for offline evaluation."""
from __future__ import annotations
import math
from typing import Any
from ..association.core import box_iou

def retrieval_metrics(timestamps: list[float], intervals: list[list[float]] | None, verdict: str, k: int=5, tolerance: float=.5) -> dict[str, Any]:
    """Positive-query retrieval metrics; negatives/unknowns have separate outcomes."""
    if k<1 or tolerance<0:raise ValueError('Invalid metric configuration')
    known=verdict in ('positive','negative')
    positive=verdict=='positive'
    if known and intervals is None:raise ValueError('Reviewed cases need intervals')
    if positive and not intervals:raise ValueError('Positive case needs relevant intervals')
    if verdict=='negative' and intervals:raise ValueError('Negative case cannot have relevant intervals')
    hits=[any(a-tolerance<=t<b+tolerance for a,b in intervals or []) for t in timestamps[:k]]
    first=next((i+1 for i,hit in enumerate(hits) if hit),None)
    return {'hit_at_1':bool(hits and hits[0]) if positive else None,'hit_at_5':any(hits) if positive else None,'precision_at_5':sum(hits)/k if positive else None,'mrr':1/first if positive and first else (0. if positive else None),'returned':len(timestamps[:k]),'abstained':not timestamps,'negative_returned_candidates':len(timestamps[:k]) if verdict=='negative' else None,'label_status':verdict}

def validate_review(review: dict, sha: str, duration: float) -> bool:
    if review['source_sha256']!=sha:raise ValueError('Annotation/video hash mismatch')
    if not math.isclose(review['duration'],duration,abs_tol=.001):raise ValueError('Annotation duration mismatch')
    if review['review_status']!='complete':return False
    if not review.get('reviewer') or not review.get('reviewed_utc') or review.get('independent_of_model_predictions') is not True:raise ValueError('Human review must be signed and model independent')
    for query in review['queries']:
        verdict=query['verdict'];intervals=query['relevant_intervals']
        if verdict not in ('positive','negative','ambiguous'):raise ValueError('Incomplete verdict')
        if verdict=='positive' and not intervals:raise ValueError('Missing positive intervals')
        if verdict=='negative' and intervals!=[]:raise ValueError('Negative needs empty intervals')
        if verdict=='ambiguous' and intervals is not None:raise ValueError('Ambiguous relevance must be null')
        for a,b in intervals or []:
            if not all(math.isfinite(t) for t in (a,b)) or not 0<=a<b<=duration:raise ValueError('Invalid annotation interval')
    for frame in review.get('detection_frames',[]):
        if not isinstance(frame['frame_index'],int) or frame['frame_index']<0:raise ValueError('Invalid annotation frame index')
        if not 0<=frame['timestamp']<duration:raise ValueError('Invalid frame timestamp')
        for box in frame['boxes']:
            coordinates=box['bbox_xyxy']
            if len(coordinates)!=4 or not all(math.isfinite(x) for x in coordinates) or not 0<=coordinates[0]<coordinates[2] or not 0<=coordinates[1]<coordinates[3]:raise ValueError('Invalid annotation bounding box')
            if not isinstance(box['class_id'],int) or not 0<=box['class_id']<80:raise ValueError('Invalid annotation class')
    return True

def match_boxes(truth: list[dict], predictions: list[dict], threshold: float=.5) -> list[tuple[int,int]]:
    """Maximum-cardinality class-aware IoU matching, with IoU-sorted augmenting paths."""
    links={i:sorted([j for j,p in enumerate(predictions) if g['class_id']==p['class_id'] and box_iou(g['bbox_xyxy'],p['bbox_xyxy'])>=threshold],key=lambda j:box_iou(g['bbox_xyxy'],predictions[j]['bbox_xyxy']),reverse=True) for i,g in enumerate(truth)}
    owners={}
    def augment(i,seen):
        for j in links[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owners or augment(owners[j],seen):owners[j]=i;return True
        return False
    for i in links:augment(i,set())
    return sorted((i,j) for j,i in owners.items())

def detection_metrics(annotations: list[dict], frames: list[dict], threshold: float=.5) -> dict:
    by_frame={f['frame_index']:f for f in frames};tp=fp=fn=eligible=0
    for row in annotations:
        classes=row.get('exhaustive_classes',[])
        if not row.get('reviewed') or not classes:continue
        frame=by_frame.get(row['frame_index'])
        if frame is None:continue # unsampled is not a detector miss
        truth=[b for b in row['boxes'] if b['class_id'] in classes];pred=[b for b in frame['detections'] if b['class_id'] in classes]
        count=len(match_boxes(truth,pred,threshold));tp+=count;fp+=len(pred)-count;fn+=len(truth)-count;eligible+=1
    return {'eligible_frames':eligible,'tp':tp if eligible else None,'fp':fp if eligible else None,'missed_objects':fn if eligible else None,'precision':tp/(tp+fp) if eligible and tp+fp else None,'recall':tp/(tp+fn) if eligible and tp+fn else None,'scope':'exhaustively reviewed classes at processed frames only'}

def tracking_metrics(sequences: list[dict], frames: list[dict], threshold: float=.5) -> dict:
    by_frame={f['frame_index']:f for f in frames};switches=fragments=matched=total=eligible=0
    for sequence in sequences:
        rows=sequence.get('frames',[])
        if not sequence.get('exhaustive') or len(rows)<2:continue
        if any(b['frame_index']!=a['frame_index']+1 for a,b in zip(rows,rows[1:])):continue
        if any(r['frame_index'] not in by_frame for r in rows):continue
        eligible+=1;last_ids={};last_seen={};had_match=set();missing=set()
        for row in rows:
            truth=[b for b in row['boxes'] if b['class_id']==0 and b.get('person_reference')]
            pred=[b for b in by_frame[row['frame_index']]['detections'] if b['class_id']==0 and b.get('track_id')]
            pairs=dict(match_boxes(truth,pred,threshold));total+=len(truth)
            visible={b['person_reference'] for b in truth}
            for i,g in enumerate(truth):
                ident=g['person_reference']
                if i not in pairs:
                    if ident in had_match:missing.add(ident)
                    continue
                matched+=1;pid=pred[pairs[i]]['track_id']
                # ID changes only across adjacent visible labelled frames; uncertain gaps excluded.
                if ident in last_ids and last_seen.get(ident)==row['frame_index']-1 and last_ids[ident]!=pid:switches+=1
                if ident in missing:fragments+=1;missing.remove(ident)
                last_ids[ident]=pid;last_seen[ident]=row['frame_index'];had_match.add(ident)
            for ident in set(last_seen)-visible:last_seen.pop(ident,None);missing.discard(ident)
    return {'eligible_continuous_sequences':eligible,'adjacent_visible_id_switches':switches if eligible else None,'matched_lost_matched_fragments':fragments if eligible else None,'matched_visible_observation_fraction':matched/total if eligible and total else None,'idf1':None,'mota':None,'hota':None,'note':'Scoped visible-frame diagnostics, not standard MOT metrics; uncertain identity gaps excluded.'}

def temporal_case_metrics(cases: list[dict], events: list[dict], threshold: float=.3) -> dict:
    counts={'tp':0,'fn':0,'negative_cases_with_predictions':0,'tn':0,'ambiguous_cases':0,'ambiguous_abstentions':0};eligible=0
    for case in cases:
        if case.get('verdict') not in ('positive','negative','ambiguous'):continue
        eligible+=1;a,b=case['start_timestamp'],case['end_timestamp']
        def overlaps(e):
            lo=max(a,e['start_timestamp']);hi=min(b,e['end_timestamp']);intersection=max(0.,hi-lo)
            union=max(b,e['end_timestamp'])-min(a,e['start_timestamp'])
            return intersection/union>=threshold if union else False
        matches=[e for e in events if e['status'] in ('candidate','observed') and e['action']==case['action'] and (not case.get('object_category') or e['object_class']==case['object_category']) and overlaps(e)]
        if case['verdict']=='positive':counts['tp' if matches else 'fn']+=1
        elif case['verdict']=='negative':counts['negative_cases_with_predictions' if matches else 'tn']+=1
        else:counts['ambiguous_cases']+=1;counts['ambiguous_abstentions']+=not matches
    return {**(counts if eligible else {k:None for k in counts}),'eligible_cases':eligible,'case_recall':counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else None,'case_precision':counts['tp']/(counts['tp']+counts['negative_cases_with_predictions']) if counts['tp']+counts['negative_cases_with_predictions'] else None,'event_precision':None,'note':'Non-exhaustive scoped interval/action cases; person-specific event accuracy and global precision require exhaustive subject-resolved labels.'}
