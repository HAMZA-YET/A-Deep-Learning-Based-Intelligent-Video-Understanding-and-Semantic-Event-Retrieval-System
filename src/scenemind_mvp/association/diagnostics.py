"""Tracklet diagnostics from recorded detections, without identity ground truth."""
import numpy as np

def diagnose(frames: list[dict], fps: float, stride: int = 1) -> dict:
    interval=stride/fps
    histories={}; observations=0; unassigned=0; counts={}
    for frame in frames:
        for detection in frame['detections']:
            observations+=1
            counts[detection['label']]=counts.get(detection['label'],0)+1
            identifier=detection['track_id']
            if identifier is None:
                unassigned+=1; continue
            histories.setdefault(identifier,[]).append((frame['timestamp'],detection))
    active_without_observation=0;lost_state_frames=0;handoffs=[]
    from .core import box_iou
    previous=[]
    for frame in frames:
        visible={d['track_id'] for d in frame['detections'] if d['track_id']}
        states=frame.get('tracker_state',{})
        active_without_observation+=len(set(states.get('active',[]))-visible)
        lost_state_frames+=len(set(states.get('lost',[])))
        current=[d for d in frame['detections'] if d['class_id']==0 and d['track_id']]
        old_ids={d['track_id'] for d in previous};new_ids={d['track_id'] for d in current}
        for old in previous:
            if old['track_id'] in new_ids: continue
            for new in current:
                if new['track_id'] in old_ids: continue
                overlap=box_iou(old['bbox_xyxy'],new['bbox_xyxy'])
                if overlap>=.7:
                    handoffs.append({'frame_index':frame['frame_index'],'timestamp':frame['timestamp'],
                        'previous_track_id':old['track_id'],'new_track_id':new['track_id'],'box_iou':overlap})
        previous=current
    tracks=[]
    for identifier,items in histories.items():
        times=np.array([x[0] for x in items]); conf=np.array([x[1]['confidence'] for x in items])
        gaps=np.diff(times)
        gap_seconds=[float(gap-interval) for gap in gaps if gap>1.5*interval]
        span=float(times[-1]-times[0]+interval)
        expected=max(1,round(span/interval))
        tracks.append({'track_id':identifier,'class_id':items[0][1]['class_id'],
            'label':items[0][1]['label'],'observations':len(items),'span_seconds':span,
            'observed_seconds':len(items)*interval,'coverage':min(1.,len(items)/expected),
            'mean_detection_confidence':float(conf.mean()),'min_detection_confidence':float(conf.min()),
            'gap_count':len(gap_seconds),'missing_seconds':sum(gap_seconds),'gap_seconds':gap_seconds,
            'first_timestamp':float(times[0]),'last_timestamp':float(times[-1])})
    persons=[t for t in tracks if t['class_id']==0]
    def median(key): return float(np.median([t[key] for t in persons])) if persons else None
    return {'detection_observations':observations,'unassigned_observations':unassigned,
        'assignment_fraction':(observations-unassigned)/observations if observations else None,
        'detections_by_class':counts,'person_tracklets':len(persons),
        'median_person_observations':median('observations'),'median_person_span_seconds':median('span_seconds'),
        'median_person_coverage':median('coverage'),'median_person_mean_confidence':median('mean_detection_confidence'),
        'short_person_tracklets_under_half_second':sum(t['span_seconds']<.5 for t in persons),
        'person_tracklets_with_gaps':sum(t['gap_count']>0 for t in persons),
        'total_person_missing_seconds':sum(t['missing_seconds'] for t in persons),
        'active_without_exported_observation_track_frames':active_without_observation,
        'lost_state_track_frames':lost_state_frames,'high_overlap_id_change_candidates':handoffs,
        'gap_interpretation':'Gaps are in exported observations; low-confidence active tracks may remain internal.',
        'identity_switches':None,'identity_switch_note':'No identity ground truth; tracklet counts/gaps do not measure ID switches.',
        'tracks':tracks}
