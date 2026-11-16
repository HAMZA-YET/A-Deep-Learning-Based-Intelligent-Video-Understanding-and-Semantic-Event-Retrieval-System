"""Controlled association fixtures are algorithm tests, not real carrying ground truth."""
import copy
import pytest
from scenemind_mvp.association.core import AssociationConfig,associate
from scenemind_mvp.association.diagnostics import diagnose

def detection(cls,box,identifier,score=.9):
    return {'class_id':cls,'label':'person' if cls==0 else 'handbag','confidence':score,'bbox_xyxy':box,'track_id':identifier}

def sequence(moving=True,object_tracked=True,second_person=False):
    frames=[]
    for i in range(6):
        x=i*6 if moving else 0
        detections=[detection(0,[x,0,x+100,200],'c0-t1'),
                    detection(26,[x+70,90,x+95,130],'c26-t1' if object_tracked else None)]
        if second_person: detections.append(detection(0,[x+50,0,x+150,200],'c0-t2'))
        frames.append({'frame_index':i,'timestamp':i*.2,'detections':detections})
    return frames

def run(frames):
    return associate(frames,diagnose(frames,5),AssociationConfig())

def test_consistent_motion_is_only_possible_carrying_candidate():
    result=run(sequence())
    assert len(result['events'])==1
    event=result['events'][0]
    assert event['relationship']=='possibly_carrying' and event['state']=='candidate'
    assert 'not probability' in event['score_type']
    assert event['start_timestamp']==0 and event['end_timestamp']==1
    assert len(event['supporting_observations'])==6

def test_stationary_overlap_does_not_establish_carrying():
    assert run(sequence(moving=False))['events'][0]['relationship']=='nearby'

def test_two_people_one_bag_no_arbitrary_winner():
    result=run(sequence(second_person=True))
    assert len(result['events'])==2
    assert {e['person_track_id'] for e in result['events']}=={'c0-t1','c0-t2'}
    assert all(e['relationship']=='unknown' for e in result['events'])
    assert all('multiple_nearby_people' in e['uncertainty_reasons'] for e in result['events'])

def test_duplicate_overlapping_objects_are_uncertain():
    frames=sequence()
    for frame in frames:
        obj=copy.deepcopy(frame['detections'][1]);obj['track_id']='c26-t2'
        frame['detections'].append(obj)
    assert all(e['relationship']=='unknown' for e in run(frames)['events'])

def test_disappearing_object_does_not_extend_evidence():
    frames=sequence()
    for frame in frames[3:]: frame['detections']=frame['detections'][:1]
    event=run(frames)['events'][0]
    assert event['end_timestamp']==.4 and len(event['supporting_observations'])==3

def test_fragmented_person_ids_are_not_merged():
    frames=sequence()
    for frame in frames[3:]: frame['detections'][0]['track_id']='c0-t2'
    events=run(frames)['events']
    assert len(events)==2 and all(e['relationship']=='nearby' for e in events)
    assert {e['person_track_id'] for e in events}=={'c0-t1','c0-t2'}

def test_untracked_objects_never_get_invented_track_ids():
    # With small motion, a geometric evidence chain may persist; it is not an object identity.
    frames=sequence(moving=False,object_tracked=False)
    events=run(frames)['events']
    assert len(events)==1 and events[0]['object_track_id'] is None
    assert events[0]['relationship']=='nearby' and events[0]['state']=='uncertain'
    assert 'object_track_unconfirmed' in events[0]['uncertainty_reasons']

def test_untracked_person_stays_unknown():
    frames=sequence()
    for frame in frames: frame['detections'][0]['track_id']=None
    assert all(e['relationship']=='unknown' for e in run(frames)['events'])

def test_object_without_person_is_explicit_unknown():
    frames=sequence()
    for frame in frames: frame['detections']=frame['detections'][1:]
    events=run(frames)['events']
    assert len(events)==1 and events[0]['person_track_id'] is None and events[0]['relationship']=='unknown'

def test_long_gap_splits_event():
    frames=sequence()
    for frame in frames[3:]: frame['timestamp']+=1
    assert len(run(frames)['events'])==2

def test_low_confidence_prevents_carrying():
    frames=sequence()
    for frame in frames: frame['detections'][1]['confidence']=.3
    assert run(frames)['events'][0]['relationship']=='nearby'

def test_gap_diagnostics():
    frames=sequence();frames[2]['detections']=[]
    diagnostic=diagnose(frames,5)
    track=next(t for t in diagnostic['tracks'] if t['track_id']=='c0-t1')
    assert track['observations']==5 and track['gap_count']==1
    assert track['coverage']==pytest.approx(5/6)
    assert track['mean_detection_confidence']==pytest.approx(.9)

def test_empty_input():
    assert run([])['summary']['candidate_events']==0

def test_out_of_order_timestamps_rejected():
    frames=sequence();frames[1]['timestamp']=0
    with pytest.raises(ValueError): run(frames)

@pytest.mark.parametrize('kwargs',[{'min_observations':1},{'near_distance':-1},{'object_chain_iou':2}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError): AssociationConfig(**kwargs)

def test_diagnostic_distinguishes_active_unexported_state():
    frames=sequence();frames[2]['detections']=[]
    frames[2]['tracker_state']={'active':['c0-t1'],'lost':['c26-t1']}
    diagnostics=diagnose(frames,5)
    assert diagnostics['active_without_exported_observation_track_frames']==1
    assert diagnostics['lost_state_track_frames']==1
    assert diagnostics['identity_switches'] is None

def test_id_handoff_is_candidate_not_verified_switch():
    frames=sequence(moving=False)
    for frame in frames[3:]: frame['detections'][0]['track_id']='c0-t2'
    diagnostics=diagnose(frames,5)
    assert len(diagnostics['high_overlap_id_change_candidates'])==1
    assert diagnostics['identity_switches'] is None
