"""Explicit fixtures; not human-labelled action accuracy."""
from copy import deepcopy
import pytest
from scenemind_mvp.unified.core import make_plan,matched_indices,action_at,match_summary
from scenemind_mvp.unified.actions import supported_intervals

def scene():
    detections=[{'class_id':0,'track_id':'c0-t1','label':'person','bbox_xyxy':[0,0,20,80]},{'class_id':0,'track_id':'c0-t2','label':'person','bbox_xyxy':[30,0,50,80]},{'class_id':24,'track_id':None,'label':'backpack','bbox_xyxy':[33,20,43,40]}]
    intervals=[{'person_track_id':t,'action':'walking','start':1.,'end':3.,'status':'model_supported','score':.9} for t in ['c0-t1','c0-t2']]
    return {'metadata':{'fps':30},'frames':[{'frame_index':30,'timestamp':1.,'detections':detections}],'actions':{'intervals':intervals}}

def test_no_query_shows_all_and_never_needs_clip():
    a=scene();p=make_plan('',a);assert matched_indices(a['frames'][0],p,a)==[0,1,2]

def test_all_walking_highlights_two_independent_people():
    a=scene();p=make_plan('Show all walking people',a)
    assert matched_indices(a['frames'][0],p,a)==[0,1]
    assert match_summary(p,a)['track_ids']==['c0-t1','c0-t2']

def test_explicit_person_reference_does_not_merge_people():
    a=scene();p=make_plan('Show Person 2',a);assert matched_indices(a['frames'][0],p,a)==[1]
    assert matched_indices(a['frames'][0],make_plan('Show Person 99',a),a)==[]

@pytest.mark.parametrize('query',['Show all sitting people','Show all running people','Person carrying a black bag','a person wearing white','not walking people'])
def test_unsupported_or_contradictory_evidence_abstains(query):
    a=scene();assert matched_indices(a['frames'][0],make_plan(query,a),a)==[]

def test_missing_bag_does_not_create_relationship():
    a=scene();a['frames'][0]['detections']=a['frames'][0]['detections'][:2]
    assert not matched_indices(a['frames'][0],make_plan('a person carrying a bag',a),a)

def test_actions_do_not_extend_to_unobserved_time_or_tracks():
    a=scene();f=deepcopy(a['frames'][0]);f['timestamp']=4.
    assert not matched_indices(f,make_plan('walking people',a),a)
    f['timestamp']=1.;f['detections'][0]['track_id']='c0-t3'
    assert matched_indices(f,make_plan('walking people',a),a)==[1]

def test_conflicting_postures_unknown_but_carrying_can_coexist():
    a=scene();e=deepcopy(a['actions']['intervals'][0]);e['action']='carrying';a['actions']['intervals'].append(e)
    assert action_at(a['actions']['intervals'],'c0-t1',2.)['action']=='walking'
    e=deepcopy(e);e['action']='sitting';a['actions']['intervals'].append(e)
    assert action_at(a['actions']['intervals'],'c0-t1',2.) is None

def observation(t=1.,action='walking'):
    return {'track_id':'c0-t1','timestamp':t,'frame_index':round(t*30),'candidate_action':action,'scores':{action:.9},'reasons':[],'clip_frame_range':[0,64]}

def test_two_temporal_windows_required_no_single_frame_confirmation():
    config={'interval_seconds':1.}
    assert not supported_intervals([observation()],config,[])
    assert supported_intervals([observation(),observation(2.)],config,[])[0]['verified'] is False

def test_gaps_unknown_or_unconfirmed_tracks_do_not_support_action():
    c={'interval_seconds':1.};left=observation();right=observation(2.);right['reasons']=['insufficient_unique_track_coverage']
    assert not supported_intervals([left,right],c,[])
    assert not supported_intervals([left,observation(4.)],c,[])

def test_carrying_requires_object_track_and_nonambiguous_temporal_evidence():
    o=[observation(1.,'carrying'),observation(2.,'carrying')];c={'interval_seconds':1.}
    assert not supported_intervals(o,c,[])
    event={'action':'possible_carrying','status':'candidate','person_track_ids':['c0-t1'],'object_track_id':'c24-t1','object_class':'backpack','start_timestamp':0.,'end_timestamp':3.,'reason_codes':[]}
    assert supported_intervals(o,c,[event])[0]['object_evidence'][0]['track_id']=='c24-t1'
    event['reason_codes']=['multiple_nearby_people'];assert not supported_intervals(o,c,[event])

def test_time_constraint_and_object_categories():
    a=scene();assert not matched_indices(a['frames'][0],make_plan('walking people after 2 seconds',a),a)
    assert matched_indices(a['frames'][0],make_plan('a backpack',a),a)==[2]
