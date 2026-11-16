"""Explicit vector/evidence fixtures test gates, not pretrained-model accuracy."""
from pathlib import Path
from dataclasses import replace
import copy,json
import numpy as np
import pytest
from scenemind_mvp.search import SearchIndex
from scenemind_mvp.hybrid.query import parse_query
from scenemind_mvp.hybrid.schema import HybridConfig
from scenemind_mvp.hybrid.evidence import Evidence
from scenemind_mvp.hybrid.engine import HybridEngine

class FixtureIndex:
    manifest={'identity':{'video_sha256':'test','config':{'sampling_fps':1}}}
    search_index=SearchIndex([[1,0],[.95,.05],[.9,.1],[.8,.2],[.7,.3]],np.arange(5),backend='numpy')
    def preview(self,index):return Path(f'explicit-test-preview-{index}.jpg')

def detection(cls,identifier,box):return {'class_id':cls,'label':'person' if cls==0 else 'backpack','confidence':.9,'track_id':identifier,'bbox_xyxy':box}

def fixture(two=False):
    frames=[];supports=[]
    for i in range(5):
        objects=[detection(0,'c0-t1',[0,0,100,200]),detection(24,'c24-t1',[70,80,95,120])]
        if two:objects.append(detection(0,'c0-t2',[30,0,130,200]))
        frames.append({'frame_index':i,'timestamp':float(i),'detections':objects})
        supports.append({'frame_index':i,'timestamp':float(i),'person_bbox_xyxy':[0,0,100,200],
            'object_bbox_xyxy':[70,80,95,120],'person_detection_confidence':.9,'object_detection_confidence':.9})
    association={'event_id':'a1','person_track_id':'c0-t1','object_track_id':'c24-t1','object_class':'backpack','object_class_id':24,
        'relationship':'nearby','start_timestamp':0,'end_timestamp':4,'supporting_observations':supports,'uncertainty_reasons':[]}
    associations=[association]
    if two:
        second=copy.deepcopy(association);second['event_id']='a2';second['person_track_id']='c0-t2';associations.append(second)
        for a in associations:a['relationship']='unknown';a['uncertainty_reasons']=['multiple_nearby_people']
    qualities={p:{'coverage':1,'gap_count':0} for p in ('c0-t1','c0-t2','c24-t1')}
    return Evidence(frames,associations,[],qualities,'test',{})

def event(action='possible_carrying',pid='c0-t1',status='candidate'):
    e=fixture();return {'event_id':'event-'+pid,'action':action,'status':status,'person_track_ids':[pid],
        'object_track_id':'c24-t1','object_class':'backpack','start_timestamp':0,'end_timestamp':4,
        'reason_codes':['explicit_fixture'],'supporting_observations':e.associations[0]['supporting_observations']}

def engine(e=None,strict=False):return HybridEngine(FixtureIndex(),e or fixture(),HybridConfig(min_gap_seconds=0,strict_events=strict))

def test_parse_requested_constraints():
    q=parse_query('Find two people carrying a black backpack between 00:01 and 00:04')
    assert q.action=='possible_carrying' and q.object_classes==(24,) and q.colors==('black',)
    assert q.minimum_people==2 and (q.start,q.end)==(1,4)
    assert 'color_not_structurally_verified' in q.unsupported_constraints

def test_leaving_with_is_not_departure_from_object():
    assert parse_query('Show someone leaving with a backpack').action=='leaving_with_object'
    assert 'action_not_modelled:leaving_with_object' in parse_query('Show someone leaving with a backpack').unsupported_constraints

@pytest.mark.parametrize('q',['','  ','Show bag between 10 and 2','Show bag at 99:99','Show bag after 2 minutes','Show person after -2 seconds'])
def test_invalid_queries(q):
    with pytest.raises(ValueError):parse_query(q)

def test_walking_standing_and_color_not_invented():
    r=engine().search('Show two people standing together with a black bag',[[1,0]])
    assert all(row['event_claim'] is None for row in r['results'])
    assert r['status']=='uncertain'

def test_multiple_people_supported_for_same_geometric_action():
    e=fixture(two=True);e.events=[event('approaching','c0-t1','observed'),event('approaching','c0-t2','observed')]
    r=engine(e).search('Show people approaching a backpack',[[1,0]])
    assert r['all_supported_action_track_ids']==['c0-t1','c0-t2']
    assert r['results'][0]['evidence']['supported_action_person_track_ids']==['c0-t1','c0-t2']

def test_two_people_near_bag_no_exclusive_assignment():
    r=engine(fixture(two=True)).search('Find all people near a bag',[[1,0]])
    assert r['all_spatial_candidate_track_ids']==['c0-t1','c0-t2']
    assert r['results'][0]['evidence']['spatial_candidate_person_track_ids']==['c0-t1','c0-t2']
    assert all(row['event_claim'] is None for row in r['results'])

def test_ambiguous_association_blocks_interaction_even_with_stale_temporal_candidate():
    e=fixture(two=True);e.events=[event()]
    r=engine(e).search('Find someone carrying a bag',[[1,0]])
    assert all(row['event_claim'] is None for row in r['results'])
    assert 'competing_association_candidates_prevent_interaction_claim' in r['results'][0]['uncertainty']

def test_bag_disappears_behind_person_not_pickup():
    e=fixture();e.frames[2]['detections']=e.frames[2]['detections'][:1];e.events=[event('possible_pickup')]
    row=engine(e).search('Find a person picking up a bag',[[1,0]],candidate_timestamps=[2])['results'][0]
    assert row['event_claim'] is None and not row['evidence']['objects']

def test_missing_object_detection_not_negative_proof():
    e=fixture()
    for frame in e.frames:frame['detections']=frame['detections'][:1]
    r=engine(e).search('Find someone carrying a bag',[[1,0]])
    assert r['status']=='uncertain'
    assert all('requested_object_not_detected_at_sample' in row['uncertainty'] for row in r['results'])
    assert all(row['event_claim'] is None for row in r['results'])

def test_fragmented_ids_remain_distinct():
    e=fixture()
    for frame in e.frames[2:]:frame['detections'][0]['track_id']='c0-t2'
    e.associations[0]['supporting_observations']=e.associations[0]['supporting_observations'][:2]
    second=copy.deepcopy(e.associations[0]);second.update(event_id='fragment',person_track_id='c0-t2',start_timestamp=2,end_timestamp=4)
    second['supporting_observations']=[{'frame_index':i,'timestamp':i} for i in (2,3,4)];e.associations.append(second)
    r=engine(e).search('Find all people near a bag',[[1,0]])
    assert r['all_spatial_candidate_track_ids']==['c0-t1','c0-t2']
    assert 'distinct_people' not in r

def test_no_valid_event_strict_and_uncertain_modes():
    assert engine(strict=True).search('Find person picking up bag',[[1,0]])['results']==[]
    assert engine().search('Find person picking up bag',[[1,0]])['status']=='uncertain'

def test_conflicting_high_clip_and_stationary_evidence():
    e=fixture();e.events=[event('stationary_near_object',status='observed')]
    row=engine(e).search('Find person picking up bag',[[1,0]],candidate_timestamps=[0])['results'][0]
    assert row['clip_cosine']==pytest.approx(1) and row['event_claim'] is None
    assert row['incompatible_motion_penalty_applied']

def test_event_candidate_is_not_physical_confirmation():
    e=fixture();e.events=[event()]
    row=engine(e).search('Find someone carrying a bag',[[1,0]])['results'][0]
    assert row['state']=='candidate_event_evidence' and row['event_claim']['status']=='candidate'
    assert 'not probability' in row['score_type']

def test_color_constraint_prevents_event_claim():
    e=fixture();e.events=[event()]
    assert all(row['event_claim'] is None for row in engine(e).search('Find someone carrying a black bag',[[1,0]])['results'])

def test_time_filters_and_no_results():
    r=engine().search('Show a person after 1 before 3 seconds',[[1,0]])
    assert [row['timestamp'] for row in r['results']]==[2]
    assert engine().search('Show a person after 99 seconds',[[1,0]])['status']=='no_reliable_match'
    assert engine().search('Show person at 0 seconds',[[1,0]])['results'][0]['timestamp']==0

def test_clip_only_does_not_apply_hybrid_gates_or_claims():
    r=engine(strict=True).search('Find person picking up bag',[[1,0]],mode='clip_only')
    assert len(r['results'])==5 and all(row['event_claim'] is None for row in r['results'])
    assert r['results'][0]['rank_score']==pytest.approx(1)

def test_negation_not_silently_confirmed():
    e=fixture();e.events=[event()]
    assert all(row['event_claim'] is None for row in engine(e).search('Find a person not carrying a bag',[[1,0]])['results'])

def test_config_and_source_validation():
    with pytest.raises(ValueError):HybridConfig(semantic_weight=.2)
    with pytest.raises(ValueError):HybridEngine(FixtureIndex(),Evidence.empty('different'),HybridConfig())


def test_aggregate_ids_obey_interaction_gate():
    e=fixture(two=True);e.events=[event()]
    result=engine(e).search('Find someone carrying a bag',[[1,0]])
    assert result['all_supported_action_track_ids']==[] and result['all_candidate_action_track_ids']==[]
    assert all(not r['admissible_for_query'] for r in result['complete_structured_track_intervals'])


def test_negation_does_not_boost_positive_event_component():
    e=fixture();e.events=[event()]
    result=engine(e).search('Find person not carrying a bag',[[1,0]])
    assert all(row['score_components']['event']==0 for row in result['results'])

def test_missing_object_at_requested_time_has_no_action_ids():
    e=fixture();e.frames[2]['detections']=e.frames[2]['detections'][:1];e.events=[event('possible_pickup')]
    result=engine(e).search('Find person picking up a bag at 2 seconds',[[1,0]])
    assert result['all_supported_action_track_ids']==[] and result['all_candidate_action_track_ids']==[]


def test_full_track_intervals_include_observation_evidence():
    r=engine(fixture(two=True)).search('Find all people near a bag',[[1,0]])
    assert r['track_scope']=='test'
    assert all(record['supporting_observations'] for record in r['complete_structured_track_intervals'])

def test_two_explicit_ids_cannot_be_satisfied_by_one_action_track():
    e=fixture(two=True);e.associations=[];e.events=[event('approaching','c0-t1','observed')]
    r=engine(e).search('Show c0-t1 and c0-t2 approaching a backpack',[[1,0]])
    assert all(row['event_claim'] is None for row in r['results'])
