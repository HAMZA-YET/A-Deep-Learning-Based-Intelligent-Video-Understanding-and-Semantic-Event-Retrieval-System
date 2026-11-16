"""Controlled trajectory fixtures validate rules, not real physical interactions."""
import copy
import pytest
from scenemind_mvp.association.core import associate,AssociationConfig
from scenemind_mvp.association.diagnostics import diagnose
from scenemind_mvp.temporal.core import TemporalConfig,reason_events,search_events
from scenemind_mvp.temporal.evaluation import evaluate_cases

def d(cls,box,identifier):
    return {'class_id':cls,'label':'person' if cls==0 else 'backpack','confidence':.9,'bbox_xyxy':box,'track_id':identifier}

def frames(kind='together',n=30):
    result=[]
    for i in range(n):
        px=0.;ox=75.;oy=100.
        if kind=='together':px=i*3.;ox=px+75
        if kind=='approach':px=-150+i*5.;ox=110
        if kind=='departure':px=-i*4.;ox=75
        if kind=='pickup':
            px=max(0,i-20)*5.;ox=px+75;oy=180 if i<10 else (180-(i-10)*8 if i<20 else 100)
        if kind=='teleport':
            px=max(0,i-20)*5.;ox=px+75;oy=180 if i<15 else 100
        result.append({'frame_index':i,'timestamp':i*.1,'detections':[
            d(0,[px,0,px+100,200],'c0-t1'),d(24,[ox-10,oy-15,ox+10,oy+15],'c24-t1')]})
    return result

def run(rows):
    diagnostic=diagnose(rows,10)
    associations=associate(rows,diagnostic,AssociationConfig())
    return reason_events(rows,associations,diagnostic,TemporalConfig())

def actions(timeline):return {e['action']:e for e in timeline['events'] if e['status']!='unknown'}

def test_moving_together_is_image_plane_observation():
    event=actions(run(frames()))['moving_together']
    assert event['status']=='observed' and event['motion_evidence']['observations']==30
    assert len(event['supporting_observations'])==30
    assert 'image-plane' in event['motion_evidence']['coordinate_scope']

def test_possible_carrying_is_never_confirmed_contact():
    event=actions(run(frames()))['possible_carrying']
    assert event['status']=='candidate' and 'contact_not_verified' in event['reason_codes']

def test_stationary_nearby_not_carrying():
    result=actions(run(frames('static')))
    assert 'stationary_near_object' in result and 'possible_carrying' not in result

def test_approaching_requires_sustained_decrease():
    assert 'approaching' in actions(run(frames('approach')))

def test_departure_requires_visible_stationary_object():
    event=actions(run(frames('departure')))['possible_departure']
    assert event['status']=='candidate' and 'object_remains_observed' in event['reason_codes']

def test_pickup_requires_before_lift_and_after_motion():
    event=actions(run(frames('pickup')))['possible_pickup']
    assert event['status']=='candidate'
    assert event['motion_evidence']['pickup_phase_evidence']['upward_step_fraction']>=.7

def test_teleport_is_not_pickup():
    assert 'possible_pickup' not in actions(run(frames('teleport')))

def test_single_frame_is_unknown():
    assert all(e['status']=='unknown' for e in run(frames(n=1))['events'])

def test_untracked_object_cannot_support_motion_claim():
    rows=frames()
    for row in rows:row['detections'][1]['track_id']=None
    assert all(e['status']=='unknown' for e in run(rows)['events'])

def test_multiple_people_remain_unknown():
    rows=frames()
    for row in rows:
        box=row['detections'][0]['bbox_xyxy'];row['detections'].append(d(0,[box[0]+20,0,box[2]+20,200],'c0-t2'))
    assert all(e['status']=='unknown' for e in run(rows)['events'])

def test_person_fragmentation_not_new_person_or_event():
    rows=frames('static')
    for row in rows[15:]:row['detections'][0]['track_id']='c0-t2'
    result=run(rows)
    assert result['summary']['distinct_people'] is None
    assert all(e['status']=='unknown' for e in result['events'])

def test_object_fragmentation_not_pickup():
    rows=frames('static')
    for row in rows[15:]:row['detections'][1]['track_id']='c24-t2'
    assert all(e['status']=='unknown' for e in run(rows)['events'])

def test_disappearance_not_departure():
    rows=frames('static')
    for row in rows[15:]:row['detections']=row['detections'][:1]
    assert 'possible_departure' not in actions(run(rows))

def test_lift_without_after_evidence_not_pickup():
    assert 'possible_pickup' not in actions(run(frames('pickup',n=20)))

def test_search_action_object_timestamp_and_status():
    timeline=run(frames())
    assert search_events(timeline,'moving_together','backpack',1.,'observed')
    assert search_events(timeline,'possible_pickup')==[]
    assert search_events(timeline,timestamp=100)==[]
    with pytest.raises(ValueError):search_events(timeline,'owner')
    with pytest.raises(ValueError):search_events(timeline,timestamp=-1)

def test_bad_timestamps_rejected():
    rows=frames();rows[1]['timestamp']=0
    with pytest.raises(ValueError):run(rows)

@pytest.mark.parametrize('values',[{'min_observations':1},{'max_gap_seconds':0},{'min_track_coverage':2}])
def test_invalid_config(values):
    with pytest.raises(ValueError):TemporalConfig(**values)

def test_empty_evidence_has_no_invented_events():
    assert run([])['events']==[]

def test_evaluation_abstention_does_not_fake_precision():
    timeline=run(frames());timeline['source_sha256']='test'
    benchmark={'source_sha256':'test','matching_policy':{'min_box_iou':.3,'max_anchor_time_distance':.32,'min_temporal_iou':.1,'accepted_statuses':['observed','candidate']},
        'cases':[{'case_id':'positive','kind':'positive','action':'moving_together','start_timestamp':0,'end_timestamp':2.9,
                  'anchor_timestamp':1.,'person_bbox_xyxy':[30,0,130,200],'object_bbox_xyxy':[95,85,115,115]}]}
    report=evaluate_cases(timeline,benchmark);assert report['case_recall']==1
    for event in timeline['events']:event['status']='unknown';event['action']='unknown'
    report=evaluate_cases(timeline,benchmark)
    assert report['case_precision'] is None and report['case_recall']==0

def test_point_event_and_video_end_playback_bounds():
    from scenemind_mvp.temporal.core import playback_bounds
    assert playback_bounds({'start_timestamp':0.,'end_timestamp':0.},60)==(0,1)
    assert playback_bounds({'start_timestamp':13.16,'end_timestamp':13.16},13.2)==(12,None)
    assert playback_bounds({'start_timestamp':0.,'end_timestamp':.1},.2)==(0,None)
    with pytest.raises(ValueError):playback_bounds({'start_timestamp':2,'end_timestamp':3},2)

def test_decimal_timestamp_search_tolerance():
    timeline={'events':[{'action':'unknown','object_class':'backpack','status':'unknown','start_timestamp':7.5600000000000005,'end_timestamp':7.5600000000000005}]}
    assert len(search_events(timeline,timestamp=7.56))==1
