"""Metric tests use explicit fixtures, never report them as video benchmark truth."""
import pytest
from scenemind_mvp.evaluation.metrics import retrieval_metrics,validate_review,match_boxes,detection_metrics,tracking_metrics,temporal_case_metrics
from scenemind_mvp.evaluation.retrieval import structured_search
from scenemind_mvp.hybrid.evidence import Evidence

def box(x=0,cls=0,track=None,person=None):return {'class_id':cls,'bbox_xyxy':[x,0,x+10,10],'track_id':track,'person_reference':person}
def test_retrieval_denominators_and_mrr():
    m=retrieval_metrics([0,2,4],[[2,3]],'positive',tolerance=0)
    assert not m['hit_at_1'] and m['hit_at_5'] and m['precision_at_5']==.2 and m['mrr']==.5
    assert retrieval_metrics([],[[2,3]],'positive')['mrr']==0
@pytest.mark.parametrize('verdict',['pending','ambiguous','negative'])
def test_unlabelled_or_negative_not_accuracy(verdict):
    m=retrieval_metrics([1],[] if verdict=='negative' else None,verdict)
    assert m['hit_at_1'] is None and m['mrr'] is None

def test_timestamp_tolerance_is_predefined():
    assert retrieval_metrics([.5],[[1,2]],'positive')['hit_at_1']
    assert not retrieval_metrics([2.5],[[1,2]],'positive')['hit_at_1']

def test_review_requires_human_signature_and_hash():
    r={'source_sha256':'a','duration':3,'review_status':'pending'}
    assert not validate_review(r,'a',3)
    with pytest.raises(ValueError):validate_review(r,'wrong',3)
    r['review_status']='complete'
    with pytest.raises(ValueError):validate_review(r,'a',3)

def test_review_intervals_are_validated():
    r={'source_sha256':'a','duration':3,'review_status':'complete','reviewer':'unit fixture','reviewed_utc':'2026-01-01T00:00:00Z','independent_of_model_predictions':True,'queries':[{'verdict':'positive','relevant_intervals':[[0,4]]}]}
    with pytest.raises(ValueError):validate_review(r,'a',3)

def test_detection_exhaustive_scope_and_duplicate_predictions():
    gt=[{'frame_index':0,'reviewed':True,'exhaustive_classes':[0],'boxes':[box()]}]
    frames=[{'frame_index':0,'detections':[box(),box(),box(cls=24)]}]
    m=detection_metrics(gt,frames);assert m['tp']==1 and m['fp']==1 and m['missed_objects']==0 and m['precision']==.5
    assert detection_metrics([],frames)['precision'] is None

def test_detection_missing_frame_is_not_detector_false_negative():
    m=detection_metrics([{'frame_index':1,'reviewed':True,'exhaustive_classes':[0],'boxes':[box()]}],[])
    assert m['eligible_frames']==0 and m['missed_objects'] is None

def test_class_matching_does_not_match_bag_to_person():
    assert match_boxes([box(cls=24)],[box(cls=0)])==[]

def test_tracking_switches_and_fragments_require_dense_labels():
    truth=[{'frame_index':i,'boxes':[box(person='manual1')]} for i in range(4)]
    seq=[{'exhaustive':True,'frames':truth}]
    pred=[{'frame_index':0,'detections':[box(track='c0-t1')]},{'frame_index':1,'detections':[box(track='c0-t2')]},{'frame_index':2,'detections':[]},{'frame_index':3,'detections':[box(track='c0-t3')]}]
    m=tracking_metrics(seq,pred);assert m['adjacent_visible_id_switches']==1 and m['matched_lost_matched_fragments']==1 and m['matched_visible_observation_fraction']==.75
    assert tracking_metrics([{'exhaustive':True,'frames':[truth[0],truth[3]]}],pred)['eligible_continuous_sequences']==0
    assert m['idf1'] is None

def test_temporal_unknown_abstains():
    cases=[{'verdict':v,'action':'possible_pickup','start_timestamp':0,'end_timestamp':2} for v in ('positive','negative','ambiguous')]
    events=[{'status':'unknown','action':'possible_pickup','start_timestamp':0,'end_timestamp':2,'object_class':'backpack'}]
    m=temporal_case_metrics(cases,events);assert m['fn']==1 and m['tn']==1 and m['ambiguous_abstentions']==1 and m['event_precision'] is None

def test_structured_search_never_uses_clip_or_unknown_event():
    evidence=Evidence.empty('a');evidence.events=[{'event_id':'e','action':'possible_pickup','object_class':'backpack','person_track_ids':['c0-t1'],'status':'unknown','start_timestamp':0,'end_timestamp':2}]
    assert not structured_search('Find a person picking up a bag',evidence,[0.,1.])['results']
    evidence.events[0]['status']='candidate'
    rows=structured_search('Find a person picking up a bag',evidence,[0.,1.])['results'];assert rows and rows[0]['rank_score'] is None and rows[0]['event_claim']['status']=='candidate'
    assert not structured_search('Find someone carrying a black bag',evidence,[0.,1.])['results']
