import json
from pathlib import Path
from copy import deepcopy
import pytest
from scenemind_mvp.recovery.annotation import add_box,add_interval,label_query
from scenemind_mvp.evaluation.metrics import validate_review

def blank():return {'duration':3.,'source_sha256':'fixture','review_status':'pending','reviewer':None,'detection_frames':[],'queries':[{'query':'fixture query','verdict':'pending','relevant_intervals':None}],'events':[]}
def test_manual_box_has_no_predicted_confidence_or_track_id():
    original=blank();new=add_box(original,1,25,100,100,24,[1,2,20,30],'human-b1')
    assert not original['detection_frames'];box=new['detection_frames'][0]['boxes'][0]
    assert box=={'class_id':24,'bbox_xyxy':[1,2,20,30],'person_reference':None,'object_reference':'human-b1'}
    assert not new['detection_frames'][0]['reviewed'] and new['review_status']=='pending'
@pytest.mark.parametrize('coords',[[0,0,0,1],[-1,0,2,2],[0,0,101,10],[0,0,float('nan'),2]])
def test_invalid_manual_boxes_rejected(coords):
    with pytest.raises(ValueError):add_box(blank(),0,25,100,100,24,coords)
def test_manual_events_and_relationships_remain_explicit():
    review=add_interval(blank(),'relationships','unknown','ambiguous',0,1,person='human-p1')
    assert review['relationships'][0]['verdict']=='ambiguous' and review['review_status']=='pending'
    with pytest.raises(ValueError):add_interval(blank(),'relationships','ownership','positive',0,1)
    with pytest.raises(ValueError):add_interval(blank(),'events','ownership','positive',0,1)
def test_query_labels_require_explicit_interval_and_reset_signature():
    r=blank();r.update(review_status='complete',reviewer='fixture')
    with pytest.raises(ValueError):label_query(r,0,'positive')
    out=label_query(r,0,'positive',1,2);assert out['queries'][0]['relevant_intervals']==[[1,2]] and out['reviewer'] is None
    out=label_query(out,0,'ambiguous');assert out['queries'][0]['relevant_intervals'] is None
    assert label_query(out,0,'negative')['queries'][0]['relevant_intervals']==[]
def test_named_human_completion_still_required():
    r=label_query(blank(),0,'negative');assert not validate_review(r,'fixture',3)
def test_frozen_development_scope_and_profiles():
    p=json.loads(Path('benchmarks/phase27/protocol.json').read_text())
    assert len(p['detectors'])==5 and p['detectors']['baseline640']['confidence']==.25
    assert 'closed' in p['selection_policy']

def test_annotation_interface_source_only_and_no_original_write():
    from streamlit.testing.v1 import AppTest
    original=Path('benchmarks/phase26/sabana_grande-review.json').read_bytes()
    app=AppTest.from_file(str(Path('scripts/phase27/review_app.py').resolve()),default_timeout=30).run()
    assert not app.exception
    assert app.session_state['review-sabana_grande']['review_status']=='pending'
    inputs={i.label:i for i in app.number_input}
    for label,value in [('x1',10.),('y1',10.),('x2',50.),('y2',50.)]:inputs[label].set_value(value)
    next(b for b in app.button if b.label=='Add human box').click().run()
    assert not app.exception
    assert len(app.session_state['review-sabana_grande']['detection_frames'][0]['boxes'])==1
    next(b for b in app.button if b.label=='Validate and sign completed review').click().run()
    assert app.session_state['review-sabana_grande']['review_status']=='pending' and app.error
    assert Path('benchmarks/phase26/sabana_grande-review.json').read_bytes()==original

