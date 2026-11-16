import pytest
from scenemind_mvp.ui import state
from scenemind_mvp.ui.presentation import label_result,timestamp

def test_new_submission_clears_all_outputs_before_work():
    s={'sm_video':'v','sm_mode':'Regions','matches':[1],'hybrid_response':{'old':True},'sm_overlay':{'old':True}}
    state.bind_context(s,'v/reg');first=state.begin(s,'v/reg','walking');state.complete(s,first,'v/reg',{'results':[{'timestamp':1}]});state.select(s,first,0)
    second=state.begin(s,'v/reg','sitting')
    assert first!=second and s[state.KEY]['result'] is None and s[state.KEY]['selected'] is None
    assert s[state.KEY]['status']=='pending' and s['sm_video']=='v' and s['sm_mode']=='Regions'
    assert 'matches' not in s and 'hybrid_response' not in s and 'sm_overlay' not in s

def test_late_completion_cannot_restore_previous_query_or_video():
    s={};state.bind_context(s,'first');old=state.begin(s,'first','old');latest=state.begin(s,'first','new')
    assert not state.complete(s,old,'first',{'results':['stale']})
    assert state.complete(s,latest,'first',{'results':[{'timestamp':2}]})
    state.bind_context(s,'second');assert not state.complete(s,latest,'first',{'results':['wrong video']})
    assert not state.select(s,latest,0) and s[state.KEY]['result'] is None

def test_typing_or_repeated_context_does_not_clear_results():
    s={};state.bind_context(s,'same');id=state.begin(s,'same','walking');state.complete(s,id,'same',{'results':[]});s['sm_query']='sitting';state.bind_context(s,'same')
    assert s[state.KEY]['query']=='walking' and s[state.KEY]['status']=='done'

def test_failed_query_does_not_fall_back_to_old_result():
    s={};state.bind_context(s,'same');id=state.begin(s,'same','new');state.fail(s,id,'same','failed')
    assert s[state.KEY]['result'] is None and s[state.KEY]['status']=='error'

def test_model_output_never_receives_verified_or_demographic_description():
    for row,query in [({'event_claim':{'status':'observed'}},{'action':'moving_together'}),({'event_claim':{'status':'candidate'}},{'action':'possible_carrying'}),({}, {'action':'walking'}),({}, {'action':None})]:
        label=label_result(row,query);assert label.status!='Verified' and 'woman' not in label.description and 'walking' not in label.description

def test_nearby_evidence_is_not_ownership():
    label=label_result({'evidence':{'associations':[{'relationship':'nearby'}]}},{'action':'nearby'})
    assert label.status=='Uncertain' and label.kind=='Spatial evidence' and 'unverified' in label.description
    unknown=label_result({'evidence':{'associations':[{'relationship':'unknown'}]}},{'action':'nearby'})
    assert unknown.status=='Uncertain' and unknown.kind=='Visual similarity' and 'unverified' in unknown.description

@pytest.mark.parametrize('seconds,expected',[(0,'00:00'),(5,'00:05'),(10.4,'00:10.4'),(65,'01:05'),(3601,'01:00:01')])
def test_readable_timestamps(seconds,expected):assert timestamp(seconds)==expected


def test_late_failure_cannot_replace_a_completed_result():
    s={};state.bind_context(s,'same');id=state.begin(s,'same','query');state.complete(s,id,'same',{'results':[]})
    assert not state.fail(s,id,'same','late failure') and s[state.KEY]['status']=='done'
