from pathlib import Path
from copy import deepcopy
import json
from streamlit.testing.v1 import AppTest
import scenemind_mvp.ui.service as service
from scenemind_mvp.ui import state
ROOT=Path(__file__).resolve().parents[2]
CARACAS='1185167f220d6e5a2e1889e59bd81c162b5f75c819df5448a5b8b3b65a2dfa62'
FIT='1e66534958801d703894bed147f30f2a33b59d0eab875056e80211a943362dfe'

def submit(app,text):
    app.text_input(key='sm_query').set_value(text);app.button(key='sm_search_submit').click().run();assert not app.exception
    return deepcopy(app.session_state[state.KEY])

def test_real_main_ui_latest_only_modes_sources_and_playback(monkeypatch):
    original=service.search;observed=[]
    def checked(*args,**kwargs):
        import streamlit as st
        run=st.session_state[state.KEY];assert run['status']=='pending' and run['result'] is None and run['selected'] is None
        assert not any(k in st.session_state for k in state.LEGACY);observed.append(run['id'])
        if args[2]=='__error_fixture__':raise RuntimeError('Injected request failure fixture')
        return original(*args,**kwargs)
    monkeypatch.setattr(service,'search',checked)
    def forbid(*args,**kwargs):raise AssertionError('Compatible saved indexes must not be recomputed')
    monkeypatch.setattr(service,'index_video',forbid);monkeypatch.setattr(service,'prepare_bundle',forbid);monkeypatch.setattr(service,'build_regions',forbid)
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run();assert not app.exception
    app.selectbox(key='sm_video').set_value(CARACAS).run()
    walking=submit(app,'a person walking.');assert walking['status']=='done' and walking['result']['results']
    assert all(r.get('event_claim') is None for r in walking['result']['results'])
    app.text_input(key='sm_query').set_value('a person sitting').run()
    assert app.session_state[state.KEY]['id']==walking['id'] and app.session_state[state.KEY]['query']=='a person walking.'
    sitting=submit(app,'a person sitting');assert sitting['id']!=walking['id'] and sitting['query']=='a person sitting'
    assert not any('a person walking' in h.value for h in app.subheader)
    assert app.text_input(key='sm_model').value=='ViT-B-32' and app.text_input(key='sm_pretrained').value=='openai'
    assert app.number_input(key='sm_sampling').value==1. and app.number_input(key='sm_batch').value==16
    assert len(app.get('video'))==1
    rows=sitting['result']['results'];assert rows
    play_key=f"play_{sitting['id']}_0";app.button(key=play_key).click().run()
    assert app.session_state[state.KEY]['selected']==0 and app.get('video')[0].proto.start_time==int(rows[0]['timestamp'])
    if len(rows)>5:
        assert len([b for b in app.button if b.label=='Play'])==5
        app.button(key='sm_more').click().run();assert len([b for b in app.button if b.label=='Play'])==min(10,len(rows))
    for mode in ('Regions','Regions + Evidence','CLIP-only'):
        app.segmented_control(key='sm_mode').set_value(mode).run();assert app.session_state[state.KEY]['result'] is None
        response=submit(app,'a visible backpack');assert response['result']['video_id']==CARACAS and response['result']['ui_mode']==mode
    app.checkbox(key='sm_strict').check().run();missing=submit(app,'a person running')
    assert missing['result']['results']==[] and any('No supported matches' in x.value for x in app.info)
    assert not any(b.label=='Play' for b in app.button)
    app.selectbox(key='sm_video').set_value(FIT).run()
    assert app.session_state[state.KEY]['result'] is None and len(app.get('video'))==1
    assert 'fitwalking.mp4' in app.get('video')[0].proto.url or Path(app.get('video')[0].proto.url).suffix=='.mp4'
    app.checkbox(key='sm_strict').uncheck().run();other=submit(app,'a person')
    assert other['result']['video_id']==FIT and all(FIT not in str(r.get('track_scope','')) or r['track_scope']==FIT for r in other['result']['results'])
    failed=submit(app,'__error_fixture__');assert failed['status']=='error' and failed['result'] is None
    assert not any(b.label=='Play' for b in app.button)
    recovered=submit(app,'a person');assert recovered['status']=='done' and recovered['id']!=failed['id']
    assert len(observed)==len(set(observed))


def test_local_video_import_preserves_mode_and_selection():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    app.text_input(key='sm_local').set_value(str(ROOT/'data/phase26/fitwalking.mp4')).run()
    app.button(key='sm_import').click().run()
    assert not app.exception and app.session_state['sm_video']==FIT and app.session_state['sm_mode']=='CLIP-only'
    assert app.session_state[state.KEY]['result'] is None and len(app.get('video'))==1


def test_existing_processing_tools_and_overlay_stay_integrated():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    app.selectbox(key='sm_video').set_value(CARACAS).run()
    app.checkbox(key='tool_detection').check().run()
    app.button(key='tool_existing').click().run();assert not app.exception
    assert app.session_state['sm_tools'][CARACAS]['detection']['manifest']['identity']['video_sha256']==CARACAS
    app.button(key='tool_quality').click().run()
    assert 'quality' in app.session_state['sm_tools'][CARACAS]
    app.checkbox(key='tool_associations').check().run()
    app.checkbox(key='tool_temporal').check().run()
    assert not app.exception and app.button(key='tool_preview_temporal')
    app.button(key='tool_preview_detection').click().run()
    assert len(app.get('video'))==1 and app.session_state['sm_overlay']['video_id']==CARACAS
    response=submit(app,'a person walking')
    assert response['status']=='done' and 'sm_overlay' not in app.session_state and len(app.get('video'))==1
    app.button(key='tool_process').click().run()
    assert not app.exception and app.session_state['sm_tools'][CARACAS]['detection']['cache_hit']
    app.button(key='tool_build_associations').click().run();assert not app.exception
    app.checkbox(key='tool_temporal').check().run()
    app.button(key='tool_build_timeline').click().run();assert not app.exception
    app.button(key='tool_preview_temporal').click().run()
    assert len(app.get('video'))==1 and app.session_state['sm_overlay']['video_id']==CARACAS


def test_existing_full_frame_hybrid_time_filters_and_all_candidates():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    app.selectbox(key='sm_video').set_value(CARACAS).run()
    response=submit(app,'a visible backpack between 7 and 13 seconds')
    assert response['result']['results'] and all(7<=r['timestamp']<=13 for r in response['result']['results'])
    app.segmented_control(key='sm_mode').set_value('Regions + Evidence').run()
    app.checkbox(key='sm_full_hybrid').check().run()
    response=submit(app,'Find all people near a bag between 7 and 13 seconds')
    result=response['result'];assert result['all_spatial_candidate_track_ids'] and result['results']
    assert all(7<=r['timestamp']<=13 and r['event_claim'] is None for r in result['results'])

def test_existing_demo_search_and_timestamp_navigation():
    from scenemind_mvp.pipeline import digest
    demo=ROOT/'data/demo/scenemind_demo.mp4'
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    app.selectbox(key='sm_video').set_value(digest(demo)).run()
    response=submit(app,'a person walking')
    assert response['status']=='done' and response['result']['results'] and Path(response['result']['source']).name=='scenemind_demo.mp4'
    app.button(key=f"play_{response['id']}_0").click().run()
    assert len(app.get('video'))==1 and app.get('video')[0].proto.start_time==int(response['result']['results'][0]['timestamp'])


def test_mode_or_video_change_and_submit_in_same_rerun_uses_latest_context():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    app.selectbox(key='sm_video').set_value(CARACAS).run()
    first=submit(app,'a person walking')
    app.segmented_control(key='sm_mode').set_value('Regions')
    app.text_input(key='sm_query').set_value('a person sitting')
    app.button(key='sm_search_submit').click().run()
    assert not app.exception
    latest=app.session_state[state.KEY]
    assert latest['status']=='done' and latest['id']!=first['id'] and latest['result']['ui_mode']=='Regions'
    app.selectbox(key='sm_video').set_value(FIT)
    app.text_input(key='sm_query').set_value('a person')
    app.button(key='sm_search_submit').click().run()
    assert not app.exception and app.session_state[state.KEY]['status']=='done'
    assert app.session_state[state.KEY]['result']['video_id']==FIT
