"""Actual cached project footage/model output; no invented ground truth."""
from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest
from scenemind_mvp.unified.pipeline import load_analysis
from scenemind_mvp.unified.core import make_plan,match_summary
from scenemind_mvp.unified.render import render
from scenemind_mvp.ui import state
ROOT=Path(__file__).resolve().parents[2]
STUDENTS='c242ea439bad72789b87929c627e1ab64a67a9c211bdb82decd79dbd037b3ce0'

def analysis():
    report=ROOT/'docs/phase3/actual-analysis.json'
    if not report.exists():pytest.skip('Run scripts/phase3/evaluate.py --render on project samples')
    row=next(r for r in json.loads(report.read_text())['runs'] if STUDENTS in r['source'])
    return load_analysis(Path(row['analysis_directory']))

def test_real_video_multiple_people_queries_and_cache(monkeypatch):
    a=analysis();assert a['actions']['model_available']
    all_plan=make_plan('',a);all_video=render(a,all_plan,ROOT/'data/unified-renders',{})
    assert all_video['manifest']['max_visible_people']>=5
    walking=make_plan('Show all walking people',a);summary=match_summary(walking,a)
    assert len(summary['track_ids'])>=5 and summary['verified'] is False
    assert not match_summary(make_plan('Person carrying a black bag',a),a)['track_ids']
    def forbid(*args,**kwargs):raise AssertionError('Cached rerun must not invoke ffmpeg')
    monkeypatch.setattr('scenemind_mvp.unified.render.subprocess.run',forbid)
    repeated=render(a,all_plan,ROOT/'data/unified-renders',{});assert repeated['cache_hit'] and repeated['video']==all_video['video']
    audit=json.loads((Path(repeated['directory'])/'frame-audit.json').read_text());assert len(audit)==a['metadata']['frame_count']
    frames={f['frame_index']:f for f in a['frames']}
    for f in audit:
        assert len(f['targets'])==len(frames.get(f['frame_index'],{}).get('detections',[]))
        assert all(0<=t['bbox_xyxy'][0]<=t['bbox_xyxy'][2]<=960 for t in f['targets'])

def test_real_ui_one_player_lifecycle_and_id_inspection():
    a=analysis();app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run()
    app.selectbox(key='sm_video').set_value(STUDENTS).run();assert not app.exception
    assert len(app.get('video'))==1 and app.segmented_control(key='sm_playback_mode').value=='Annotated Video'
    assert app.segmented_control(key='sm_unified_view').value=='Show All'
    app.text_input(key='sm_query').set_value('Show all walking people');app.button(key='sm_search_submit').click().run();assert not app.exception
    first=app.session_state[state.KEY]['id'];assert app.segmented_control(key='sm_unified_view').value=='Search Matches'
    assert any('c0-t1, c0-t2, c0-t3, c0-t4, c0-t5' in c.value for c in app.caption)
    app.selectbox(key='sm_unified_inspect').set_value('c0-t3').run();assert not app.exception and len(app.get('video'))==1
    app.text_input(key='sm_query').set_value('Show all sitting people');app.button(key='sm_search_submit').click().run();assert not app.exception
    assert app.session_state[state.KEY]['id']!=first and any('No supported structured matches' in i.value for i in app.info)
    app.text_input(key='sm_query').set_value('');app.button(key='sm_search_submit').click().run();assert not app.exception
    assert app.session_state[state.KEY]['result'] is None and app.segmented_control(key='sm_unified_view').value=='Show All'
    assert len(app.get('video'))==1
    app.segmented_control(key='sm_playback_mode').set_value('Original Video').run();assert not app.exception
    assert len(app.get('video'))==1


def test_repaired_queries_modes_and_repeated_ui_without_model_reanalysis(monkeypatch):
    a=analysis();from scenemind_mvp.pipeline import digest
    before=digest(Path(a['detection_directory'])/'records.json')
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run();app.selectbox(key='sm_video').set_value(STUDENTS).run();assert not app.exception
    from scenemind_mvp.detection.detector import Detector
    from scenemind_mvp.unified.actions import ActionModel
    def forbid(*args,**kw):raise AssertionError('Repeated searches must reuse analysis, not rerun detection/actions')
    monkeypatch.setattr(Detector,'infer',forbid);monkeypatch.setattr(ActionModel,'predict',forbid)
    for mode in ('CLIP-only','Regions','Regions + Evidence'):
        app.segmented_control(key='sm_mode').set_value(mode).run()
        for query,expected in [('a person walking',['c0-t1','c0-t2','c0-t3','c0-t4','c0-t5']),('a person walking in green shirt',['c0-t3']),('a person wearing a black shirt',[])]:
            app.text_input(key='sm_query').set_value(query);app.button(key='sm_search_submit').click().run();assert not app.exception
            assert not any('Annotated playback unavailable' in w.value for w in app.warning)
            evidence=app.session_state[state.KEY]['result']['unified_evidence'];assert evidence['track_ids']==expected and evidence['verified_matches']==[]
            if 'green' in query:assert evidence['status']=='uncertain_clothing_candidates'
            if 'black' in query:assert evidence['status']=='unknown_or_no_match' and evidence['plan']['intervals']==[]
            assert len(app.get('video'))==1
        app.button(key='sm_clear_search').click().run();assert not app.exception
        assert app.session_state[state.KEY]['result'] is None and app.segmented_control(key='sm_unified_view').value=='Show All'
    assert digest(Path(a['detection_directory'])/'records.json')==before


def test_same_timestamp_request_forces_only_player_remount():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run();app.selectbox(key='sm_video').set_value(STUDENTS).run();assert not app.exception
    app.text_input(key='sm_query').set_value('a person walking in green shirt');app.button(key='sm_search_submit').click().run();assert not app.exception
    first=app.session_state['sm_unified_seek_revision'];button=next(b for b in app.button if b.label=='Jump to interval');button.click().run();assert not app.exception
    assert app.session_state['sm_unified_seek_revision']>first and len(app.get('video'))==1
    assert app.get('video')[0].proto.start_time==4
