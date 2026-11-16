"""Actual cached Mixkit embeddings, tracker records, ffmpeg and main UI."""
from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest
from scenemind_mvp.highlighting.render import highlighted_video,validate_regions
from scenemind_mvp.ui import state
ROOT=Path(__file__).resolve().parents[2]
SHA='011209fbf780baa5ae05e9ea020c70bdb4de1580f04d3d6bcb4aa74da205d935'

def test_actual_video_boxes_cache_and_loss_policy(monkeypatch):
    from copy import deepcopy
    sample=ROOT/'docs/highlighting/smoke.json'
    if not sample.exists():pytest.skip('Run real highlighting evaluation first')
    result=json.loads(sample.read_text());row=result['results'][0];loc=row['localization'];targets=[t for t in loc['candidates'] if t['id'] in loc['primary_target_ids']]
    output=highlighted_video(Path(result['source']),loc,targets,ROOT/'data/highlights')
    def forbid(*a,**kw):raise AssertionError('Cache must not re-encode video')
    monkeypatch.setattr('scenemind_mvp.highlighting.render.subprocess.run',forbid)
    cached=highlighted_video(Path(result['source']),loc,targets,ROOT/'data/highlights');assert cached['cache_hit']
    assert cached['video']==output['video']
    audit=json.loads((Path(cached['directory'])/'frame-audit.json').read_text())
    assert sum(bool(f['visible_target_ids']) for f in audit)==cached['manifest']['highlighted_frames']
    assert cached['manifest']['source_frames']==len(audit)
    bad=deepcopy(targets);bad[0]['bbox_xyxy'][0]+=1
    with pytest.raises(ValueError):validate_regions(loc,bad)
    assert Path(result['source']).exists()

def test_actual_ui_highlight_selection_and_stale_clear():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
    app.selectbox(key='sm_video').set_value(SHA).run()
    app.segmented_control(key='sm_mode').set_value('Regions').run()
    app.text_input(key='sm_query').set_value('A person walking.')
    app.button(key='sm_search_submit').click().run();assert not app.exception
    run=app.session_state[state.KEY];assert run['status']=='done',run
    assert run['result']['results'][0]['localization']['primary_target_ids']
    app.button(key=f"highlight_{run['id']}_0").click().run();assert not app.exception
    assert app.segmented_control(key='sm_playback_mode').value=='Highlighted Video'
    assert len(app.get('video'))==1 and app.get('video')[0].proto.start_time==int(run['result']['results'][0]['timestamp'])
    assert not app.warning
    app.segmented_control(key='sm_playback_mode').set_value('Original Video').run();assert not app.exception
    alternatives=run['result']['results'][0]['localization']['candidates']
    app.selectbox(key=f"sm_target_{run['id']}_0").set_value(alternatives[-1]['id']).run();assert not app.exception
    app.text_input(key='sm_query').set_value('An outdoor scene.')
    app.button(key='sm_search_submit').click().run();assert not app.exception
    latest=app.session_state[state.KEY];assert latest['id']!=run['id']
    assert app.segmented_control(key='sm_playback_mode').value=='Annotated Video'
    assert all(b.disabled for b in app.button if b.label=='Play Highlighted Video')
    assert any('No specific target localized.' in c.value for c in app.caption)


def test_clip_only_cache_miss_prepares_localization_and_preserves_ranking(monkeypatch):
    from scenemind_mvp.ui.catalog import catalog
    import scenemind_mvp.ui.service as service
    video=next(v for v in catalog(ROOT/'data',[]) if v['id']==SHA)
    video['regions']=[] # Actual UI path for a new video/configuration without discovered ROIs.
    settings={'sampling_fps':1.,'batch_size':16,'model':'ViT-B-32','pretrained':'openai','device':'auto','max_results':5,'gap':1.,'cutoff':-1.,'strict':False,'padding':.2,'detector_sizes':[640,960],'detection_confidence':.1,'evidence_profile':'Baseline','full_hybrid':False}
    original=service.prepare;calls=[]
    def checked(video,mode,settings,*args):
        calls.append(mode);return original(video,mode,settings,*args)
    monkeypatch.setattr(service,'prepare',checked)
    result=service.search(video,'CLIP-only','a person walking',settings,ROOT,ROOT/'data')
    assert calls==['CLIP-only','Regions']
    assert result['ui_mode']=='CLIP-only' and result['asset_kind']=='index'
    assert result['localization_pipeline']['ranking_unchanged']
    assert result['results'][0]['localization']['primary_target_ids']
