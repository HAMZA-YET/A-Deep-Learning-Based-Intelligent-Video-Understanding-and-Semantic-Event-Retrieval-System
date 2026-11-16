from pathlib import Path
import json
from streamlit.testing.v1 import AppTest

def test_region_ui_real_search_preview_and_action_abstention():
    app=AppTest.from_file(str(Path('scripts/regions/region_app.py').resolve()),default_timeout=45).run()
    assert not app.exception
    report=json.loads(Path('benchmarks/regions/comparison.json').read_text())
    index=next(e['region_directory'] for e in report['experiments'] if e['video_id']=='sabana_grande' and e['config_path']=='configs/regions.json')
    app.selectbox(key='region_index').set_value(index).run()
    app.button(key='region_submit').click().run()
    assert not app.exception
    result=app.session_state['region_result']['result'];assert result['results'] and result['results'][0]['regions']
    app.button(key='region_play_0').click().run()
    assert not app.exception and app.session_state['region_playback']['timestamp']==result['results'][0]['timestamp']
    app.text_input(key='region_query').set_value('Show a person running.')
    app.button(key='region_submit').click().run()
    assert not app.exception and app.session_state['region_result']['result']['status']=='no_supported_match'
    app.text_input(key='region_query').set_value('Show a person walking.')
    app.checkbox(key='region_strict').uncheck()
    app.button(key='region_submit').click().run()
    result=app.session_state['region_result']['result'];assert result['results'] and all(r['event_claim'] is None for r in result['results'])
