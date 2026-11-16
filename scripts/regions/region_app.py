"""Optional region search interface; existing app.py is unchanged."""
from pathlib import Path
import json,threading
import streamlit as st
from scenemind_mvp.config import Config
from scenemind_mvp.model import Encoder
from scenemind_mvp.regions.schema import RegionConfig
from scenemind_mvp.regions.pipeline import load_regions
from scenemind_mvp.regions.engine import RegionEngine
from scenemind_mvp.hybrid.evidence import load_bundle
from scenemind_mvp.hybrid.schema import HybridConfig
ROOT=Path(__file__).resolve().parents[2]
st.set_page_config(page_title='SceneMind region search',layout='wide')
st.title('SceneMind region-aware search')
st.caption('Full-frame and native padded region scores rank visual candidates. Single frames do not establish actions, relationships or identity.')
@st.cache_resource
def encoder_resource(config_json: str):return Encoder(Config(**json.loads(config_json)),ROOT/'data/models'),threading.RLock()
paths=sorted((ROOT/'data/regions').glob('*/manifest.json'))
if not paths:st.info('Build a region index with scripts/regions/index_search.py first.');st.stop()
metadata={str(p.parent):json.loads(p.read_text(encoding='utf-8')) for p in paths}
selected=st.selectbox('Region index',list(metadata),format_func=lambda p:f"{Path(metadata[p]['source']).name} | {metadata[p]['identity']['region_config']['sampling_fps']} FPS | YOLO {metadata[p]['identity']['region_config']['detector_sizes']} | {metadata[p]['region_count']} regions",key='region_index')
mode=st.segmented_control('Search mode',['CLIP-only','Regions','Regions + evidence'],default='Regions + evidence',key='region_mode')
with st.form('region-search'):
    text=st.text_input('Query',value='Show a visible backpack.',key='region_query')
    strict=st.checkbox('Require temporal support for actions and relationships',value=True,key='region_strict')
    submitted=st.form_submit_button('Search regions',key='region_submit')
if submitted:
    try:
        region=load_regions(Path(selected));_,evidence=load_bundle(Path(region.manifest['bundle']));d=region.manifest['identity']['region_config'];d['detector_sizes']=tuple(d['detector_sizes']);config=RegionConfig(**d)
        encoder,lock=encoder_resource(json.dumps({k:v for k,v in region.full.manifest['identity']['config'].items() if k in ('model','pretrained','device')},sort_keys=True))
        with lock:vector=encoder.encode_text(text)
        engine=RegionEngine(region,evidence,config,HybridConfig(**json.loads((ROOT/'configs/hybrid.json').read_text())))
        result=engine.search(text,vector,{'CLIP-only':'clip_only','Regions':'region','Regions + evidence':'region_hybrid'}.get(mode,'region_hybrid'),strict_actions=strict)
        st.session_state['region_result']={'result':result,'source':region.manifest['source'],'index':selected}
    except (ValueError,OSError,RuntimeError) as error:st.error(str(error))
if 'region_result' in st.session_state:
    saved=st.session_state['region_result'];result=saved['result'];st.write('Result state:',result['status']);st.caption('Displayed results use the last submitted query and mode. Scores are not probabilities.')
    st.json(result['query'],expanded=False)
    for i,row in enumerate(result['results']):
        st.subheader(f"{row['timestamp']:.2f}s | ranking score {row['rank_score']:.4f}")
        columns=st.columns(3);columns[0].image(row['preview'],caption=f"Full-frame cosine {row['full_frame_cosine']:.4f}",alt='Full source frame at the retrieved timestamp',width='stretch')
        for column,crop in zip(columns[1:],row['regions']):
            column.image(crop['preview'],caption=f"{crop['label']}; cosine {crop['cosine']:.4f}; source pixels {crop['native_crop_size']}",alt='Padded source-resolution retrieved region',width='stretch')
            column.caption('Track reference: '+(crop['track_id'] or 'unconfirmed'))
        st.write(row['description']);st.caption('All matched proposal tracklets: '+', '.join(row['all_proposal_track_ids']))
        with st.expander(f'Evidence and uncertainty at {row["timestamp"]:.2f}s'):st.json({'event_claim':row['event_claim'],'uncertainty':row['uncertainty'],'structured_evidence':row.get('evidence',{}),'regions':row['regions']})
        if st.button('Play source near this moment',key=f'region_play_{i}'):
            st.session_state['region_playback']={'source':saved['source'],'timestamp':row['timestamp']}
    st.download_button('Download region results',json.dumps(result,indent=2),file_name='region-results.json',mime='application/json',key='region_download')
if 'region_playback' in st.session_state:
    playback=st.session_state['region_playback'];st.caption(f"Selected sample {playback['timestamp']:.2f}s; playback starts at the preceding whole second.")
    st.video(playback['source'],start_time=int(playback['timestamp']),alt='Original source video near the selected region result')
