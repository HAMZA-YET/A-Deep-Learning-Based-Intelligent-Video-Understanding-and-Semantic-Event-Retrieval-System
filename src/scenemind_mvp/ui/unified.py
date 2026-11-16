"""Resources and controls for the single multi-target video player."""
from pathlib import Path
import json
import streamlit as st
from ..unified.pipeline import analyze
from ..unified.actions import ActionModel
from ..unified.core import make_plan,match_summary,label
from ..unified.render import render
from .tools import detector_resource
from . import state

@st.cache_resource(max_entries=2,show_spinner=False)
def action_resource(cache: str,device: str,revision: tuple):return ActionModel(Path(cache),device)

@st.cache_resource(max_entries=8,show_spinner=False)
def analysis_resource(source: str,stamp: tuple,data: str,config_json: str,weights_revision: tuple):
    config=json.loads(config_json)
    return analyze(Path(source),Path(data),config,model_factory=lambda:action_resource(str(Path(data)/'models/actions'),config['detection']['device'],weights_revision),detector_factory=lambda c,p:detector_resource(json.dumps(c.to_dict(),sort_keys=True),str(p)))

def ensure(source: str,root: Path,data: Path,device: str) -> dict:
    p=Path(source);s=p.stat();config=json.loads((root/'configs/unified.json').read_text());config['detection']['device']=device;weights=data/'models/actions/SLOWFAST_8x8_R50_DETECTION.pyth';revision=(weights.stat().st_size,weights.stat().st_mtime_ns) if weights.is_file() else ()
    return analysis_resource(str(p.resolve()),(s.st_size,s.st_mtime_ns),str(data),json.dumps(config,sort_keys=True),revision)

def seek(seconds: float) -> None:
    run=st.session_state.get(state.KEY)
    if run:run['selected']=None
    st.session_state.sm_unified_seek_revision=st.session_state.get('sm_unified_seek_revision',0)+1
    st.session_state.sm_unified_seek=seconds
    st.session_state.sm_unified_timeline=seconds

def timeline_changed():seek(float(st.session_state.sm_unified_timeline))

def track_changed():
    chosen=st.session_state.get('sm_unified_inspect');first=st.session_state.get('sm_unified_first',{})
    seek(first.get(chosen,0.));st.session_state.sm_unified_view='Search Matches' if chosen else 'Show All';st.session_state.sm_playback_mode='Annotated Video'

def options() -> dict:
    view=st.segmented_control('Targets',['Show All','Search Matches'],key='sm_unified_view',required=True)
    with st.popover('Overlay controls',icon=':material/tune:'):
        st.checkbox('Bounding boxes',value=True,key='sm_u_boxes');st.checkbox('Tracking IDs',value=True,key='sm_u_ids');st.checkbox('Action labels',value=True,key='sm_u_actions');st.checkbox('Object labels',value=True,key='sm_u_objects');st.checkbox('Hide nonmatching detections',key='sm_u_hide')
    return {'view':view,'boxes':st.session_state.sm_u_boxes,'ids':st.session_state.sm_u_ids,'actions':st.session_state.sm_u_actions,'objects':st.session_state.sm_u_objects,'hide_nonmatches':st.session_state.sm_u_hide}

def footer(analysis: dict,plan: dict) -> None:
    meta=analysis['metadata'];diag=analysis['diagnostics'];performance=analysis['detection_performance']
    st.caption(f"Up to {performance['max_simultaneous_person_detections']} people per frame · {diag['person_tracklets']} person tracklets (not distinct people). Live frame counts appear inside the video.")
    st.slider('Timeline (seconds)',0.,float(meta['duration']),step=.1,key='sm_unified_timeline',on_change=timeline_changed)
    tracks=diag['tracks'];st.session_state.sm_unified_first={t['track_id']:t['first_timestamp'] for t in tracks}
    names={t['track_id']:label({'class_id':t['class_id'],'label':t.get('label','object'),'track_id':t['track_id']}) for t in tracks}
    st.selectbox('Inspect tracking ID',[None]+list(names),format_func=lambda v:'All tracks' if v is None else names.get(v,v)+' ['+v+']',key='sm_unified_inspect',on_change=track_changed)
    if not analysis['actions']['model_available']:st.warning('Action unknown: '+analysis['actions'].get('error','Temporal model unavailable'))
    st.caption('Actions marked model are repeated temporal predictions, not independently verified labels. Missing/ambiguous actions remain unknown; track IDs are temporary.')
    intervals=plan.get('intervals',[]) if plan['mode']=='search' else analysis['actions']['intervals']
    if intervals:
        with st.expander('Relevant action timeline'):
            for i,e in enumerate(intervals):
                st.button(f"{e['person_track_id']} / {e['action']} / {e['start']:.2f}-{e['end']:.2f}s",key=f"u_interval_{analysis['directory']}_{i}_{plan['query']}",on_click=seek,args=(e['start'],))


@st.cache_resource(max_entries=16,show_spinner=False)
def attribute_resource(directory: str,identity: str,data: str):
    from ..unified.pipeline import load_analysis
    from ..unified.attributes import analyze_attributes
    return analyze_attributes(load_analysis(Path(directory)),Path(data)/'attributes')

def with_attributes(scene: dict,data: Path) -> dict:
    from ..unified.attributes import POLICY
    try:return {**scene,'attributes':attribute_resource(scene['directory'],json.dumps({'analysis':scene['identity'],'attributes':POLICY},sort_keys=True),str(data))}
    except (OSError,ValueError,RuntimeError) as error:return {**scene,'attributes':{'error':str(error),'intervals':[],'observations':[]}}
