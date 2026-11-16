"""Selected detected targets, scoped to the current search execution."""
from pathlib import Path
import streamlit as st
from . import state
from ..highlighting.render import thumbnail,highlighted_video

def key(run: dict,rank: int) -> str:return f"sm_target_{run['id']}_{rank}"

def selected_targets(run: dict,rank: int) -> list[dict]:
    row=run['result']['results'][rank];loc=row.get('localization',{});choice=st.session_state.get(key(run,rank),'Suggested candidates')
    identifiers=loc.get('primary_target_ids',[]) if choice=='Suggested candidates' else [choice]
    return [t for t in loc.get('candidates',[]) if t['id'] in identifiers]

def play_highlight(identifier: str,rank: int) -> None:
    if state.select(st.session_state,identifier,rank):st.session_state.sm_playback_mode='Highlighted Video'

def target_picker(run: dict,rank: int) -> list[dict]:
    loc=run['result']['results'][rank].get('localization',{});candidates=loc.get('candidates',[])
    if candidates:
        with st.popover('Inspect candidates',icon=':material/person_search:'):
            labels={c['id']:f"{c['label']} / region score {c['region_cosine']:.3f}" for c in candidates}
            st.selectbox('Detected candidate',['Suggested candidates']+list(labels),format_func=lambda v:labels.get(v,v),key=key(run,rank))
            st.caption('Manual selection inspects a detection. It does not confirm the query. Track IDs are temporary references.')
    return selected_targets(run,rank)

def preview(source: str,row: dict,targets: list[dict],data: Path) -> Path|None:
    if not targets:return None
    return thumbnail(Path(source),row['localization'],targets,data/'highlights')

def playback(source: str,row: dict,targets: list[dict],data: Path,include_objects: bool=False) -> dict:
    drawn=list(targets)
    if include_objects:
        for target in targets:
            for obj in target.get('associated_objects',[]):
                if not any(t['id']==obj['id'] for t in drawn):drawn.append(obj)
    return highlighted_video(Path(source),row['localization'],drawn,data/'highlights')
