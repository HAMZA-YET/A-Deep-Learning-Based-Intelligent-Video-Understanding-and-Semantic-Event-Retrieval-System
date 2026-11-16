"""Latest-submission-only state, independent of Streamlit for deterministic tests."""
from __future__ import annotations
from collections.abc import MutableMapping
from uuid import uuid4

KEY='sm_execution'
LEGACY=('matches','play_timestamp','hybrid_response','hybrid_play','region_result','region_playback','query_seconds')
def clear_outputs(state: MutableMapping) -> None:
    for key in LEGACY:state.pop(key,None)
    state.pop('sm_overlay',None)
    state['sm_playback_mode']='Annotated Video'
    state['sm_unified_view']='Show All'
    state['sm_unified_seek_revision']=state.get('sm_unified_seek_revision',0)+1
    state['sm_unified_seek']=0.
    state['sm_unified_timeline']=0.
    state.pop('sm_unified_completed',None)
    state.pop('sm_unified_inspect',None)
    state.pop('sm_highlight_objects',None)

def invalidate(state: MutableMapping,context: str) -> None:
    clear_outputs(state)
    state[KEY]={'id':uuid4().hex,'context':context,'status':'idle','query':'','result':None,'error':None,'visible':5,'selected':None}

def bind_context(state: MutableMapping,context: str) -> None:
    if KEY not in state or state[KEY]['context']!=context:invalidate(state,context)

def begin(state: MutableMapping,context: str,query: str) -> str:
    clear_outputs(state);identifier=uuid4().hex
    state[KEY]={'id':identifier,'context':context,'status':'pending','query':query.strip(),'result':None,'error':None,'visible':5,'selected':None}
    return identifier

def complete(state: MutableMapping,identifier: str,context: str,result: dict) -> bool:
    run=state.get(KEY)
    if not run or run['id']!=identifier or run['context']!=context or run['status']!='pending':return False
    run.update(status='done',result=result,error=None);return True

def fail(state: MutableMapping,identifier: str,context: str,message: str) -> bool:
    run=state.get(KEY)
    if not run or run['id']!=identifier or run['context']!=context or run['status']!='pending':return False
    run.update(status='error',result=None,error=message,selected=None);return True

def select(state: MutableMapping,identifier: str,index: int) -> bool:
    run=state.get(KEY)
    if not run or run['id']!=identifier or run['status']!='done' or not 0<=index<len(run['result']['results']):return False
    state.pop('sm_overlay',None);state['sm_unified_seek_revision']=state.get('sm_unified_seek_revision',0)+1;run['selected']=index;return True
