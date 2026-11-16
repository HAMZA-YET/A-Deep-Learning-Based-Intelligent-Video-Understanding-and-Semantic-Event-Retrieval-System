"""SceneMind: unified presentation over the existing retrieval and evidence pipelines."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,logging,os
import streamlit as st
from PIL import Image,ImageOps
from scenemind_mvp.config import Config
from scenemind_mvp.ui import state
from scenemind_mvp.ui.catalog import catalog,read_json
from scenemind_mvp.ui.service import source_info,search,asset_signature
from scenemind_mvp.ui.presentation import label_result,timestamp,friendly_error
from scenemind_mvp.ui.tools import render_tools
from scenemind_mvp.ui import highlights,unified
from scenemind_mvp.unified.core import make_plan,match_summary
from scenemind_mvp.unified.render import render as render_unified

ROOT=Path(__file__).resolve().parent
DATA=Path(os.getenv('SCENEMIND_DATA_DIR',str(ROOT/'data'))).resolve()
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title='SceneMind · Video search',page_icon=':material/video_library:',layout='wide',initial_sidebar_state='auto')
base=Config.load(ROOT/'configs/default.json')
defaults={'sm_mode':'CLIP-only','sm_video':None,'sm_query':'','sm_sampling':base.sampling_fps,'sm_batch':base.batch_size,'sm_model':base.model,'sm_pretrained':base.pretrained,'sm_device':base.device,'sm_limit':20,'sm_gap':1.,'sm_cutoff':-1.,'sm_strict':False,'sm_padding':.2,'sm_multiscale':True,'sm_region_conf':.1,'sm_evidence_profile':'Baseline','sm_full_hybrid':False,'sm_extras':[]}
for key,value in defaults.items():
    if key in ('sm_video','sm_query','sm_extras') and key not in st.session_state:st.session_state[key]=deepcopy(value)

def settings() -> dict:
    return {'sampling_fps':float(st.session_state.sm_sampling),'batch_size':int(st.session_state.sm_batch),'model':st.session_state.sm_model.strip(),'pretrained':st.session_state.sm_pretrained.strip(),'device':st.session_state.sm_device,'max_results':int(st.session_state.sm_limit),'gap':float(st.session_state.sm_gap),'cutoff':float(st.session_state.sm_cutoff),'strict':st.session_state.sm_strict,'padding':float(st.session_state.sm_padding),'detector_sizes':[640,960] if st.session_state.sm_multiscale else [640],'detection_confidence':float(st.session_state.sm_region_conf),'evidence_profile':st.session_state.sm_evidence_profile,'full_hybrid':st.session_state.sm_full_hybrid and st.session_state.get('sm_mode','CLIP-only')=='Regions + Evidence'}

def restore_settings() -> None:
    for key,value in defaults.items():
        if key not in ('sm_mode','sm_video','sm_query','sm_extras'):st.session_state.pop(key,None)

def import_video() -> None:
    try:
        upload=st.session_state.get('sm_upload');local=st.session_state.get('sm_local','').strip().strip('"')
        if upload is not None:
            content=upload.getbuffer();path=DATA/'uploads'/(hashlib.sha256(content).hexdigest()+'.mp4');path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_bytes(content)
            name=upload.name
        else:path=Path(local).expanduser().resolve();name=path.name
        info=source_info(path);entry={'id':info['id'],'source':info['source'],'name':name}
        st.session_state.sm_extras=[e for e in st.session_state.sm_extras if e['id']!=entry['id']]+[entry]
        st.session_state.sm_video=entry['id'];st.session_state.pop('sm_import_error',None)
    except Exception as error:st.session_state.sm_import_error=friendly_error(error)

def execution_context(video_id: str|None) -> str:
    selected=st.session_state.get('sm_catalog',{}).get(video_id);stamp=None
    if selected:
        try:
            stat=Path(selected['source']).stat();stamp=[stat.st_size,stat.st_mtime_ns]
        except OSError:pass
    return json.dumps({'localization_revision':'phase3-unified-ava-v1','video':video_id,'source_stamp':stamp,'mode':st.session_state.get('sm_mode') or 'CLIP-only','settings':settings()},sort_keys=True)

def submit_search() -> None:
    # Callbacks see new widget values before the body reruns. Never use an older cached context.
    context=execution_context(st.session_state.sm_video);st.session_state.sm_context=context
    if not st.session_state.sm_query.strip():
        clear_search();return
    state.begin(st.session_state,context,st.session_state.sm_query)
    st.session_state.sm_unified_view='Search Matches'

def clear_search() -> None:
    st.session_state.sm_query='';state.invalidate(st.session_state,execution_context(st.session_state.sm_video))

def play(identifier: str,index: int) -> None:
    if state.select(st.session_state,identifier,index):st.session_state.sm_playback_mode='Original Video'
def more(identifier: str) -> None:
    run=st.session_state.get(state.KEY)
    if run and run['id']==identifier and run['status']=='done':run['visible']+=5

@st.cache_data(max_entries=100,show_spinner=False)
def thumbnail(path: str,mtime: int):
    with Image.open(path) as im:return ImageOps.pad(im.convert('RGB'),(320,180),color='#ECF1F7')

with st.sidebar:
    st.markdown('### Video library')
    library=catalog(DATA,st.session_state.sm_extras);videos={v['id']:v for v in library};st.session_state.sm_catalog=videos
    if st.session_state.sm_video not in videos:st.session_state.sm_video=None
    video_id=st.selectbox('Selected video',[None]+list(videos),format_func=lambda key:'Choose a video…' if key is None else videos[key]['name'],key='sm_video')
    with st.expander('Add a video',icon=':material/add:'):
        upload=st.file_uploader('Upload MP4',type=['mp4'],key='sm_upload')
        local=st.text_input('Local MP4 path',placeholder='videos/example.mp4',key='sm_local')
        st.button('Use this video',key='sm_import',on_click=import_video,disabled=upload is None and not local.strip(),width='stretch')
        if st.session_state.get('sm_import_error'):st.error(st.session_state.sm_import_error)
    st.caption('Your files and reusable indexes stay local.')
video=videos.get(video_id);info=None;source_error=None
if video:
    try:
        info=source_info(Path(video['source']))
        if info['id']!=video['id']:raise ValueError('Selected video changed')
    except Exception as error:source_error=friendly_error(error)

st.title('SceneMind')
st.caption('Find moments in your videos. Inspect the evidence behind each match.')
# Keep the search widget mounted while downstream media is replaced.
search_slot=st.container(key='sm_search_area');workspace_slot=st.empty()
run_before=st.session_state.get(state.KEY,{})
with st.expander('Advanced Details',icon=':material/tune:',expanded=False,key='sm_advanced'):
    preferences,processing,technical=st.tabs(['Search settings','Evidence tools','Latest execution'])
    with preferences:
        st.button('Restore default search settings',key='sm_restore',on_click=restore_settings)
        cols=st.columns(3)
        cols[0].number_input('Sampling FPS',min_value=.1,max_value=30.,value=base.sampling_fps,key='sm_sampling')
        cols[1].number_input('Batch size',min_value=1,max_value=128,value=base.batch_size,key='sm_batch')
        cols[2].number_input('Maximum moments',min_value=5,max_value=100,step=5,value=20,key='sm_limit')
        cols=st.columns(3)
        cols[0].number_input('Moment separation (seconds)',min_value=0.,value=1.,key='sm_gap')
        cols[1].slider('Minimum visual similarity',-1.,1.,value=-1.,step=.01,key='sm_cutoff')
        cols[2].selectbox('Compute device',['auto','cuda','cpu'],key='sm_device')
        st.checkbox('Only show temporally supported actions and relationships',key='sm_strict')
        with st.container(border=True):
            cols=st.columns(2);cols[0].text_input('OpenCLIP model',value=base.model,key='sm_model');cols[1].text_input('Pretrained tag or checkpoint',value=base.pretrained,key='sm_pretrained')
            cols=st.columns(3);cols[0].number_input('Region padding',min_value=0.,max_value=2.,value=.2,step=.05,key='sm_padding');cols[1].slider('Region detector confidence',.05,.95,value=.1,step=.05,key='sm_region_conf');cols[2].checkbox('Multi-scale region detection',value=True,key='sm_multiscale')
            st.selectbox('Structured evidence profile',['Baseline','Higher resolution experiment'],key='sm_evidence_profile')
            st.checkbox('Use full-frame hybrid comparison in Regions + Evidence',key='sm_full_hybrid')
        st.caption('These controls preserve the existing algorithms. Changes clear displayed results; model output is never independently verified by a similarity score.')
    current_settings=settings()
    context=execution_context(video_id)
    st.session_state.sm_context=context;state.bind_context(st.session_state,context)
    with processing:render_tools(video,current_settings,ROOT,DATA)
    with technical:technical_slot=st.empty()

with search_slot:
    st.segmented_control('Search mode',['CLIP-only','Regions','Regions + Evidence'],default='CLIP-only',required=True,key='sm_mode',help='CLIP-only compares full frames. Regions includes individual crops. Regions + Evidence also uses existing detections and temporal records.')
    # A changed mode can arrive here after its widget callback; bind from current widget values.
    current_settings=settings();context=execution_context(video_id)
    st.session_state.sm_context=context;state.bind_context(st.session_state,context)
    # Draft edits stay in the browser; only submission starts a search/rerun.
    with st.form('sm_search_form',clear_on_submit=False,enter_to_submit=True,border=False):
        query_col,button_col=st.columns([5,1],vertical_alignment='bottom')
        query_col.text_input('Describe a moment',placeholder='A person near a backpack…',key='sm_query',label_visibility='collapsed')
        button_col.form_submit_button('Search',key='sm_search_submit',type='primary',width='stretch',icon=':material/search:',on_click=submit_search,disabled=video is None or source_error is not None)
    # Outside the form: Enter must have exactly one submit target. Keep Clear
    # enabled even when the server has not received the browser's draft yet.
    st.button('Clear search',key='sm_clear_search',on_click=clear_search)
    st.caption('Press Enter or Search to apply your query. Clear search (or submit an empty query) returns to Show All.')
run=st.session_state[state.KEY]
if run['status']=='pending':
    workspace_slot.empty();technical_slot.empty();identifier=run['id']
    with workspace_slot.container():
        with st.spinner('Searching your video…',show_time=True):
            try:
                result=search(video,st.session_state.sm_mode or 'CLIP-only',run['query'],current_settings,ROOT,DATA)
                result['execution_id']=identifier;result['asset_signature']=asset_signature(Path(result['asset_path']),result['asset_kind'])
                state.complete(st.session_state,identifier,context,result)
            except Exception as error:
                logging.exception('SceneMind search failed')
                if state.fail(st.session_state,identifier,context,friendly_error(error)):st.session_state[state.KEY]['debug_error']=str(error)
    st.rerun()

run=st.session_state[state.KEY]
if run['status']=='done':
    try:
        result=run['result']
        if result['video_id']!=video_id or tuple(tuple(x) for x in result['asset_signature'])!=asset_signature(Path(result['asset_path']),result['asset_kind']):state.invalidate(st.session_state,context);run=st.session_state[state.KEY]
    except (OSError,ValueError,KeyError):state.invalidate(st.session_state,context);run=st.session_state[state.KEY]

scene=None;unified_error=None
if video and info:
    try:
        with workspace_slot.container():
            with st.spinner('Analyzing all people, objects and temporal actions (cached per video)...',show_time=True):scene=unified.ensure(video['source'],ROOT,DATA,current_settings['device'])
    except Exception as error:
        logging.exception('Unified video analysis failed');unified_error=str(error)
    workspace_slot.empty()
if scene and run['status']=='done':
    from scenemind_mvp.regions.engine import parse_region_query
    if parse_region_query(run['query']).colors:
        with st.spinner('Inspecting localized person crops for uncertain clothing cues...'):scene=unified.with_attributes(scene,DATA)
plan=make_plan(run['query'] if run['status']=='done' else '',scene,st.session_state.get('sm_unified_inspect')) if scene else None
if scene and run['status']=='done' and st.session_state.get('sm_unified_completed')!=run['id']:
    st.session_state.sm_unified_completed=run['id']
    supported=match_summary(plan,scene)
    if supported['sample_moments']:unified.seek(supported['sample_moments'][0]['timestamp'])
if scene and run.get('result'):
    run['result']['unified_evidence']={**match_summary(plan,scene),'analysis_identity':scene['identity'],'verified_matches':[],'appearance_mode':st.session_state.sm_mode}

with workspace_slot.container():
    if source_error:st.error(source_error)
    if unified_error:st.warning('Unified analysis unavailable: '+unified_error)
    player_column=st.container();results_column=st.container()
    with player_column:
        with st.container(border=True):
            if video and info:
                st.subheader(video['name'])
                result=run.get('result');rows=result['results'] if result else [];selected=run['selected'];at=rows[selected]['timestamp'] if selected is not None and selected<len(rows) else st.session_state.get('sm_unified_seek',0.)
                overlay=st.session_state.get('sm_overlay');overlay=overlay if overlay and overlay['video_id']==video_id else None
                player_path=overlay['source'] if overlay else video['source'];at=overlay.get('timestamp',0.) if overlay else at
                active_targets=highlights.selected_targets(run,selected) if selected is not None and selected<len(rows) else []
                playback_mode=st.segmented_control('Playback',['Original Video','Annotated Video','Highlighted Video'],key='sm_playback_mode',required=True)
                if scene:
                    overlay_options=unified.options()
                    if playback_mode=='Annotated Video' and not overlay:
                        try:
                            with st.spinner('Preparing unified annotated playback (cached)...'):
                                rendered=render_unified(scene,plan,DATA/'unified-renders',overlay_options)
                            player_path=rendered['video']
                        except Exception as error:st.warning('Annotated playback unavailable: '+str(error));player_path=video['source']
                if playback_mode=='Highlighted Video':
                    if active_targets:
                        include_objects=st.checkbox('Show association candidates (unverified)',key='sm_highlight_objects',help='Amber boxes indicate candidate associations, not carrying or ownership.')
                        try:
                            with st.spinner('Preparing selected-target video...'):
                                rendered=highlights.playback(video['source'],rows[selected],active_targets,DATA,include_objects)
                            player_path=rendered['video'];overlay=None
                            st.caption('Track references are temporary tracklets, not real-world identities.')
                            st.caption('Selected visual candidates only. Lost or ambiguous tracks stop highlighting. Attributes and actions remain unverified.')
                        except Exception as error:
                            logging.exception('Target video failed');st.warning('Highlighted playback unavailable: '+friendly_error(error));player_path=video['source']
                    else:st.caption('No specific target localized. Select a localized result first.')
                with st.container(key=f"sm_media_seek_{st.session_state.get('sm_unified_seek_revision',0)}_{hashlib.sha256(str(player_path).encode()).hexdigest()[:12]}"):
                    st.video(player_path,start_time=at,autoplay=selected is not None or at>0,muted=True,alt='Selected SceneMind video at the chosen result timestamp',width=200 if info['height']>info['width'] else 'stretch')
                play_label=(f'Playing from {timestamp(at)}' if float(at).is_integer() else f'Selected {timestamp(at)} · playback starts at {timestamp(int(at))}')
                st.caption((overlay['label']+' · ' if overlay else '')+(play_label if selected is not None or overlay else 'Select a result to jump to that moment.'))
                st.caption(f"{timestamp(info['duration'])} duration · {info['width']} × {info['height']}")
                with st.container(horizontal=True,gap='small'):
                    st.button('Previous',icon=':material/skip_previous:',key='sm_previous',disabled=not rows or selected is None or selected<=0,on_click=play,args=(run['id'],0 if selected is None else selected-1))
                    st.button('Next',icon=':material/skip_next:',key='sm_next',disabled=not rows or (selected is not None and selected>=len(rows)-1),on_click=play,args=(run['id'],0 if selected is None else selected+1))
                if scene:unified.footer(scene,plan)
                attribution=Path(video['source']).with_name(Path(video['source']).stem+'_attribution.json')
                if attribution.exists():
                    try:
                        a=read_json(attribution);st.caption('Credit: '+a.get('creator','')+' · '+a.get('license',''))
                    except (OSError,ValueError):pass
            else:
                st.subheader('Your video')
                st.info('Choose a saved video or add an MP4 to begin.',icon=':material/video_library:')
    with results_column:
        if run['status']=='error':st.error(run['error'])
        if run['status']=='done':
            result=run['result'];rows=result['results'];st.subheader(f"Results for “{run['query']}”")
            if scene:
                summary=match_summary(plan,scene)
                st.caption(('Uncertain clothing candidates: ' if plan.get('colors') else 'Structured temporal/track matches: ')+(', '.join(summary['track_ids']) or 'None')+'. No independently verified matches.')
                if plan.get('colors'):
                    st.warning('Color and garment type (shirt versus dress) are unverified. Amber boxes mark uncertain torso-color candidates only; detection and CLIP do not prove clothing attributes.')
                    with st.expander('Localized clothing evidence'):
                        for observation in scene.get('attributes',{}).get('observations',[]):
                            if observation['track_id'] in summary['track_ids'] and observation['candidate_colors']:
                                st.caption(observation['track_id']+' / '+timestamp(observation['timestamp'])+' / torso pixel fractions '+json.dumps(observation['fractions']))
                                if Path(observation['preview']).is_file():st.image(observation['preview'],width=144,alt='Detected person crop with the torso analysis region marked')
                mode=st.session_state.sm_mode or 'CLIP-only'
                if mode=='CLIP-only':st.info('CLIP-only ranks full-frame appearance. It cannot verify actions or clothing. The unified player applies separate cached YOLO/temporal evidence; any clothing cues remain uncertain.')
                elif mode=='Regions':st.info('Regions ranks localized crop appearance. Crop similarity cannot verify actions or clothing. The unified player applies separate cached temporal evidence and uncertain torso-color cues.')
                else:st.info('Regions + Evidence combines crop ranking and structured evidence. Temporal predictions are model-supported, not independently verified; clothing colors remain uncertain.')
                if plan['reasons']:st.info('No supported structured matches: '+', '.join(plan['reasons']))
                for i,event in enumerate(plan['intervals']):
                    with st.container(border=True):
                        st.caption(event['person_track_id']+' / '+event['action']+' (model) / '+timestamp(event['start'])+' - '+timestamp(event['end']))
                        st.button('Jump to interval',key=f"u_match_{run['id']}_{i}",on_click=unified.seek,args=(event['start'],))
            st.caption(result['ui_mode']+' · Visual similarity and ranking scores are not probabilities.')
            with st.expander('Appearance-only results (unverified)',expanded=not bool(scene and (plan['action'] or plan['track_ids'] or plan['reasons'] or plan.get('colors'))),key='sm_visual_results'):
                st.caption('These are appearance comparisons, not matches to every query constraint. Legacy region track references belong to a separate detection run; they are not the unified Person IDs above.')
                if not rows:
                    st.info('No supported matches found. Try a different description, or allow visual candidates in Advanced Details.',icon=':material/search_off:')
                else:
                    if result['query'].get('action') or result['query'].get('unsupported_constraints'):st.caption('Visual candidates may be shown; the requested action or attributes are not verified.')
                    for rank,row in enumerate(rows[:run['visible']]):
                        label=label_result(row,result['query'])
                        with st.container(border=True,key=f"result_{run['id']}_{rank}"):
                            image_column,text_column=st.columns([1,3],gap='small',vertical_alignment='center',wrap=True)
                            with text_column:targets=highlights.target_picker(run,rank)
                            preview=Path(row['preview'])
                            if targets:
                                try:preview=highlights.preview(video['source'],row,targets,DATA) or preview
                                except Exception as error:st.warning('Target preview unavailable: '+friendly_error(error))
                            if preview.is_file():image_column.image(thumbnail(str(preview),preview.stat().st_mtime_ns),alt=f"Video thumbnail at {timestamp(row['timestamp'])}",width=144)
                            with text_column:
                                with st.container(horizontal=True,gap='small',vertical_alignment='center'):
                                    st.markdown('**'+timestamp(row['timestamp'])+'**');st.badge(label.status,color=label.color)
                                st.caption(' + '.join(t['label'] for t in targets) if targets else 'No specific target localized.')
                                if not targets and row.get('localization',{}).get('reason'):st.caption(row['localization']['reason'])
                                st.caption(label.kind+' · '+label.description)
                                if targets:st.caption('Localized visual candidate / attributes/actions/relationships unverified. Region score '+', '.join(f"{t['region_cosine']:.3f}" for t in targets))
                                with st.container(horizontal=True,gap='small',vertical_alignment='center'):
                                    st.caption(f"Similarity {row['clip_cosine']:.3f} · Ranking {row['rank_score']:.3f}")
                                    st.button('Play',key=f"play_{run['id']}_{rank}",icon=':material/play_arrow:',type='primary' if selected==rank else 'secondary',on_click=play,args=(run['id'],rank))
                                st.button('Play Highlighted Video',key=f"highlight_{run['id']}_{rank}",disabled=not targets,on_click=highlights.play_highlight,args=(run['id'],rank),icon=':material/center_focus_strong:')
                    if run['visible']<len(rows):st.button('Show 5 more',key='sm_more',on_click=more,args=(run['id'],),width='stretch')
                    st.caption(f"Showing {min(run['visible'],len(rows))} of {len(rows)} moments.")
        elif run['status']!='error':
            st.subheader('Search moments')
            st.info('Describe what you want to find. Your latest search will appear here.',icon=':material/search:')

with technical_slot.container():
    st.caption('Verified is reserved for independently reviewed evidence. Model detections, visual similarity and possible events do not receive that label.')
    if run['status'] in ('done','error'):
        st.json({'execution_id':run['id'],'status':run['status'],'query':run['query'],'context':json.loads(run['context']),'error':run.get('debug_error'),'result':run.get('result')},expanded=False)
        if run.get('result'):st.download_button('Download latest search JSON',json.dumps(run['result'],indent=2),file_name='scenemind-search.json',mime='application/json',key='sm_download_'+run['id'])
    else:st.caption('No completed search in the current video and mode.')
