"""Existing offline processing capabilities, kept behind Advanced Details."""
from pathlib import Path
from dataclasses import replace
import json
import streamlit as st
from .catalog import read_json
from .service import source_info,semantic_config,encoder_for


@st.cache_resource(max_entries=4,show_spinner=False)
def detector_resource(settings_json: str,cache: str):
    from ..detection.schema import DetectionConfig
    from ..detection.detector import Detector
    d=json.loads(settings_json);d['classes']=tuple(d['classes'])
    return Detector(DetectionConfig(**d),Path(cache))

def render_tools(video: dict|None,settings: dict,root: Path,data: Path) -> None:
    st.caption('These tools call the existing processing pipelines. Track IDs are temporary; proximity does not establish ownership or contact.')
    if video is None:st.info('Choose a video before preparing evidence.');return
    all_tools=st.session_state.setdefault('sm_tools',{});tools=all_tools.setdefault(video['id'],{})
    if st.button('Prepare full-frame index',key='tool_index'):
        try:
            from ..pipeline import index_video
            encoder,lock=encoder_for(settings,data)
            with st.spinner('Preparing or reusing the frame index…'):
                with lock:index=index_video(Path(video['source']),data/'indexes',semantic_config(settings),lambda:encoder)
            tools['index']=str(index.directory);st.success('Frame index ready. Existing compatible embeddings were reused.')
        except Exception as error:st.error(str(error))
    enabled=st.checkbox('Enable object detection',key='tool_detection')
    if not enabled:return
    from ..detection.schema import DetectionConfig
    from ..detection.pipeline import process_video,load_run
    tracking=st.checkbox('Enable temporary object tracking',value=True,key='tool_tracking')
    cols=st.columns(3)
    confidence=cols[0].slider('Detection confidence',.05,.95,.25,.05,key='tool_confidence')
    stride=cols[1].number_input('Process every Nth frame',min_value=1,max_value=30,value=1,key='tool_stride')
    size=cols[2].selectbox('Detector image size',[320,480,640,960,1280],index=2,key='tool_size')
    profile=st.selectbox('Tracker profile',['Baseline','Continuity experiment','Low-init experiment'],key='tool_profile')
    device=st.selectbox('Detection device',['auto','cuda','cpu'],key='tool_device')
    classes={0:'person',24:'backpack',26:'handbag',28:'suitcase',56:'chair'}
    chosen=st.multiselect('COCO classes',list(classes),default=list(classes),format_func=classes.get,key='tool_classes')
    if st.button('Open saved evidence',key='tool_existing',disabled=not video['bundles']):
        try:
            b=read_json(Path(video['bundles'][0]));tools['detection']=load_run(Path(b['detection_directory']))
            tools['association']={'directory':b['association_directory'],'manifest':read_json(Path(b['association_directory'])/'manifest.json')}
            tools['temporal']={'directory':b['temporal_directory'],'manifest':read_json(Path(b['temporal_directory'])/'manifest.json')}
        except Exception as error:st.error(str(error))
    if st.button('Process detection / tracking',key='tool_process'):
        st.session_state.pop('sm_overlay',None)
        tools.clear()
        try:
            if source_info(Path(video['source']))['id']!=video['id']:raise ValueError('Source video changed')
            d=read_json(root/'configs'/{'Baseline':'detection.json','Continuity experiment':'tracker-continuity.json','Low-init experiment':'tracker-low-init.json'}[profile]);d.update(confidence=confidence,every_n_frames=int(stride),image_size=size,device=device,tracking=tracking,classes=tuple(chosen))
            with st.spinner('Processing detection and tracking…'):tools['detection']=process_video(Path(video['source']),data/'detection',DetectionConfig(**d),detector_factory=lambda config,cache:detector_resource(json.dumps(config.to_dict(),sort_keys=True),str(cache)))
        except Exception as error:st.error(str(error))
    result=tools.get('detection')
    if not result:return
    if result['manifest']['identity']['video_sha256']!=video['id']:st.error('Saved evidence belongs to another video.');return
    directory=Path(result['directory']);st.json(result['manifest'],expanded=False)
    if st.button('Preview detection overlay',key='tool_preview_detection'):st.session_state['sm_overlay']={'video_id':video['id'],'source':str(directory/'annotated.mp4'),'label':'Detection and tracking overlay'}
    st.download_button('Download detection/tracking JSON',(directory/'records.json').read_bytes(),file_name='scenemind-tracks.json',mime='application/json',key='tool_tracks_download')
    from ..association.diagnostics import diagnose
    if st.button('Measure tracking quality',key='tool_quality'):
        frames=read_json(directory/'records.json')['frames'];m=result['manifest'];tools['quality']=diagnose(frames,m['metadata']['fps'],m['identity']['config']['every_n_frames'])
    if 'quality' in tools:st.json(tools['quality'],expanded=False)
    if not st.checkbox('Enable candidate person-object associations',key='tool_associations'):return
    from ..association.core import AssociationConfig
    from ..association.pipeline import process_associations
    near=st.slider('Nearby distance in person heights',.05,.8,.35,.05,key='tool_near')
    if st.button('Build candidate association evidence',key='tool_build_associations'):
        try:
            with st.spinner('Building candidate association evidence…'):tools['association']=process_associations(directory,data/'associations',AssociationConfig(near_distance=near))
            tools.pop('temporal',None)
        except Exception as error:st.error(str(error))
    association=tools.get('association')
    if not association:return
    if Path(association['manifest']['detection_directory']).resolve()!=directory.resolve():return
    assoc_dir=Path(association['directory']);association_data=read_json(assoc_dir/'associations.json');st.json(association_data,expanded=False)
    if st.button('Preview association overlay',key='tool_preview_association'):st.session_state['sm_overlay']={'video_id':video['id'],'source':str(assoc_dir/'annotated.mp4'),'label':'Candidate association overlay'}
    st.download_button('Download association evidence JSON',(assoc_dir/'associations.json').read_bytes(),file_name='scenemind-associations.json',mime='application/json',key='tool_association_download')
    if not st.checkbox('Enable temporal event reasoning',key='tool_temporal'):return
    from ..temporal.core import TemporalConfig,search_events,ACTIONS
    from ..temporal.pipeline import process_timeline
    if st.button('Build temporal event timeline',key='tool_build_timeline'):
        try:
            with st.spinner('Building temporal evidence…'):tools['temporal']=process_timeline(assoc_dir,data/'timelines',TemporalConfig(**read_json(root/'configs/temporal.json')))
        except Exception as error:st.error(str(error))
    temporal=tools.get('temporal')
    if not temporal:return
    if Path(temporal['manifest']['association_directory']).resolve()!=assoc_dir.resolve():return
    timeline_dir=Path(temporal['directory']);timeline=read_json(timeline_dir/'timeline.json')
    columns=st.columns(3);action=columns[0].selectbox('Event action',[None]+list(ACTIONS),key='tool_event_action');category=columns[1].selectbox('Event object category',[None]+sorted({e['object_class'] for e in timeline['events']}),key='tool_event_object');status=columns[2].selectbox('Event evidence status',[None,'observed','candidate','unknown'],key='tool_event_status')
    filter_time=st.checkbox('Filter events at timestamp',key='tool_event_time_enabled');at=st.number_input('Event timestamp (seconds)',min_value=0.,value=0.,key='tool_event_time') if filter_time else None
    hits=search_events(timeline,action,category,at,status);st.json(hits,expanded=False)
    ids=[None]+[e['event_id'] for e in hits];selected=st.selectbox('Inspect temporal event evidence',ids,key='tool_event_selection')
    if selected:st.json(next(e for e in hits if e['event_id']==selected))
    if st.button('Preview temporal overlay',key='tool_preview_temporal'):
        st.session_state['sm_overlay']={'video_id':video['id'],'source':str(timeline_dir/'annotated.mp4'),'label':'Temporal evidence overlay','timestamp':next((e['start_timestamp'] for e in hits if e['event_id']==selected),0.)}
    st.download_button('Download temporal timeline JSON',(timeline_dir/'timeline.json').read_bytes(),file_name='scenemind-timeline.json',mime='application/json',key='tool_timeline_download')
