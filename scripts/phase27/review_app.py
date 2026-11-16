"""Standalone, source-only manual review UI; exports labels without overwriting originals."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import json
import cv2
from PIL import Image,ImageDraw
import streamlit as st
from scenemind_mvp.recovery.annotation import add_box,add_interval,label_query
from scenemind_mvp.evaluation.metrics import validate_review
ROOT=Path(__file__).resolve().parents[2]
st.set_page_config(page_title='SceneMind human annotation',layout='wide')
st.title('Independent human annotation')
st.caption('Source video and human labels only. No model predictions are loaded. Temporary manual references are not real identities. Original files are never overwritten.')
manifest=json.loads((ROOT/'benchmarks/phase26/manifest.json').read_text(encoding='utf-8'))
video_id=st.selectbox('Video',[v['video_id'] for v in manifest['videos']]);video=next(v for v in manifest['videos'] if v['video_id']==video_id);meta=video['metadata']
key='review-'+video_id
if key not in st.session_state:st.session_state[key]=json.loads((ROOT/video['annotation_file']).read_text(encoding='utf-8'))
review=st.session_state[key]
imported=st.file_uploader('Resume a manually edited review JSON',type=['json'])
if imported and st.button('Load uploaded review'):
    candidate=json.loads(imported.getvalue());validate_review(candidate,video['sha256'],meta['duration']);st.session_state[key]=candidate;st.rerun()
st.video(str(ROOT/video['path']))
st.write(f"{meta['width']}x{meta['height']}, {meta['fps']} FPS, {meta['duration']} seconds. Coordinates below use source pixels.")
frame_index=int(st.number_input('Frame index',min_value=0,max_value=meta['frame_count']-1,value=0,key='frame-'+video_id))
cap=cv2.VideoCapture(str(ROOT/video['path']));cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,frame=cap.read();cap.release()
if not ok:st.error('Source frame could not be decoded');st.stop()
im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(im)
for row in review['detection_frames']:
    if row['frame_index']==frame_index:
        for b in row['boxes']:draw.rectangle(b['bbox_xyxy'],outline='lime',width=3)
st.image(im,caption=f'Source frame {frame_index}; timestamp {frame_index/meta["fps"]:.3f}s. Green boxes, if any, were entered by a human.')
with st.form('box-entry'):
    classes={'person':0,'backpack':24,'handbag':26,'suitcase':28,'chair':56};label=st.selectbox('Manual object class',list(classes));reference=st.text_input('Manual local reference (optional)')
    columns=st.columns(4);coords=[c.number_input(name,min_value=0.,max_value=float(meta['width'] if i%2==0 else meta['height']),value=0.) for i,(c,name) in enumerate(zip(columns,['x1','y1','x2','y2']))]
    if st.form_submit_button('Add human box'):
        try:st.session_state[key]=add_box(review,frame_index,meta['fps'],meta['width'],meta['height'],classes[label],coords,reference);st.rerun()
        except ValueError as e:st.error(str(e))
with st.form('frame-complete'):
    exhaustive=st.multiselect('Categories exhaustively reviewed in this frame',list(classes));confirmed=st.checkbox('I manually checked every instance of these categories')
    if st.form_submit_button('Mark frame reviewed'):
        if not confirmed or not exhaustive:st.error('Explicit exhaustive review is required')
        else:
            out=deepcopy(review);row=next((r for r in out['detection_frames'] if r['frame_index']==frame_index),None)
            if row is None:row={'frame_index':frame_index,'timestamp':frame_index/meta['fps'],'boxes':[]};out['detection_frames'].append(row)
            row.update(reviewed=True,exhaustive_classes=[classes[c] for c in exhaustive]);out.update(review_status='pending',reviewer=None,reviewed_utc=None,independent_of_model_predictions=None);st.session_state[key]=out;st.rerun()
with st.form('interval-entry'):
    kind=st.selectbox('Annotation kind',['events','relationships']);action=st.selectbox('Action or relationship',['approaching','moving_together','stationary_near_object','possible_pickup','possible_carrying','possible_departure','nearby','possibly_carrying','unknown']);verdict=st.selectbox('Evidence verdict',['ambiguous','positive','negative'])
    start=st.number_input('Start seconds',min_value=0.,max_value=float(meta['duration']),value=0.);end=st.number_input('End seconds',min_value=0.,max_value=float(meta['duration']),value=0.)
    person=st.text_input('Manual person reference');objref=st.text_input('Manual object reference');category=st.selectbox('Object category',['backpack','handbag','suitcase','chair']);notes=st.text_area('Observed evidence / uncertainty (no ownership inference)')
    if st.form_submit_button('Add human interval'):
        try:st.session_state[key]=add_interval(review,kind,action,verdict,start,end,person,objref,category,notes);st.rerun()
        except ValueError as e:st.error(str(e))
st.subheader('Manual query relevance')
with st.form('query-entry'):
    query_index=st.selectbox('Query to review',range(len(review['queries'])),format_func=lambda i:review['queries'][i]['query'])
    query_verdict=st.selectbox('Query verdict',['ambiguous','positive','negative'])
    qa=st.number_input('Relevant interval start',min_value=0.,max_value=float(meta['duration']),value=0.)
    qb=st.number_input('Relevant interval end',min_value=0.,max_value=float(meta['duration']),value=0.)
    qnote=st.text_area('Query review evidence / ambiguity')
    st.caption('Positive adds this interval; negative means the entire clip was reviewed and has no supported interval; ambiguous leaves relevance unknown.')
    if st.form_submit_button('Save human query label'):
        try:st.session_state[key]=label_query(review,query_index,query_verdict,qa,qb,qnote);st.rerun()
        except ValueError as e:st.error(str(e))
st.subheader('Query relevance and continuous tracking labels')
st.caption('Edit the JSON below to enter query verdicts/relevant intervals or continuous tracking sequences using Phase 2.6 guidelines. Boxes and intervals above can also be corrected here. Saving resets the signature to pending.')
text=st.text_area('Human annotation JSON',json.dumps(review,indent=2),height=300,key='json-'+video_id+'-'+str(len(json.dumps(review))))
if st.button('Apply edited annotation JSON'):
    try:
        candidate=json.loads(text);candidate.update(review_status='pending',reviewer=None,reviewed_utc=None,independent_of_model_predictions=None);validate_review(candidate,video['sha256'],meta['duration']);st.session_state[key]=candidate;st.rerun()
    except (ValueError,KeyError) as e:st.error(str(e))
with st.form('sign-review'):
    reviewer=st.text_input('Human reviewer name or initials');independent=st.checkbox('I reviewed the complete source clip independently of model predictions');second=st.text_input('Second reviewer / adjudication note (recommended for action positives)')
    if st.form_submit_button('Validate and sign completed review'):
        try:
            if not reviewer.strip() or not independent:raise ValueError('Named independent review is required')
            candidate=deepcopy(review);candidate.update(review_status='complete',reviewer=reviewer.strip(),reviewed_utc=datetime.now(timezone.utc).isoformat(),independent_of_model_predictions=True);candidate['adjudication_note']=second
            validate_review(candidate,video['sha256'],meta['duration']);st.session_state[key]=candidate;st.rerun()
        except (ValueError,KeyError) as e:st.error(str(e))
st.write('Review status:',review['review_status'])
st.download_button('Download human review JSON',json.dumps(review,indent=2),file_name=f'{video_id}-review.json',mime='application/json')
st.info('Download and manually place the reviewed file in a separate review directory. Reproducible evaluation accepts that directory explicitly; pending original files stay intact.')
