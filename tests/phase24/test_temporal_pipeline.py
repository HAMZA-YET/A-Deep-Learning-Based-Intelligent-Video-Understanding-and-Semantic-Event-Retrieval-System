"""Temporal video integration from explicit trajectory fixtures, not detector accuracy."""
from pathlib import Path
import json
import cv2
import numpy as np
import pytest
from scenemind_mvp.temporal.pipeline import process_timeline
from scenemind_mvp.temporal.core import TemporalConfig
from scenemind_mvp.pipeline import digest

def source_and_evidence(tmp_path):
    source=tmp_path/'test.mp4';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(100,100))
    assert writer.isOpened()
    for _ in range(20):writer.write(np.zeros((100,100,3),np.uint8))
    writer.release();detection=tmp_path/'detections';detection.mkdir();association=tmp_path/'associations';association.mkdir()
    frames=[];supports=[]
    for i in range(20):
        p=[10+i,5,60+i,95];b=[45+i,45,58+i,65]
        frames.append({'frame_index':i,'timestamp':i/10,'detections':[
            {'class_id':0,'label':'person','confidence':.9,'bbox_xyxy':p,'track_id':'c0-t1'},
            {'class_id':24,'label':'backpack','confidence':.9,'bbox_xyxy':b,'track_id':'c24-t1'}]})
        supports.append({'frame_index':i,'timestamp':i/10,'person_bbox_xyxy':p,'object_bbox_xyxy':b,
            'person_detection_confidence':.9,'object_detection_confidence':.9,'uncertainty_reasons':[]})
    (detection/'records.json').write_text(json.dumps({'frames':frames}))
    (association/'manifest.json').write_text(json.dumps({'source':str(source),'detection_directory':str(detection),
        'identity':{'video_sha256':digest(source)}}))
    (association/'associations.json').write_text(json.dumps({'events':[{'event_id':'fixture','person_track_id':'c0-t1','object_track_id':'c24-t1','object_class':'backpack',
        'relationship':'possibly_carrying','supporting_observations':supports}]}))
    (association/'diagnostics.json').write_text(json.dumps({'tracks':[{'track_id':x,'coverage':1} for x in ('c0-t1','c24-t1')]}))
    return source,association

def test_timeline_json_video_cache_and_provenance(tmp_path):
    source,association=source_and_evidence(tmp_path);before=digest(source)
    result=process_timeline(association,tmp_path/'timelines',TemporalConfig())
    directory=Path(result['directory']);timeline=json.loads((directory/'timeline.json').read_text())
    assert any(e['action']=='moving_together' for e in timeline['events'])
    assert all(e['supporting_observations'] for e in timeline['events'])
    assert digest(source)==before
    capture=cv2.VideoCapture(str(directory/'annotated.mp4'));count=0
    while capture.read()[0]:count+=1
    capture.release();assert count==20
    assert process_timeline(association,tmp_path/'timelines',TemporalConfig())['cache_hit']
    (association/'diagnostics.json').write_text(json.dumps({'tracks':[]}))
    changed=process_timeline(association,tmp_path/'timelines',TemporalConfig())
    assert changed['directory']!=result['directory']

def test_changed_source_refused(tmp_path):
    source,association=source_and_evidence(tmp_path);source.write_bytes(source.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Source changed'):process_timeline(association,tmp_path/'out',TemporalConfig())
