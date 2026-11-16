"""Schema/timestamp tests and real ByteTrack state tests with explicit detection fixtures."""
from pathlib import Path
import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from scenemind_mvp.detection.schema import DetectionConfig,Observation
from scenemind_mvp.detection.tracker import ClassTrackers
from scenemind_mvp.detection.pipeline import validated_timestamp,process_video

@pytest.mark.parametrize('kwargs',[{'confidence':0},{'every_n_frames':0},{'image_size':33},{'device':'bad'},
    {'classes':()},{'classes':(80,)},{'low_threshold':.6},{'lost_seconds':-1}])
def test_config_invalid(kwargs):
    with pytest.raises(ValueError): DetectionConfig(**kwargs)

def test_observation_json_roundtrip():
    item=Observation(0,'person',.9,(1,2,10,20),'c0-t1').to_dict()
    assert json.loads(json.dumps(item))['bbox_xyxy']==[1,2,10,20]
    assert item['track_state']=='matched'

@pytest.mark.parametrize('kwargs',[{'confidence':float('nan')},{'bbox_xyxy':(10,0,0,20)},
    {'track_id':'c26-t1'}])
def test_observation_invalid(kwargs):
    values=dict(class_id=0,label='person',confidence=.9,bbox_xyxy=(0,0,20,40))
    values.update(kwargs)
    with pytest.raises(ValueError): Observation(**values)

def test_timestamp_mapping():
    assert validated_timestamp(75,3.,25)==3
    with pytest.raises(ValueError): validated_timestamp(75,4.,25)

def boxes(x=10,score=.9,cls=0):
    return np.array([[x,10,x+20,60,score,cls]],dtype=np.float32)

EMPTY=np.empty((0,6),dtype=np.float32)

def test_actual_bytetrack_continuity_and_occlusion():
    tracker=ClassTrackers(DetectionConfig(classes=(0,)),10)
    first,_=tracker.update(boxes(),(100,100))
    second,_=tracker.update(boxes(11),(100,100))
    assert first[0]==second[0]
    _,states=tracker.update(EMPTY,(100,100))
    assert first[0] in states['lost']
    recovered,_=tracker.update(boxes(12),(100,100))
    assert recovered[0]==first[0]

def test_low_confidence_can_continue_existing_track():
    tracker=ClassTrackers(DetectionConfig(classes=(0,)),10)
    first,_=tracker.update(boxes(),(100,100))
    next_frame,_=tracker.update(boxes(11,score=.15),(100,100))
    assert first[0]==next_frame[0]

def test_lost_track_expires_then_new_id():
    tracker=ClassTrackers(DetectionConfig(classes=(0,),lost_seconds=.2),10)
    first,_=tracker.update(boxes(),(100,100))
    for _ in range(4): tracker.update(EMPTY,(100,100))
    tracker.update(boxes(12),(100,100))
    new,_=tracker.update(boxes(13),(100,100))
    assert new[0]!=first[0]

def test_classes_never_share_an_id():
    tracker=ClassTrackers(DetectionConfig(classes=(0,26)),10)
    records=np.vstack([boxes(),boxes(cls=26)])
    matches,_=tracker.update(records,(100,100))
    assert matches[0].startswith('c0-') and matches[1].startswith('c26-')
    assert matches[0]!=matches[1]

@pytest.fixture
def video(tmp_path):
    path=tmp_path/'test.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(100,100))
    assert writer.isOpened()
    for _ in range(20): writer.write(np.zeros((100,100,3),np.uint8))
    writer.release()
    return path

class FixtureDetector:
    """Explicit deterministic detector double; never imported by production UI."""
    def __init__(self,config,cache):
        self.weights=cache/'fixture.pt'; self.weights.parent.mkdir(parents=True,exist_ok=True)
        self.weights.write_bytes(b'test fixture; not real model weights')
        self.device='cpu'; self.torch=SimpleNamespace(); self.names={0:'person'}
    def infer(self,image): return boxes()

def test_pipeline_records_video_sampling_and_cache(video,tmp_path):
    config=DetectionConfig(every_n_frames=2,tracking=False,classes=(0,))
    source=video.read_bytes()
    result=process_video(video,tmp_path/'runs',config,detector_factory=FixtureDetector)
    directory=Path(result['directory'])
    records=json.loads((directory/'records.json').read_text())['frames']
    assert len(records)==10
    assert [r['timestamp'] for r in records]==pytest.approx(np.arange(0,2,.2),abs=.01)
    assert all(r['detections'][0]['track_id'] is None for r in records)
    capture=cv2.VideoCapture(str(directory/'annotated.mp4'))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT)==20
    capture.release()
    assert video.read_bytes()==source
    cached=process_video(video,tmp_path/'runs',config,detector_factory=FixtureDetector)
    assert cached['cache_hit']
    changed=process_video(video,tmp_path/'runs',DetectionConfig(every_n_frames=4,tracking=False,classes=(0,)),detector_factory=FixtureDetector)
    assert changed['directory']!=result['directory']

def test_pipeline_with_actual_bytetrack(video,tmp_path):
    result=process_video(video,tmp_path/'tracks',DetectionConfig(classes=(0,)),detector_factory=FixtureDetector)
    records=json.loads((Path(result['directory'])/'records.json').read_text())['frames']
    assert {r['detections'][0]['track_id'] for r in records}=={'c0-t1'}
    assert result['manifest']['unique_person_track_ids']==1

def test_failed_run_not_published(video,tmp_path):
    class Broken(FixtureDetector):
        def infer(self,image): raise RuntimeError('intentional')
    with pytest.raises(RuntimeError,match='intentional'):
        process_video(video,tmp_path/'broken',DetectionConfig(),detector_factory=Broken)
    assert not list((tmp_path/'broken').glob('*/manifest.json'))
