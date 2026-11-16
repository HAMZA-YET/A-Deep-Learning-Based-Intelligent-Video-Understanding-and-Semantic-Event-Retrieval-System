"""Video/JSON integration with explicit artificial detections and actual ByteTrack."""
from pathlib import Path
from types import SimpleNamespace
import json
import cv2
import numpy as np
import pytest
from scenemind_mvp.detection.schema import DetectionConfig
from scenemind_mvp.detection.pipeline import process_video
from scenemind_mvp.association.pipeline import process_associations
from scenemind_mvp.association.core import AssociationConfig
from scenemind_mvp.pipeline import digest

class FixtureDetector:
    def __init__(self,config,cache):
        cache.mkdir(parents=True,exist_ok=True);self.weights=cache/'test.pt';self.weights.write_bytes(b'explicit test double')
        self.torch=SimpleNamespace();self.device='cpu';self.names={0:'person',26:'handbag'};self.frame=0
    def infer(self,image):
        x=self.frame;self.frame+=1
        return np.array([[10+x,5,60+x,95,.9,0],[45+x,45,58+x,65,.9,26]],dtype=np.float32)

@pytest.fixture
def run(tmp_path):
    source=tmp_path/'test.mp4';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(100,100))
    assert writer.isOpened()
    for _ in range(20): writer.write(np.zeros((100,100,3),np.uint8))
    writer.release()
    detection=process_video(source,tmp_path/'detections',DetectionConfig(classes=(0,26)),detector_factory=FixtureDetector)
    return source,Path(detection['directory'])

def test_association_video_json_and_cache(run,tmp_path):
    source,directory=run;before=digest(source)
    result=process_associations(directory,tmp_path/'associations',AssociationConfig())
    target=Path(result['directory']);evidence=json.loads((target/'associations.json').read_text())
    assert evidence['events'][0]['relationship']=='possibly_carrying'
    assert len(evidence['events'][0]['supporting_observations'])==20
    assert digest(source)==before
    capture=cv2.VideoCapture(str(target/'annotated.mp4'));count=0
    while capture.read()[0]: count+=1
    capture.release();assert count==20
    assert process_associations(directory,tmp_path/'associations',AssociationConfig())['cache_hit']
    manifest=json.loads((directory/'manifest.json').read_text())
    manifest['identity']['config']['new_track_threshold']=.4
    (directory/'manifest.json').write_text(json.dumps(manifest))
    changed=process_associations(directory,tmp_path/'associations',AssociationConfig())
    assert changed['directory']!=result['directory']

def test_changed_source_rejected(run,tmp_path):
    source,directory=run;source.write_bytes(source.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Source changed'):
        process_associations(directory,tmp_path/'associations',AssociationConfig())
