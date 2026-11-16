"""Explicit opt-in integration test with actual YOLO weights and the existing demo."""
import os
import json
from pathlib import Path
import cv2
import pytest
from scenemind_mvp.detection.schema import DetectionConfig,Observation
from scenemind_mvp.detection.pipeline import process_video
from scenemind_mvp.pipeline import digest

@pytest.mark.skipif(os.getenv('SCENEMIND_RUN_DETECTOR')!='1',reason='Set SCENEMIND_RUN_DETECTOR=1 for actual pretrained demo integration')
def test_pretrained_demo_video(tmp_path):
    root=Path(__file__).resolve().parents[2]
    video=root/'data/demo/scenemind_demo.mp4'
    before=digest(video)
    result=process_video(video,tmp_path/'runs',DetectionConfig())
    manifest=result['manifest']
    assert manifest['decoded_frames']==900 and manifest['processed_frames']==900
    assert digest(video)==before
    records=json.loads((Path(result['directory'])/'records.json').read_text())['frames']
    assert len(records)==900
    assert [row['timestamp'] for row in records]==pytest.approx([i/15 for i in range(900)],abs=.001)
    for row in records:
        for item in row['detections']:
            Observation(item['class_id'],item['label'],item['confidence'],tuple(item['bbox_xyxy']),item['track_id'])
    capture=cv2.VideoCapture(str(Path(result['directory'])/'annotated.mp4'))
    count=0
    while capture.read()[0]: count+=1
    capture.release()
    assert count==900
