"""Real installed ByteTrack: source indexes survive high/low score filtering."""
import numpy as np
from scenemind_mvp.detection.tracker import ClassTrackers
from scenemind_mvp.detection.schema import DetectionConfig

def test_high_confidence_subset_index_is_not_original_index():
    tracker=ClassTrackers(DetectionConfig(classes=(0,)),30)
    detections=np.array([[10,10,30,60,.15,0],[100,10,120,60,.9,0]],dtype=np.float32)
    assignments,_=tracker.update(detections,(100,200))
    assert set(assignments)=={1},'The low-score row must not inherit the high-score person ID'

def test_high_and_low_subsets_preserve_two_separate_people():
    tracker=ClassTrackers(DetectionConfig(classes=(0,)),30)
    first=np.array([[10,10,30,60,.9,0],[100,10,120,60,.9,0]],dtype=np.float32)
    original,_=tracker.update(first,(100,200))
    second=np.array([[11,10,31,60,.15,0],[101,10,121,60,.9,0]],dtype=np.float32)
    next_frame,_=tracker.update(second,(100,200))
    assert next_frame==original and next_frame[0]!=next_frame[1]
