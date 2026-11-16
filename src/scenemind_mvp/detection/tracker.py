"""One ByteTrack instance per class; IDs refer only to this processed video."""
from types import SimpleNamespace
import math
import numpy as np
from .schema import DetectionConfig

# Ultralytics 8.3.221 filters high/low confidence batches before init_track.
# Preserve the source row through slicing instead of interpreting subset-local idx.
def indexed_tracker_types():
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    class IndexedBoxes(Boxes):
        def __init__(self,data,orig_shape,source_indices=None):
            super().__init__(data,orig_shape)
            self.source_indices=np.arange(len(self)) if source_indices is None else np.atleast_1d(source_indices)
        def __getitem__(self,index):
            return type(self)(self.data[index],self.orig_shape,self.source_indices[index])
    class IndexedBYTETracker(BYTETracker):
        def init_track(self,results,img=None):
            tracks=super().init_track(results,img)
            for track,index in zip(tracks,results.source_indices):track.idx=int(index)
            return tracks
    return IndexedBoxes,IndexedBYTETracker

class ClassTrackers:
    def __init__(self, config: DetectionConfig, processed_fps: float):
        self.Boxes,BYTETracker=indexed_tracker_types()
        args=SimpleNamespace(track_high_thresh=config.high_threshold,
            track_low_thresh=config.low_threshold,new_track_thresh=config.new_track_threshold,
            track_buffer=max(1,math.ceil(config.lost_seconds*processed_fps)),
            match_thresh=config.match_threshold,fuse_score=True)
        # BYTETracker scales buffer by frame_rate/30; the buffer above already uses processed FPS.
        self.trackers={c:BYTETracker(args,frame_rate=30) for c in config.classes}
        self.ids: dict[tuple[int,int],str]={}
        self.counts={c:0 for c in config.classes}

    def _id(self, class_id: int, raw_id: int) -> str:
        key=(class_id,raw_id)
        if key not in self.ids:
            self.counts[class_id]+=1
            self.ids[key]=f"c{class_id}-t{self.counts[class_id]}"
        return self.ids[key]

    def update(self, detections: np.ndarray, shape: tuple[int,int], image=None) -> tuple[dict[int,str],dict]:
        assignments={}
        states={'active':[],'lost':[],'removed':[]}
        for class_id,tracker in self.trackers.items():
            indexes=np.flatnonzero(detections[:,5]==class_id)
            boxes=self.Boxes(detections[indexes],shape)
            tracks=tracker.update(boxes,image)
            for row in tracks:
                detection_index=int(row[-1])
                if not 0<=detection_index<len(indexes):raise ValueError('Tracker returned an invalid source detection index')
                original=int(indexes[detection_index])
                if original in assignments:raise ValueError('Tracker assigned a detection twice')
                assignments[original]=self._id(class_id,int(row[4]))
            for name,items in (('active',tracker.tracked_stracks),('lost',tracker.lost_stracks),('removed',tracker.removed_stracks)):
                for item in items:
                    key=(class_id,item.track_id)
                    if key in self.ids:
                        states[name].append(self.ids[key])
        return assignments,states
