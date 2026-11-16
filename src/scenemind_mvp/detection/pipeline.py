"""Transactional detector/tracker runs; full-length annotated CFR video and streaming JSON."""
from dataclasses import asdict, replace
from pathlib import Path
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import cv2
import numpy as np
from ..pipeline import digest
from ..video import inspect_video
from .schema import DetectionConfig, Observation
from .tracker import ClassTrackers
from .detector import Detector

# Ultralytics BaseTrack IDs and GPU peak counters are process-global. Serialize runs, not CLIP queries.
RUN_LOCK=threading.Lock()

def find_ffmpeg() -> str:
    candidate=os.getenv('FFMPEG_BINARY') or shutil.which('ffmpeg')
    if candidate and Path(candidate).is_file(): return str(candidate)
    base=Path(os.getenv('LOCALAPPDATA',''))/'Microsoft/WinGet/Packages'
    matches=sorted(base.glob('Gyan.FFmpeg*/ffmpeg*/bin/ffmpeg.exe')) if base.is_dir() else []
    if matches: return str(matches[-1])
    raise RuntimeError('FFmpeg not found. Set FFMPEG_BINARY to its executable path')

def validated_timestamp(index: int, timestamp: float, fps: float) -> float:
    expected=index/fps
    if not math.isfinite(timestamp) or timestamp<0 or abs(timestamp-expected)>.5/fps+.001:
        raise ValueError('Output requires CFR video with valid decoded timestamps; transcode a separate source copy first')
    return timestamp

def annotate(image: np.ndarray, observations: list[dict], processed: bool) -> np.ndarray:
    output=image.copy()
    for item in observations:
        x1,y1,x2,y2=map(round,item['bbox_xyxy'])
        color=(40,210,70) if item['track_id'] else (30,180,240)
        cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
        text=f"{item['label']} {item['confidence']:.2f} {item['track_id'] or 'unassigned'}"
        cv2.putText(output,text,(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    if not processed:
        cv2.putText(output,'Frame not processed (sampling)',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,160,255),1)
    return output

def load_run(directory: Path) -> dict:
    manifest=json.loads((directory/'manifest.json').read_text())
    attribution=Path(manifest['source']).with_name(Path(manifest['source']).stem+'_attribution.json')
    if attribution.is_file() and 'source_attribution' not in manifest:
        manifest['source_attribution']=json.loads(attribution.read_text(encoding='utf-8'))
    for name in ('annotated.mp4','records.json'):
        if not (directory/name).is_file(): raise ValueError('Detection cache incomplete; rebuild it')
    return {'directory':str(directory.resolve()),'manifest':manifest,'cache_hit':True}

def process_video(source: Path, output_root: Path, config: DetectionConfig,
                  progress=None, detector_factory=None) -> dict:
    """Process one frame at a time. Original source and Phase 1 indexes are never modified."""
    info=inspect_video(source)
    ffmpeg=find_ffmpeg()
    with RUN_LOCK:
        detector=(detector_factory or Detector)(config,output_root.parent/'models/yolo')
        signature={'schema':1,'implementation':'yolo-bytetrack-source-index-v2',
            'video_sha256':digest(source),'weights_sha256':digest(detector.weights),
            'config':config.to_dict(),'ultralytics':importlib.metadata.version('ultralytics')}
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        destination=output_root/key
        if (destination/'manifest.json').exists():
            return load_run(destination)
        output_root.mkdir(parents=True,exist_ok=True)
        tracker=ClassTrackers(config,info.fps/config.every_n_frames) if config.tracking else None
        torch=detector.torch
        if detector.device.startswith('cuda'):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        counts=Counter(); ids=set(); person_ids=set(); maximum_people=0
        processed_frames=0; frame_index=0; inference_seconds=0.; tracks={}
        with tempfile.TemporaryDirectory(prefix='building-',dir=output_root) as temporary:
            staging=Path(temporary)
            raw=staging/'annotated.avi'
            writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'MJPG'),info.fps,(info.width,info.height))
            if not writer.isOpened(): raise RuntimeError('Cannot initialize annotated video encoder')
            capture=cv2.VideoCapture(str(source))
            try:
                with (staging/'records.json').open('w',encoding='utf-8') as stream:
                    stream.write('{"schema":1,"coordinate_system":"source pixels xyxy","frames":[')
                    while True:
                        ok,image=capture.read()
                        if not ok: break
                        timestamp=validated_timestamp(frame_index,capture.get(cv2.CAP_PROP_POS_MSEC)/1000,info.fps)
                        selected=frame_index%config.every_n_frames==0
                        observations=[]
                        if selected:
                            inference_start=time.perf_counter()
                            candidates=detector.infer(image)
                            inference_seconds+=time.perf_counter()-inference_start
                            assignment,states=tracker.update(candidates,image.shape[:2],image) if tracker else ({},{})
                            for i,row in enumerate(candidates):
                                if row[4]<config.confidence: continue
                                class_id=int(row[5])
                                box=np.clip(row[:4],[0,0,0,0],[info.width,info.height,info.width,info.height])
                                if box[2]<=box[0] or box[3]<=box[1]: continue
                                observation=Observation(class_id,str(detector.names[class_id]),float(row[4]),
                                    tuple(float(x) for x in box),assignment.get(i)).to_dict()
                                observations.append(observation)
                                counts[observation['label']]+=1
                                track_id=observation['track_id']
                                if track_id:
                                    ids.add(track_id)
                                    if class_id==0: person_ids.add(track_id)
                                    summary=tracks.setdefault(track_id,{'class_id':class_id,'label':observation['label'],
                                        'first_timestamp':timestamp,'last_timestamp':timestamp,'observations':0,'gaps':0})
                                    if summary['observations'] and timestamp-summary['last_timestamp']>1.5*config.every_n_frames/info.fps:
                                        summary['gaps']+=1
                                    summary['last_timestamp']=timestamp
                                    summary['observations']+=1
                            maximum_people=max(maximum_people,sum(item['class_id']==0 for item in observations))
                            record={'frame_index':frame_index,'timestamp':timestamp,'detections':observations,'tracker_state':states}
                            if processed_frames: stream.write(',')
                            json.dump(record,stream,allow_nan=False)
                            processed_frames+=1
                        writer.write(annotate(image,observations,selected))
                        if frame_index%max(1,round(info.fps))==0 and progress:
                            progress(min(.98,(frame_index+1)/info.frame_count))
                        frame_index+=1
                    stream.write(']}')
            finally:
                capture.release(); writer.release()
            if frame_index!=info.frame_count: raise ValueError('Source video did not fully decode; no cache published')
            if detector.device.startswith('cuda'): torch.cuda.synchronize()
            processing_seconds=time.perf_counter()-started
            subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-n','-i',str(raw),
                '-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(staging/'annotated.mp4')],check=True)
            raw.unlink()
            output_info=inspect_video(staging/'annotated.mp4')
            if output_info.frame_count!=frame_index: raise ValueError('Annotated video frame count mismatch')
            manifest={'identity':signature,'source':str(source.resolve()),'metadata':asdict(info),
                'device':detector.device,'decoded_frames':frame_index,'processed_frames':processed_frames,
                'processing_seconds':processing_seconds,'end_to_end_seconds':time.perf_counter()-started,
                'processed_fps':processed_frames/processing_seconds,'inference_seconds':inference_seconds,
                'detections_by_class':dict(counts),'person_detection_observations':counts['person'],
                'max_simultaneous_person_detections':maximum_people,'unique_person_track_ids':len(person_ids),
                'unique_track_ids':len(ids),'tracks':tracks,
                'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if detector.device.startswith('cuda') else None,
                'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/1024**2 if detector.device.startswith('cuda') else None,
                'id_switches':None,'id_switches_note':'Not measurable without object-level ground-truth identities',
                'notes':['IDs are temporary video-local tracklets, not real-world identities.',
                    'Lost/removed states are tracker estimates; no boxes are fabricated for occluded objects.',
                    'Unsampled output frames have no boxes; JSON contains only processed frames.',
                    'Audio is omitted; annotated output is CFR and preserves source frame count.']}
            attribution=source.with_name(source.stem+'_attribution.json')
            if attribution.is_file(): manifest['source_attribution']=json.loads(attribution.read_text(encoding='utf-8'))
            (staging/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            staging.rename(destination)
        if progress: progress(1.)
        return {'directory':str(destination.resolve()),'manifest':manifest,'cache_hit':False}
