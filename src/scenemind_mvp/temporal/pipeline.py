"""Immutable-source temporal timelines and evidence videos; benchmark labels are never inputs."""
from pathlib import Path
import hashlib,json,tempfile,subprocess,time
import cv2
from ..pipeline import digest
from ..video import inspect_video
from ..detection.pipeline import annotate,find_ffmpeg,validated_timestamp
from .core import TemporalConfig,reason_events

def process_timeline(association_directory: Path,output_root: Path,config: TemporalConfig) -> dict:
    manifest=json.loads((association_directory/'manifest.json').read_text(encoding='utf-8'))
    detection=Path(manifest['detection_directory']);source=Path(manifest['source'])
    if digest(source)!=manifest['identity']['video_sha256']:raise ValueError('Source changed; rebuild detections')
    association_file=association_directory/'associations.json';diagnostic_file=association_directory/'diagnostics.json'
    signature={'revision':'temporal-geometry-v2','source_sha256':digest(source),
        'association_sha256':digest(association_file),'diagnostics_sha256':digest(diagnostic_file),'config':config.to_dict()}
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();destination=output_root/key
    if (destination/'manifest.json').exists():
        if not all((destination/name).is_file() for name in ('timeline.json','annotated.mp4')):raise ValueError('Timeline cache incomplete')
        return {'directory':str(destination.resolve()),'cache_hit':True,'manifest':json.loads((destination/'manifest.json').read_text())}
    started=time.perf_counter()
    frames=json.loads((detection/'records.json').read_text())['frames']
    associations=json.loads(association_file.read_text());diagnostics=json.loads(diagnostic_file.read_text())
    reasoning_start=time.perf_counter();timeline=reason_events(frames,associations,diagnostics,config)
    reasoning_seconds=time.perf_counter()-reasoning_start
    timeline.update(source=str(source.resolve()),source_sha256=signature['source_sha256'],association_directory=str(association_directory.resolve()),
        reasoning_device='cpu',reasoning_seconds=reasoning_seconds)
    output_root.mkdir(parents=True,exist_ok=True);info=inspect_video(source)
    by_frame={f['frame_index']:f for f in frames};events_by_frame={}
    for event in timeline['events']:
        for evidence in event['supporting_observations']:
            events_by_frame.setdefault(evidence['frame_index'],[]).append((event,evidence))
    with tempfile.TemporaryDirectory(prefix='building-',dir=output_root) as temporary:
        staging=Path(temporary);(staging/'timeline.json').write_text(json.dumps(timeline,indent=2,allow_nan=False),encoding='utf-8')
        writer=cv2.VideoWriter(str(staging/'render.avi'),cv2.VideoWriter_fourcc(*'MJPG'),info.fps,(info.width,info.height))
        if not writer.isOpened():raise RuntimeError('Cannot initialize event video writer')
        capture=cv2.VideoCapture(str(source));index=0
        try:
            while True:
                ok,image=capture.read()
                if not ok:break
                validated_timestamp(index,capture.get(cv2.CAP_PROP_POS_MSEC)/1000,info.fps)
                image=annotate(image,by_frame.get(index,{}).get('detections',[]),index in by_frame)
                cv2.putText(image,'TEMPORAL EVIDENCE: 2D geometry / candidates; no ownership/contact',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,200,255),1,cv2.LINE_AA)
                seen=set();line=0
                for event,evidence in events_by_frame.get(index,[]):
                    label=event['action']+' ['+event['status']+'] '+','.join(event['person_track_ids'])
                    if label in seen:continue
                    seen.add(label)
                    if line<5:
                        cv2.putText(image,label,(8,42+line*18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,200,255),1,cv2.LINE_AA);line+=1
                    if evidence.get('person_bbox_xyxy') and event['status']!='unknown':
                        p=evidence['person_bbox_xyxy'];b=evidence['object_bbox_xyxy']
                        cv2.line(image,(round((p[0]+p[2])/2),round((p[1]+p[3])/2)),(round((b[0]+b[2])/2),round((b[1]+b[3])/2)),(40,220,220),2)
                writer.write(image);index+=1
        finally:capture.release();writer.release()
        if index!=info.frame_count:raise ValueError('Incomplete event source decoding')
        subprocess.run([find_ffmpeg(),'-hide_banner','-loglevel','error','-n','-i',str(staging/'render.avi'),
            '-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(staging/'annotated.mp4')],check=True)
        (staging/'render.avi').unlink()
        if inspect_video(staging/'annotated.mp4').frame_count!=index:raise ValueError('Event video length mismatch')
        result={'identity':signature,'source':str(source.resolve()),'association_directory':str(association_directory.resolve()),'summary':timeline['summary'],
            'reasoning_seconds':reasoning_seconds,'total_seconds':time.perf_counter()-started,'frames':index,
            'reasoning_device':'cpu','additional_reasoning_gpu_memory_mib':None,
            'gpu_note':'Rules/rendering run on CPU; detector GPU measurements are stored in the upstream run.',
            'source_attribution':manifest.get('source_attribution')}
        (staging/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        try:staging.rename(destination)
        except OSError:
            if not (destination/'manifest.json').exists():raise
    return {'directory':str(destination.resolve()),'cache_hit':False,'manifest':result}
