"""Cached association evidence and annotated visualization; no source mutation."""
from pathlib import Path
import hashlib
import json
import tempfile
import subprocess
import time
import cv2
from ..pipeline import digest
from ..detection.pipeline import load_run,find_ffmpeg,annotate,validated_timestamp
from ..video import inspect_video
from .core import AssociationConfig,associate
from .diagnostics import diagnose

def process_associations(detection_directory: Path, output_root: Path, config: AssociationConfig, progress=None) -> dict:
    run=load_run(detection_directory);manifest=run['manifest'];source=Path(manifest['source'])
    if digest(source)!=manifest['identity']['video_sha256']: raise ValueError('Source changed since detection; reprocess it')
    records_path=detection_directory/'records.json'
    signature={'revision':'candidate-associations-v4','records_sha256':digest(records_path),
               'video_sha256':manifest['identity']['video_sha256'],'detection_identity':manifest['identity'],'config':config.to_dict()}
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    destination=output_root/key
    if (destination/'manifest.json').exists():
        for name in ('associations.json','diagnostics.json','annotated.mp4'):
            if not (destination/name).is_file(): raise ValueError('Association cache incomplete')
        return {'directory':str(destination.resolve()),'cache_hit':True,
                'manifest':json.loads((destination/'manifest.json').read_text())}
    started=time.perf_counter()
    frames=json.loads(records_path.read_text())['frames']
    metadata=manifest['metadata'];fps=metadata['fps'];stride=manifest['identity']['config']['every_n_frames']
    diagnostics=diagnose(frames,fps,stride)
    associations=associate(frames,diagnostics,config)
    associations['source_detection_directory']=str(detection_directory.resolve())
    associations['track_scope']=digest(records_path)
    associations['assumptions']=['Candidate geometry and temporal consistency only; no ownership or physical contact.',
        'Untracked-object evidence chains are not confirmed object identities.',
        'Fragmented person/object track IDs are never automatically merged.',
        'Missing detections cannot establish absence or an event.']
    relation_map={}
    for event in associations['events']:
        for row in event['supporting_observations']:
            relation_map[(row['frame_index'],row['person_detection_index'],row['object_detection_index'])]=event['relationship']
    output_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='building-',dir=output_root) as temporary:
        staging=Path(temporary)
        (staging/'associations.json').write_text(json.dumps(associations,indent=2,allow_nan=False),encoding='utf-8')
        (staging/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
        detections={row['frame_index']:row for row in frames}
        candidates={row['frame_index']:row['candidates'] for row in associations['frames']}
        writer=cv2.VideoWriter(str(staging/'render.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,(metadata['width'],metadata['height']))
        if not writer.isOpened(): raise RuntimeError('Cannot create association video')
        capture=cv2.VideoCapture(str(source));index=0
        try:
            while True:
                ok,image=capture.read()
                if not ok: break
                validated_timestamp(index,capture.get(cv2.CAP_PROP_POS_MSEC)/1000,fps)
                image=annotate(image,detections.get(index,{}).get('detections',[]),index in detections)
                for candidate in candidates.get(index,[]):
                    box=candidate['object_bbox_xyxy'];center=(round((box[0]+box[2])/2),round((box[1]+box[3])/2))
                    relation=relation_map[(index,candidate['person_detection_index'],candidate['object_detection_index'])]
                    color=(0,210,255) if relation=='unknown' else ((40,180,255) if relation=='nearby' else (80,220,80))
                    if candidate['spatial']:
                        person_center=tuple(round(x) for x in candidate['spatial']['person_center'])
                        cv2.line(image,person_center,center,color,2,cv2.LINE_AA)
                    cv2.putText(image,relation+' candidate',(center[0],max(15,center[1]-10)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
                cv2.putText(image,'CANDIDATES ONLY - no ownership/contact inference',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,200,255),1,cv2.LINE_AA)
                writer.write(image);index+=1
                if progress and index%max(1,round(fps))==0: progress(min(.95,index/metadata['frame_count']))
        finally:
            capture.release();writer.release()
        if index!=metadata['frame_count']: raise ValueError('Association source video failed full decode')
        subprocess.run([find_ffmpeg(),'-hide_banner','-loglevel','error','-n','-i',str(staging/'render.avi'),
            '-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(staging/'annotated.mp4')],check=True)
        (staging/'render.avi').unlink()
        if inspect_video(staging/'annotated.mp4').frame_count!=index: raise ValueError('Output frame count changed')
        result={'identity':signature,'source':str(source.resolve()),'detection_directory':str(detection_directory.resolve()),
                'summary':associations['summary'],'frames':index,'processing_seconds':time.perf_counter()-started,
                'source_attribution':manifest.get('source_attribution'),
                'limitations':associations['assumptions']}
        (staging/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        try: staging.rename(destination)
        except OSError:
            if not (destination/'manifest.json').exists(): raise
    if progress: progress(1.)
    return {'directory':str(destination.resolve()),'manifest':result,'cache_hit':False}
