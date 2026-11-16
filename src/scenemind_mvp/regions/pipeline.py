"""Transactional source/config-bound region indexing; streaming frames and crop batches."""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import json,hashlib,tempfile,time,bisect,importlib.metadata
import numpy as np
from PIL import Image
from ..pipeline import digest,index_video,load_index,IndexedVideo
from ..config import Config
from ..model import Encoder
from ..video import inspect_video,iter_frames
from ..search import normalize
from ..detection.schema import DetectionConfig
from ..detection.detector import Detector
from ..hybrid.evidence import load_bundle
from .schema import RegionConfig
from .crops import padded_box,square_view,merge_proposals,track_reference,context_boxes

class RegionIndex:
    def __init__(self,directory: Path,manifest: dict,full: IndexedVideo,records: list[dict],vectors: np.ndarray):
        self.directory=directory;self.manifest=manifest;self.full=full;self.records=records;self.vectors=vectors
    def preview(self,region_id: int) -> Path:return self.directory/'previews'/f'{region_id:08d}.jpg'

def load_regions(directory: Path) -> RegionIndex:
    m=json.loads((directory/'manifest.json').read_text(encoding='utf-8'));identity=m['identity']
    if identity['revision']!='region-native-square-v1':raise ValueError('Unsupported region cache')
    if digest(Path(m['source']))!=identity['source_sha256']:raise ValueError('Region source changed')
    if digest(Path(m['bundle']))!=identity['bundle_sha256']:raise ValueError('Structured bundle changed')
    if any(digest(Path(p))!=sha for p,sha in identity['structured_provenance'].items()):raise ValueError('Region track/evidence provenance changed')
    full=load_index(Path(m['full_index_directory']))
    if full.manifest['identity']['video_sha256']!=identity['source_sha256'] or full.manifest['identity']!=identity['full_identity']:raise ValueError('Region/full embedding model or source mismatch')
    rows=json.loads((directory/'regions.json').read_text(encoding='utf-8'));vectors=np.load(directory/'embeddings.npy',allow_pickle=False)
    if digest(directory/'regions.json')!=m['regions_sha256'] or digest(directory/'embeddings.npy')!=m['embeddings_sha256']:raise ValueError('Region cache modified or corrupted')
    if vectors.shape!=(len(rows),m['dimension']) or not np.isfinite(vectors).all():raise ValueError('Invalid region embedding shape')
    if len(vectors) and not np.allclose(np.linalg.norm(vectors,axis=1),1.,atol=1e-4):raise ValueError('Region embeddings must be normalized')
    times=set(float(t) for t in full.search_index.timestamps)
    if any(r['region_id']!=i or r['timestamp'] not in times for i,r in enumerate(rows)):raise ValueError('Region timestamp/ID mapping invalid')
    if any(not (directory/'previews'/r['preview']).is_file() for r in rows):raise ValueError('Region preview cache incomplete')
    return RegionIndex(directory,m,full,rows,vectors)

def build_regions(bundle: Path,root: Path,config: RegionConfig,encoder: Encoder|None=None) -> tuple[RegionIndex,bool]:
    started=time.perf_counter();base,evidence=load_bundle(bundle);source=Path(evidence.provenance['source']);info=inspect_video(source)
    semantic=replace(Config(**base.manifest['identity']['config']),sampling_fps=config.sampling_fps,batch_size=config.batch_size)
    if encoder is None:encoder=Encoder(semantic,root.parent/'models')
    full=index_video(source,root.parent/'indexes',semantic,lambda:encoder)
    weights=root.parent/'models/yolo/yolo11n.pt'
    identity={'revision':'region-native-square-v1','source_sha256':digest(source),'bundle_sha256':digest(bundle),'structured_provenance':{str(p):digest(p) for p in [Path(evidence.provenance['detection_directory'])/'records.json',Path(evidence.provenance['association_directory'])/'diagnostics.json']},'full_identity':full.manifest['identity'],'region_config':config.to_dict(),'detector_weights_sha256':digest(weights),'ultralytics':importlib.metadata.version('ultralytics')}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();destination=root/key
    if (destination/'manifest.json').is_file():return load_regions(destination),True
    root.mkdir(parents=True,exist_ok=True);detector=Detector(DetectionConfig(confidence=config.confidence,image_size=config.detector_sizes[0],tracking=False),weights.parent)
    frames={f['frame_index']:f for f in evidence.frames};rows=[];vectors=[];batch=[];prediction_seconds=embedding_seconds=0.;raw_proposals=0;full_times=full.search_index.timestamps.tolist();decoded_samples=[]
    with tempfile.TemporaryDirectory(prefix='building-',dir=root) as temporary:
        staging=Path(temporary);(staging/'previews').mkdir()
        def flush():
            nonlocal embedding_seconds
            if not batch:return
            start=time.perf_counter();matrix=normalize(encoder.encode_images([square_view(im) for im in batch]));embedding_seconds+=time.perf_counter()-start
            if matrix.shape!=(len(batch),full.manifest['dimension']):raise ValueError('Image encoder incompatible with full index')
            vectors.append(matrix);batch.clear()
        for frame in iter_frames(source,config.sampling_fps):
            timestamp=frame.timestamp
            if not any(abs(timestamp-t)<1e-5 for t in full_times):raise ValueError('Region/full sampling mismatch')
            source_index=round(timestamp*info.fps);observations=frames.get(source_index,{}).get('detections',[]);image=np.asarray(frame.image);bgr=np.ascontiguousarray(image[:,:,::-1]);proposals=[]
            for size in config.detector_sizes:
                detector.config=replace(detector.config,image_size=size);start=time.perf_counter();predicted=detector.infer(bgr);prediction_seconds+=time.perf_counter()-start
                for r in predicted:
                    box=np.clip(r[:4],[0,0,0,0],[info.width,info.height,info.width,info.height]).tolist()
                    if box[2]<=box[0] or box[3]<=box[1]:continue
                    proposals.append({'bbox_xyxy':box,'class_id':int(r[5]),'label':detector.names[int(r[5])],'confidence':float(r[4]),'detector_sizes':[size],'kind':'detection'})
            raw_proposals+=len(proposals);proposals=merge_proposals(proposals,config.dedup_iou)
            if config.context_tiles:
                proposals += [{'bbox_xyxy':box,'class_id':None,'label':'context tile (not a detection)','confidence':None,'detector_sizes':[],'kind':'context'} for box in context_boxes(info.width,info.height)]
            for proposal in proposals:
                track,candidates=track_reference(proposal,observations,config.track_match_iou) if proposal['kind']=='detection' else (None,[])
                box=padded_box(proposal['bbox_xyxy'],info.width,info.height,config.padding,config.min_crop_side) if proposal['kind']=='detection' else proposal['bbox_xyxy'];crop=frame.image.crop(box);region_id=len(rows);native_size=list(crop.size);crop.save(staging/'previews'/f'{region_id:08d}.jpg',quality=92)
                area=(proposal['bbox_xyxy'][2]-proposal['bbox_xyxy'][0])*(proposal['bbox_xyxy'][3]-proposal['bbox_xyxy'][1])/(info.width*info.height)
                rows.append({'region_id':region_id,'timestamp':timestamp,'source_frame_index':source_index,**proposal,'crop_bbox_xyxy':box,'native_crop_size':native_size,'source_area_fraction':area,'track_id':track,'candidate_track_ids':candidates,'track_reference_status':'source_frame_unique_iou_match' if track else 'unconfirmed','track_scope':identity['source_sha256'],'preview':f'{region_id:08d}.jpg'})
                batch.append(crop)
                if len(batch)>=config.batch_size:flush()
            decoded_samples.append(timestamp)
        flush();matrix=np.concatenate(vectors) if vectors else np.empty((0,full.manifest['dimension']),dtype=np.float32)
        np.save(staging/'embeddings.npy',matrix,allow_pickle=False);(staging/'regions.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        manifest={'identity':identity,'source':str(source.resolve()),'bundle':str(bundle.resolve()),'full_index_directory':str(full.directory.resolve()),'dimension':full.manifest['dimension'],'region_count':len(rows),'sampled_timestamps':decoded_samples,'raw_proposals':raw_proposals,'index_seconds':time.perf_counter()-started,'prediction_seconds':prediction_seconds,'embedding_seconds':embedding_seconds,'regions_sha256':digest(staging/'regions.json'),'embeddings_sha256':digest(staging/'embeddings.npy'),'device':encoder.device,'encoder_notices':encoder.notices,'limits':['Native source pixels; no super-resolution. CLIP input remains224 square.','YOLO classes/confidence and track references are hypotheses, not human identity or object truth.','Context tiles carry no object label or track.','No cap discards small regions by area.']}
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        if not destination.exists():staging.rename(destination)
    return load_regions(destination),False
