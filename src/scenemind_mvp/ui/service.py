"""Cached adapters; existing model/index/search implementations are called unchanged."""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import json,time,threading
import streamlit as st
from ..config import Config
from ..pipeline import digest,index_video,load_index
from ..video import inspect_video
from ..model import Encoder
from ..hybrid.evidence import load_bundle,Evidence
from ..hybrid.schema import HybridConfig
from ..hybrid.engine import HybridEngine
from ..hybrid.prepare import prepare_bundle
from ..detection.schema import DetectionConfig
from ..association.core import AssociationConfig
from ..temporal.core import TemporalConfig
from ..regions.pipeline import load_regions,build_regions
from ..regions.schema import RegionConfig
from ..regions.engine import RegionEngine,parse_region_query
from .catalog import read_json,index_choice,region_choice

@st.cache_data(max_entries=128,show_spinner=False)
def inspect_source(path: str,size: int,mtime: int) -> dict:
    p=Path(path);metadata=inspect_video(p)
    return {'id':digest(p),'source':str(p.resolve()),'duration':metadata.duration,'fps':metadata.fps,'width':metadata.width,'height':metadata.height}

def source_info(path: Path) -> dict:
    stat=path.stat();return inspect_source(str(path.resolve()),stat.st_size,stat.st_mtime_ns)

@st.cache_resource(max_entries=4,show_spinner=False)
def model_resource(model: str,pretrained: str,device: str,cache: str,revision: tuple=()):
    return Encoder(Config(model=model,pretrained=pretrained,device=device),Path(cache)),threading.RLock()

def encoder_for(settings: dict,data: Path):
    p=Path(settings['pretrained']);revision=(p.stat().st_size,p.stat().st_mtime_ns) if p.is_file() else ()
    return model_resource(settings['model'],settings['pretrained'],settings['device'],str(data/'models'),revision)

def fingerprints(paths: list[Path]) -> tuple:
    return tuple((str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in paths if p.is_file())

def asset_signature(path: Path,kind: str) -> tuple:
    files=[]
    if kind=='region':
        m=read_json(path/'manifest.json');files += [path/n for n in ('manifest.json','regions.json','embeddings.npy')]
        full=Path(m['full_index_directory']);bundle=Path(m['bundle'])
    elif kind=='bundle':bundle=path;full=Path(read_json(bundle)['semantic_directory'])
    else:full=path;bundle=None
    files += [full/n for n in ('manifest.json','embeddings.npy','timestamps.npy')]
    if bundle:
        b=read_json(bundle);files += [bundle,Path(b['source'])]
        for field,names in [('detection_directory',('manifest.json','records.json')),('association_directory',('manifest.json','associations.json','diagnostics.json')),('temporal_directory',('manifest.json','timeline.json'))]:files += [Path(b[field])/n for n in names]
    else:files.append(Path(read_json(full/'manifest.json')['source']))
    return fingerprints(files)

@st.cache_resource(max_entries=8,show_spinner=False)
def loaded_asset(path: str,kind: str,signature: tuple):
    if kind=='region':
        region=load_regions(Path(path));_,evidence=load_bundle(Path(region.manifest['bundle']));return region,evidence
    if kind=='bundle':return load_bundle(Path(path))
    return load_index(Path(path)),None

def semantic_config(settings: dict) -> Config:
    return Config(sampling_fps=settings['sampling_fps'],batch_size=settings['batch_size'],model=settings['model'],pretrained=settings['pretrained'],device=settings['device'],top_k=settings['max_results'],min_gap_seconds=settings['gap'],min_similarity=settings['cutoff'])

def prepare(video: dict,mode: str,settings: dict,root: Path,data: Path) -> tuple[Path,str]:
    current=source_info(Path(video['source']))
    if current['id']!=video['id'] or digest(Path(video['source']))!=video['id']:raise ValueError('Video changed since selection')
    if mode=='CLIP-only' and not settings['full_hybrid']:
        chosen=index_choice(video,settings)
        if chosen:return Path(chosen),'index'
        encoder,lock=encoder_for(settings,data)
        with lock:index=index_video(Path(video['source']),data/'indexes',semantic_config(settings),lambda:encoder)
        return index.directory,'index'
    chosen=region_choice(video,settings) if not settings['full_hybrid'] else None
    if chosen:return Path(chosen),'region'
    detection_file='detection.json' if settings['evidence_profile']=='Baseline' else 'detector-960.json'
    d=read_json(root/'configs'/detection_file);d['classes']=tuple(d['classes']);detconfig=DetectionConfig(**d)
    bundle=None
    for candidate in video['bundles']:
        try:
            b=read_json(Path(candidate));c=read_json(Path(b['semantic_directory'])/'manifest.json')['identity']['config'];det_identity=read_json(Path(b['detection_directory'])/'manifest.json')['identity'];det=det_identity['config']
            if det_identity.get('implementation')!='yolo-bytetrack-source-index-v2':continue
            if b['source_sha256']==video['id'] and all(c[k]==settings[k] for k in ('model','pretrained')) and det==detconfig.to_dict():bundle=Path(candidate);break
        except (OSError,ValueError,KeyError):continue
    encoder,lock=encoder_for(settings,data)
    if bundle is None:
        with lock:prepared=prepare_bundle(Path(video['source']),data,semantic_config(settings),detconfig,encoder_factory=lambda:encoder,association_config=AssociationConfig(**read_json(root/'configs/association.json')),temporal_config=TemporalConfig(**read_json(root/'configs/temporal.json')))
        bundle=Path(prepared['bundle'])
    if settings['full_hybrid']:return bundle,'bundle'
    regionconfig=RegionConfig(sampling_fps=settings['sampling_fps'],batch_size=settings['batch_size'],padding=settings['padding'],detector_sizes=tuple(settings['detector_sizes']),confidence=settings['detection_confidence'])
    with lock:region,_=build_regions(bundle,data/'regions',regionconfig,encoder)
    return region.directory,'region'

def search(video: dict,mode: str,text: str,settings: dict,root: Path,data: Path) -> dict:
    if not text.strip():raise ValueError('Enter a nonempty query')
    started=time.perf_counter();path,kind=prepare(video,mode,settings,root,data);asset,evidence=loaded_asset(str(path),kind,asset_signature(path,kind))
    index=asset.full if kind=='region' else asset
    if index.manifest['identity']['video_sha256']!=video['id']:raise ValueError('Selected video and cached index mismatch')
    cfg=index.manifest['identity']['config']
    if any(cfg[k]!=settings[k] for k in ('model','pretrained')):raise ValueError('Selected encoder and index mismatch')
    p=Path(cfg['pretrained'])
    if p.is_file() and index.manifest['identity'].get('weights_sha256')!=digest(p):raise ValueError('Model checkpoint changed')
    encoder,lock=encoder_for(settings,data)
    with lock:vector=encoder.encode_text(text)
    parsed=parse_region_query(text);h=replace(HybridConfig(**read_json(root/'configs/hybrid.json')),top_k=settings['max_results'],min_gap_seconds=settings['gap'],strict_events=settings['strict'])
    if kind=='region':
        d=dict(asset.manifest['identity']['region_config']);d['detector_sizes']=tuple(d['detector_sizes'])
        result=RegionEngine(asset,evidence,RegionConfig(**d),h).search(text,vector,'region' if mode=='Regions' else 'region_hybrid',top_k=settings['max_results'],min_gap=settings['gap'],strict_actions=settings['strict'])
    elif kind=='bundle':
        result=HybridEngine(index,evidence,h).search(text,vector,'hybrid');result['query']=parsed.to_dict()
        if settings['strict'] and parsed.action and parsed.action not in ('approaching','moving_together','stationary_near_object','possible_pickup','possible_carrying','possible_departure'):result['results']=[]
    else:
        result=HybridEngine(index,Evidence.empty(video['id']),h).search(text,vector,'clip_only')
        result['query']=parsed.to_dict()
        if settings['strict'] and parsed.action:result['results']=[]
    result['results']=[r for r in result['results'] if r['clip_cosine']>=settings['cutoff']]
    result.update(video_id=video['id'],source=video['source'],query_text=text.strip(),ui_mode=mode+(' (full-frame comparison)' if settings['full_hybrid'] else ''),asset_path=str(path),asset_kind=kind,elapsed_seconds=time.perf_counter()-started,encoder_notices=encoder.notices)
    # Localization is independent of frame ranking and only uses existing ROI embeddings.
    from ..highlighting.targets import localize,attribute_profile,PALETTE
    region=asset if kind=='region' else None
    region_evidence=evidence
    localization_error=None
    if region is None and result['results'] and (parsed.minimum_people or parsed.object_classes or parsed.action):
        # CLIP-only ranking must not silently disable detection/localization on a cache miss.
        # Prepare the same existing detection -> tracking -> native-region pipeline used by Regions.
        try:
            chosen=region_choice(video,settings)
            if chosen is None:
                chosen,_=prepare(video,'Regions',{**settings,'full_hybrid':False},root,data)
            region,region_evidence=loaded_asset(str(chosen),'region',asset_signature(Path(chosen),'region'))
        except (OSError,ValueError,RuntimeError,MemoryError) as error:
            localization_error=str(error)
    result['localization_pipeline']={'revision':'frame-bound-auto-regions-v2','region_directory':str(region.directory) if region is not None else None,'error':localization_error,'ranking_unchanged':True}

    if region is not None:
        profile=attribute_profile(text,result['query']);attributes=None
        if profile:
            attributes=attribute_vectors(settings['model'],settings['pretrained'],settings['device'],str(data),profile['kind'])
        localize(result,region,vector,region_evidence,attributes)
    else:
        for row in result['results']:
            row['localization']={'candidates':[],'primary_target_ids':[],'state':'no_specific_target','message':'No specific target localized.','reason':localization_error or 'No detected object category was requested.'}
    result['elapsed_seconds']=time.perf_counter()-started
    return result

@st.cache_data(max_entries=16,show_spinner=False)
def attribute_vectors(model: str,pretrained: str,device: str,data: str,kind: str) -> dict:
    from ..highlighting.targets import PALETTE
    encoder,lock=encoder_for({'model':model,'pretrained':pretrained,'device':device},Path(data))
    noun='person wearing a {} shirt' if kind=='shirt' else '{} bag'
    # Uncalibrated alternative prompts, never a clothing/color classifier.
    with lock:return {color:encoder.encode_text('a '+noun.format(color)) for color in PALETTE}
