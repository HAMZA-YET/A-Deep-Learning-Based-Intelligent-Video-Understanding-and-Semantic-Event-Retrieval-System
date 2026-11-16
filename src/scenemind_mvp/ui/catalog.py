"""Lightweight catalog discovery; no model, vector index or detection work here."""
from pathlib import Path
import json

def read_json(path: Path) -> dict:return json.loads(path.read_text(encoding='utf-8-sig'))
def catalog(data: Path,extras: list[dict]|None=None) -> list[dict]:
    videos={}
    def item(sha,source):
        if sha not in videos:videos[sha]={'id':sha,'source':str(Path(source).resolve()),'name':Path(source).name,'indexes':[],'regions':[],'bundles':[]}
        return videos[sha]
    for manifest in sorted((data/'indexes').glob('*/manifest.json')):
        try:
            m=read_json(manifest);v=item(m['identity']['video_sha256'],m['source']);v['indexes'].append(str(manifest.parent.resolve()))
        except (OSError,ValueError,KeyError):continue
    for path in sorted((data/'hybrid-bundles').glob('*/bundle.json')):
        try:
            b=read_json(path);v=item(b['source_sha256'],b['source']);v['bundles'].append(str(path.resolve()))
        except (OSError,ValueError,KeyError):continue
    for manifest in sorted((data/'regions').glob('*/manifest.json')):
        try:
            m=read_json(manifest);v=item(m['identity']['source_sha256'],m['source']);v['regions'].append(str(manifest.parent.resolve()))
            if m['bundle'] not in v['bundles']:v['bundles'].append(m['bundle'])
            if m['full_index_directory'] not in v['indexes']:v['indexes'].append(m['full_index_directory'])
        except (OSError,ValueError,KeyError):continue
    for extra in extras or []:
        entry=item(extra['id'],extra['source'])
        if extra.get('name'):entry['name']=extra['name']
        if not Path(entry['source']).is_file() and Path(extra['source']).is_file():entry['source']=extra['source']
    return sorted((v for v in videos.values() if Path(v['source']).is_file()),key=lambda v:v['name'].lower())

def region_choice(video: dict,settings: dict) -> str|None:
    candidates=[]
    for path in video['regions']:
        try:
            m=read_json(Path(path)/'manifest.json');c=m['identity']['region_config'];bundle=read_json(Path(m['bundle']));det_identity=read_json(Path(bundle['detection_directory'])/'manifest.json')['identity']
            if det_identity.get('implementation')!='yolo-bytetrack-source-index-v2':continue
            evidence=det_identity['config']
            if evidence['image_size']!=(640 if settings['evidence_profile']=='Baseline' else 960):continue
            model=m['identity']['full_identity']['config']
            if abs(c['sampling_fps']-settings['sampling_fps'])<1e-6 and all(model[k]==settings[k] for k in ('model','pretrained')) and abs(c['padding']-settings['padding'])<1e-6 and c['detector_sizes']==settings['detector_sizes'] and c['confidence']==settings['detection_confidence']:
                candidates.append(path)
        except (OSError,ValueError,KeyError):continue
    return candidates[0] if candidates else None

def index_choice(video: dict,settings: dict) -> str|None:
    for path in video['indexes']:
        try:
            manifest=read_json(Path(path)/'manifest.json');identity=manifest['identity'];c=identity['config']
            if identity.get('encoder_revision')!='openai-quickgelu-v1' or identity.get('schema')!=2:continue
            if abs(c['sampling_fps']-settings['sampling_fps'])<1e-6 and all(c[k]==settings[k] for k in ('model','pretrained')):return path
        except (OSError,ValueError,KeyError):continue
    return None
