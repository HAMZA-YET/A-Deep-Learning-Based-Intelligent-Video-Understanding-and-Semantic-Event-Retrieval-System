"""Load source-bound evidence; never silently combine different videos/configurations."""
from dataclasses import dataclass
from pathlib import Path
import json
from ..pipeline import load_index,digest

@dataclass
class Evidence:
    frames: list[dict]
    associations: list[dict]
    events: list[dict]
    qualities: dict
    source_sha256: str
    provenance: dict

    @classmethod
    def empty(cls,sha: str):return cls([],[],[],{},sha,{})

def load_bundle(path: Path):
    bundle=json.loads(path.read_text(encoding='utf-8'))
    index=load_index(Path(bundle['semantic_directory']))
    sha=index.manifest['identity']['video_sha256']
    if bundle['source_sha256']!=sha:raise ValueError('Bundle semantic/video identity mismatch')
    if digest(Path(bundle['source']))!=sha:raise ValueError('Bundle source video changed')
    detection=Path(bundle['detection_directory']);association=Path(bundle['association_directory']);temporal=Path(bundle['temporal_directory'])
    d=json.loads((detection/'manifest.json').read_text());a=json.loads((association/'manifest.json').read_text());t=json.loads((temporal/'manifest.json').read_text())
    if d['identity']['video_sha256']!=sha or a['identity']['video_sha256']!=sha or t['identity']['source_sha256']!=sha:
        raise ValueError('Cross-video evidence is forbidden')
    if Path(a['detection_directory']).resolve()!=detection.resolve() or Path(t['association_directory']).resolve()!=association.resolve():
        raise ValueError('Evidence pipeline/configuration provenance mismatch')
    if a['identity'].get('detection_identity')!=d['identity']:
        raise ValueError('Detection configuration differs from association provenance')
    if t['identity']['diagnostics_sha256']!=digest(association/'diagnostics.json'):
        raise ValueError('Track quality changed after temporal indexing')
    if a['identity']['records_sha256']!=digest(detection/'records.json') or t['identity']['association_sha256']!=digest(association/'associations.json'):
        raise ValueError('Evidence records changed after indexing')
    frames=json.loads((detection/'records.json').read_text())['frames']
    associations=json.loads((association/'associations.json').read_text())['events']
    events=json.loads((temporal/'timeline.json').read_text())['events']
    quality=json.loads((association/'diagnostics.json').read_text())
    return index,Evidence(frames,associations,events,{r['track_id']:r for r in quality['tracks']},sha,bundle)
