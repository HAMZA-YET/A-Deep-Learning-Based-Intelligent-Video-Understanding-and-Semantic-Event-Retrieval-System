"""Source/provenance fixture caches; these files are not videos or model predictions."""
from pathlib import Path
import json
import numpy as np
import pytest
from scenemind_mvp.pipeline import digest
from scenemind_mvp.hybrid.evidence import load_bundle

def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')

@pytest.fixture
def bundle(tmp_path):
    source=tmp_path/'source.mp4';source.write_bytes(b'explicit provenance test fixture')
    sha=digest(source)
    semantic=tmp_path/'semantic';detection=tmp_path/'detection';association=tmp_path/'association';temporal=tmp_path/'temporal'
    for p in (semantic,detection,association,temporal):p.mkdir()
    np.save(semantic/'embeddings.npy',np.eye(2,dtype=np.float32));np.save(semantic/'timestamps.npy',np.array([0,1]))
    write(semantic/'manifest.json',{'identity':{'schema':2,'encoder_revision':'openai-quickgelu-v1','video_sha256':sha,'config':{'sampling_fps':1}},'sample_count':2,'dimension':2})
    identity={'video_sha256':sha,'config':{'test_only':True}}
    write(detection/'manifest.json',{'identity':identity})
    write(detection/'records.json',{'frames':[]})
    write(association/'associations.json',{'events':[]});write(association/'diagnostics.json',{'tracks':[]})
    write(association/'manifest.json',{'detection_directory':str(detection),'identity':{'video_sha256':sha,'detection_identity':identity,'records_sha256':digest(detection/'records.json')}})
    write(temporal/'timeline.json',{'events':[]})
    write(temporal/'manifest.json',{'association_directory':str(association),'identity':{'source_sha256':sha,
        'association_sha256':digest(association/'associations.json'),'diagnostics_sha256':digest(association/'diagnostics.json')}})
    target=tmp_path/'bundle.json';write(target,{'source':str(source),'source_sha256':sha,'semantic_directory':str(semantic),
        'detection_directory':str(detection),'association_directory':str(association),'temporal_directory':str(temporal)})
    return target,detection,association,temporal

def test_valid_fixture_bundle(bundle):
    index,evidence=load_bundle(bundle[0]);assert len(index.search_index.timestamps)==2 and not evidence.frames

def test_cross_video_rejected(bundle):
    path=bundle[1]/'manifest.json';m=json.loads(path.read_text());m['identity']['video_sha256']='wrong';write(path,m)
    with pytest.raises(ValueError,match='Cross-video'):load_bundle(bundle[0])

def test_changed_config_rejected(bundle):
    path=bundle[1]/'manifest.json';m=json.loads(path.read_text());m['identity']['config']['test_only']=False;write(path,m)
    with pytest.raises(ValueError,match='configuration'):load_bundle(bundle[0])

def test_changed_quality_rejected(bundle):
    write(bundle[2]/'diagnostics.json',{'tracks':[{'track_id':'invented'}]})
    with pytest.raises(ValueError,match='quality'):load_bundle(bundle[0])

def test_changed_records_rejected(bundle):
    write(bundle[1]/'records.json',{'frames':[{'changed':True}]})
    with pytest.raises(ValueError,match='records changed'):load_bundle(bundle[0])
