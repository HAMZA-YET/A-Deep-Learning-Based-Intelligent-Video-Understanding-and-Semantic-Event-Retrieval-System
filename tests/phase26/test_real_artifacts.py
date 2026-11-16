import json
from pathlib import Path
import pytest
from scenemind_mvp.hybrid.evidence import load_bundle
from scenemind_mvp.evaluation.retrieval import structured_search

def test_phase26_real_artifacts():
    pointer=Path('benchmarks/phase26/latest-run.json')
    if not pointer.exists():pytest.skip('Run phase26 benchmark_processing.py first')
    processing=json.loads(Path(json.loads(pointer.read_text())['processing_file']).read_text())
    assert len(processing['videos'])==3
    for video in processing['videos']:
        index,evidence=load_bundle(Path(video['bundle']))
        assert index.manifest['sample_count']>0 and evidence.frames
        assert not any(video['cache_hits'].values())
        result=structured_search('Find a person picking up a bag',evidence,index.search_index.timestamps.tolist())
        assert all(r['event_claim']['status'] in ('observed','candidate') for r in result['results'])
        detection=Path(evidence.provenance['detection_directory'])/'annotated.mp4'
        from scenemind_mvp.video import inspect_video
        assert inspect_video(detection).frame_count==video['metadata']['frame_count']
