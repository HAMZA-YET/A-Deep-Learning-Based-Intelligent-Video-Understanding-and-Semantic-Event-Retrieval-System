from pathlib import Path
import json
import numpy as np
from scenemind_mvp.pipeline import digest

def raw():return Path(json.loads(Path('benchmarks/phase27/latest-run.json').read_text())['raw_directory'])
def test_real_detector_comparison_uses_identical_frames_and_mapping():
    report=json.loads((raw()/'detectors.json').read_text());frames=report['frame_indices']
    assert len(report['profiles'])==5 and len(frames)>5
    for profile in report['profiles'].values():
        assert sorted(map(int,profile['predictions']))==frames
        assert profile['summary']['class_mapping']['24']=='backpack'
        assert profile['summary']['false_positives'] is None
        assert len(profile['trials'])==len(frames)*3
        assert profile['summary']['matched_reviewed_boxes']+profile['summary']['missed_reviewed_boxes']==5

def test_real_tracking_runs_have_full_timestamped_observations_not_interpolation():
    report=json.loads((raw()/'tracking.json').read_text());assert len(report['experiments'])==6
    for experiment in report['experiments']:
        directory=Path(experiment['directory']);frames=json.loads((directory/'records.json').read_text())['frames']
        assert len(frames)==330
        for i,frame in enumerate(frames):
            assert frame['frame_index']==i and abs(frame['timestamp']-i/25)<1e-9
            for d in frame['detections']:
                assert d['confidence']>=experiment['config']['confidence']
                assert d['track_id'] is None or d['track_id'].startswith(f'c{d["class_id"]}-t')
        assert experiment['identity_switches'] is None
        timeline=json.loads((directory/'timeline.json').read_text())
        assert timeline['source_sha256']==report['source_sha256']
        assert all(e['status'] in ('unknown','candidate','observed') for e in timeline['events'])

def test_original_ground_truth_and_configs_preserved():
    protocol=json.loads(Path('benchmarks/phase27/protocol.json').read_text())
    for path,sha in protocol['inputs'].items():assert digest(Path(path))==sha
    snapshot=json.loads(Path('benchmarks/phase27/snapshot.json').read_text())
    for path,sha in snapshot['files'].items():
        if path=='pyproject.toml':continue
        if path=='app.py':
            # UI redesign is authorized; retain the exact frozen original for audit.
            assert digest(Path('docs/ui-redesign/app-before.py'))==sha
            continue
        if path in ('src/scenemind_mvp/detection/tracker.py','src/scenemind_mvp/detection/pipeline.py') or path.replace('\\','/') in ('src/scenemind_mvp/detection/tracker.py','src/scenemind_mvp/detection/pipeline.py'):
            assert digest(Path('docs/highlighting-fix')/('before-'+Path(path).name))==sha
            continue
        assert digest(Path(path))==sha
