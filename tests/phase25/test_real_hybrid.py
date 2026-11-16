"""Opt-in integration with actual pretrained CLIP and previously prepared real YOLO evidence."""
from pathlib import Path
import os,json
import pytest
from scenemind_mvp.config import Config
from scenemind_mvp.model import Encoder
from scenemind_mvp.hybrid.evidence import load_bundle
from scenemind_mvp.hybrid.engine import HybridEngine
from scenemind_mvp.hybrid.schema import HybridConfig

@pytest.mark.skipif(os.getenv('SCENEMIND_RUN_HYBRID')!='1',reason='Set SCENEMIND_RUN_HYBRID=1 for actual pretrained hybrid integration')
def test_actual_real_video_hybrid():
    root=Path(__file__).resolve().parents[2]
    prepared=json.loads((root/'docs/hybrid-real-preparation.json').read_text())
    index,evidence=load_bundle(Path(prepared['bundle']))
    encoder=Encoder(Config(**index.manifest['identity']['config']),root/'data/models')
    result=HybridEngine(index,evidence,HybridConfig(min_gap_seconds=1)).search('Find all people near a bag',encoder.encode_text('Find all people near a bag'))
    assert result['results'] and any(row['evidence']['objects'] for row in result['results'])
    assert result['all_spatial_candidate_track_ids']
    assert all(row['event_claim'] is None for row in result['results'])
    assert all(Path(row['preview']).is_file() for row in result['results'])
