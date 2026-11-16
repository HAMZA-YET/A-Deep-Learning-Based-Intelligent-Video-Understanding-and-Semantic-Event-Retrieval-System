from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from scenemind_mvp.regions.crops import padded_box,square_view,merge_proposals,track_reference,context_boxes
from scenemind_mvp.regions.schema import RegionConfig
from scenemind_mvp.regions.pipeline import RegionIndex,load_regions
from scenemind_mvp.regions.engine import RegionEngine,parse_region_query
from scenemind_mvp.hybrid.evidence import Evidence
from scenemind_mvp.search import SearchIndex

def fixture_engine(tmp_path):
    full=SimpleNamespace(manifest={'identity':{'video_sha256':'test-source','config':{'sampling_fps':1.}}},search_index=SearchIndex(np.array([[.866,.5],[.9165,.4]]),np.array([0.,1.]),backend='numpy'),preview=lambda i:tmp_path/f'full{i}.jpg')
    rows=[{'region_id':0,'timestamp':0.,'class_id':0,'track_id':'c0-t1','source_area_fraction':.2,'kind':'detection'}, {'region_id':1,'timestamp':1.,'class_id':0,'track_id':'c0-t2','source_area_fraction':.001,'kind':'detection'},{'region_id':2,'timestamp':1.,'class_id':24,'track_id':None,'source_area_fraction':.0005,'kind':'detection'}]
    regions=RegionIndex(tmp_path,{'identity':{'source_sha256':'test-source'},'dimension':2},full,rows,np.array([[.995,.1],[.1,.995],[.1,.995]],dtype=np.float32))
    return RegionEngine(regions,Evidence.empty('test-source'),RegionConfig())

def test_background_small_region_can_reverse_foreground_visual_bias(tmp_path):
    engine=fixture_engine(tmp_path);v=np.array([[0.,1.]])
    full=engine.search('a small person in the background',v,'clip_only',min_gap=0)
    region=engine.search('a small person in the background',v,'region',min_gap=0)
    assert full['results'][0]['timestamp']==0 and region['results'][0]['timestamp']==1
    assert region['results'][0]['regions'][0]['source_area_fraction']<.01
    assert all(r['event_claim'] is None for r in region['results'])

def test_small_bag_is_not_discarded_by_area_or_person_overlap(tmp_path):
    result=fixture_engine(tmp_path).search('a backpack',np.array([[0.,1.]]),'region',min_gap=0)
    assert any(r['class_id']==24 for row in result['results'] for r in row['regions'])

@pytest.mark.parametrize('action',['walking','running','jogging'])
def test_single_frame_never_confirms_gait(tmp_path,action):
    engine=fixture_engine(tmp_path);text=f'a person {action}'
    assert engine.search(text,np.array([[0.,1.]]),strict_actions=True)['results']==[]
    result=engine.search(text,np.array([[0.,1.]]),strict_actions=False)
    assert result['results'] and all(r['event_claim'] is None and r['state']=='unknown_action_visual_candidate' for r in result['results'])
    assert 'action_not_modelled' in ' '.join(parse_region_query(text).unsupported_constraints)

def test_near_people_do_not_establish_carrying_or_ownership(tmp_path):
    engine=fixture_engine(tmp_path)
    for text in ('two people near a bag','a person carrying a black bag','a person picking up a bag'):
        result=engine.search(text,np.array([[0.,1.]]),strict_actions=False)
        assert all(r['event_claim'] is None for r in result['results'])

def test_padded_native_crop_preserves_boundaries_and_aspect():
    assert padded_box([0,0,10,20],100,80,.2,48)==[0,0,29,34]
    view=square_view(Image.new('RGB',(20,80),'white'))
    assert view.size==(80,80) and view.getpixel((0,0))==(127,127,127)
    with pytest.raises(ValueError):padded_box([0,0,101,20],100,80)

def test_multiscale_dedup_keeps_distinct_classes_and_small_subjects():
    def p(box,c,size,confidence):return {'bbox_xyxy':box,'class_id':c,'detector_sizes':[size],'confidence':confidence}
    merged=merge_proposals([p([0,0,90,90],0,640,.9),p([1,1,89,89],0,960,.8),p([40,40,45,48],24,960,.2),p([95,95,99,99],0,960,.15)])
    assert len(merged)==3 and merged[0]['detector_sizes']==[640,960]

def test_ambiguous_track_references_remain_unknown():
    prop={'class_id':0,'bbox_xyxy':[0,0,10,10]}
    observations=[{**prop,'track_id':f'c0-t{i}'} for i in (1,2)]
    identifier,candidates=track_reference(prop,observations)
    assert identifier is None and candidates==['c0-t1','c0-t2']
    assert track_reference(prop,[observations[0]])[0]=='c0-t1'

def test_context_views_do_not_claim_object_categories():
    boxes=context_boxes(960,540);assert len(boxes)==4
    assert all(0<=b[0]<b[2]<=960 and 0<=b[1]<b[3]<=540 for b in boxes)

def test_invalid_inputs_and_empty_region_fallback(tmp_path):
    engine=fixture_engine(tmp_path)
    with pytest.raises(ValueError):engine.search('',np.array([[0.,1.]]))
    with pytest.raises(ValueError):engine.search('person',np.array([[0.,1.,0.]]))
    with pytest.raises(ValueError):RegionConfig(sampling_fps=0)
    engine.regions.records=[];engine.regions.vectors=np.empty((0,2))
    result=engine.search('person',np.array([[0.,1.]]),'region')
    assert result['results'] and not result['results'][0]['regions']

def test_real_region_indices_are_normalized_and_timestamped():
    path=Path('benchmarks/regions/comparison.json')
    if not path.exists():pytest.skip('Run real benchmark first')
    report=json.loads(path.read_text())
    assert len(report['experiments'])==5
    for experiment in report['experiments']:
        index=load_regions(Path(experiment['region_directory']))
        assert index.vectors.shape==(len(index.records),512)
        assert np.allclose(np.linalg.norm(index.vectors,axis=1),1,atol=1e-4)
        for row in index.records:
            assert index.preview(row['region_id']).is_file()
            assert row['timestamp'] in index.full.search_index.timestamps
            assert row['track_id'] is None or row['track_reference_status']=='source_frame_unique_iou_match'
        assert not any(row.get('event_claim') for case in experiment['cases'] for method in case['methods'].values() for row in method['result']['results'])


def test_actual_center_crop_can_remove_edge_subjects():
    from scenemind_mvp.regions.crops import center_crop_source_fov
    box=center_crop_source_fov((960,540),(398,224),(224,224))
    assert 209<box[0]<211 and 749<box[2]<751
    assert box[2]<792 # reviewed side bag is outside the actual full-frame CLIP field of view
    assert center_crop_source_fov((80,80),(224,224),(224,224))==[0.,0.,80.,80.]
