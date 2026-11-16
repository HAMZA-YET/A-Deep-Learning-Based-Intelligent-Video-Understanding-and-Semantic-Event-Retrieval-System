from types import SimpleNamespace
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from scenemind_mvp.highlighting.targets import localize
from scenemind_mvp.regions.engine import parse_region_query
from scenemind_mvp.hybrid.evidence import Evidence

def fixture(text,records=None):
    records=records or [dict(class_id=0,bbox_xyxy=[5,5,25,70],kind='detection',track_id='c0-t1'),dict(class_id=0,bbox_xyxy=[150,50,170,80],kind='detection',track_id='c0-t2'),dict(class_id=24,bbox_xyxy=[155,60,163,75],kind='detection',track_id=None),dict(class_id=None,bbox_xyxy=[0,0,200,100],kind='context',track_id=None)]
    records=[dict(region_id=i,timestamp=1.,source_frame_index=30,track_scope='sha',**r) for i,r in enumerate(records)]
    index=SimpleNamespace(full=SimpleNamespace(manifest={'metadata':{'width':200,'height':100}}),manifest={'identity':{'source_sha256':'sha'},'bundle':'bundle.json'},directory=Path('region'),records=records,vectors=np.array([[.6,.8] for r in records]),preview=lambda i:Path(str(i)+'.jpg'))
    result={'query':parse_region_query(text).to_dict(),'query_text':text,'video_id':'sha','results':[{'timestamp':1.,'rank_score':.7}]}
    return result,index,Evidence.empty('sha')

def run(text,records=None,attrs=None):
    r,i,e=fixture(text,records);localize(r,i,np.array([[1.,0.]]),e,attrs);return r['results'][0]['localization']

def test_two_people_small_background_not_discarded():
    loc=run('two people walking together');assert len(loc['primary_target_ids'])==2
    assert len(loc['candidates'])==2 and loc['candidates'][1]['bbox_xyxy']==[150,50,170,80]
    assert all(not t['query_fulfilled'] for t in loc['candidates'])

def test_scene_only_cannot_invent_box():
    assert run('a beautiful outdoor scene')['primary_target_ids']==[]

def test_bag_category_uses_small_detected_bag_not_person_or_context():
    loc=run('a backpack');assert len(loc['candidates'])==1 and loc['candidates'][0]['class_id']==24

def test_duplicate_proposals_do_not_become_two_people():
    rows=[dict(class_id=0,bbox_xyxy=[5,5,25,70],kind='detection',track_id='c0-t1')]*2
    assert run('two people walking',rows)['primary_target_ids']==[]

def test_missing_bag_and_relationship_never_invented():
    loc=run('a person carrying a black bag');assert all(t['associated_objects']==[] and not t['query_fulfilled'] for t in loc['candidates'])

def test_color_contradiction_abstains_and_cue_stays_unverified():
    attrs={'white':np.array([[0.,1.]]),'black':np.array([[1.,0.]])}
    white=run('a person wearing a white shirt',attrs=attrs)
    assert white['primary_target_ids'] and white['candidates'][0]['attribute_evidence']['status']=='unverified'
    assert not run('a person wearing a black shirt',attrs=attrs)['primary_target_ids']

def test_no_color_prompts_means_no_automatic_color_selection():
    assert not run('a person wearing a white shirt')['primary_target_ids']

def test_wrong_video_and_unscoped_detections_rejected():
    r,i,e=fixture('a person');e.source_sha256='other'
    with pytest.raises(ValueError):localize(r,i,np.array([[1.,0.]]),e)
    r,i,e=fixture('a person');i.records[0]['track_scope']='other';i.records[1]['track_scope']='other'
    localize(r,i,np.array([[1.,0.]]),e);assert not r['results'][0]['localization']['primary_target_ids']

@pytest.mark.parametrize('action',['walking','running','carrying','picking up'])
def test_localized_region_does_not_verify_action(action):
    assert all(not t['query_fulfilled'] for t in run('a person '+action)['candidates'])


def test_two_people_one_object_keep_both_candidates_uncertain():
    r,i,e=fixture('two people near a bag')
    observation={'frame_index':30,'object_detection_index':7,'object_bbox_xyxy':[155,60,163,75]}
    e.associations=[{'event_id':'a'+str(n),'person_track_id':'c0-t'+str(n),'object_class_id':24,'object_class':'backpack','object_track_id':None,'relationship':'unknown','supporting_observations':[deepcopy(observation)]} for n in (1,2)]
    e.events=[{'action':'unknown','person_track_ids':['c0-t1'],'start_timestamp':0.,'end_timestamp':2.}]
    localize(r,i,np.array([[1.,0.]]),e)
    targets=r['results'][0]['localization']['candidates']
    assert len(targets)==2
    assert all(t['associated_objects'][0]['relationship']=='unverified' for t in targets)
    assert all(t['associated_objects'][0]['competing_people']==['c0-t1','c0-t2'] for t in targets)
    assert targets[0]['temporal_evidence']['status']=='unknown'
    assert targets[0]['temporal_evidence']['records'][0]['action']=='unknown'
    assert all(not t['query_fulfilled'] for t in targets)


def test_same_source_frame_rounding_does_not_drop_detections():
    r,i,e=fixture('a person walking');r['results'][0]['timestamp']+=1e-8
    localize(r,i,np.array([[1.,0.]]),e)
    loc=r['results'][0]['localization'];assert loc['primary_target_ids']
    assert loc['timestamp_mapping']['detection_timestamp']==1.

def test_nearby_other_frame_is_not_borrowed():
    r,i,e=fixture('a person walking');r['results'][0]['timestamp']+=.01
    localize(r,i,np.array([[1.,0.]]),e)
    assert not r['results'][0]['localization']['candidates']

def test_stable_track_candidate_preferred_without_claiming_action():
    r,i,e=fixture('a person walking');i.full.manifest['metadata']['fps']=30
    # First person has a marginally higher crop score but no source track history.
    i.vectors=np.array([[.8,.6],[.6,.8],[.6,.8],[.6,.8]])
    e.frames=[{'frame_index':n,'detections':[{'class_id':0,'track_id':'c0-t2','bbox_xyxy':[150,50,170,80]}]} for n in range(60)]
    localize(r,i,np.array([[1.,0.]]),e)
    loc=r['results'][0]['localization'];selected=next(c for c in loc['candidates'] if c['id'] in loc['primary_target_ids'])
    assert selected['track_id']=='c0-t2' and selected['highlight_continuity']['usable_video_segment']
    assert not selected['query_fulfilled'] and len(loc['candidates'])==2
