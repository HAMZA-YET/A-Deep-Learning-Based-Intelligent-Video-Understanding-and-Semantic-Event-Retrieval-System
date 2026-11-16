"""Localized pixel-cue fixtures: no shirt/action accuracy assertion."""
import numpy as np
from scenemind_mvp.unified.attributes import pixel_fractions,cue_intervals
from scenemind_mvp.unified.core import make_plan,matched_indices
from test_unified_core import scene

def test_bag_mask_does_not_turn_dark_object_into_black_shirt():
    image=np.zeros((100,100,3),dtype=np.uint8);image[:,:40]=[0,180,0];visible=np.zeros((100,100),dtype=bool);visible[:,:40]=True
    result=pixel_fractions(image,visible);assert result['green']==1 and result['black']==0

def test_two_color_crops_required_and_unknown_gaps_not_bridged():
    a={'track_id':'c0-t1','timestamp':1.,'candidate_colors':['green'],'reasons':[]};assert not cue_intervals([a])
    assert cue_intervals([a,{**a,'timestamp':2.}])[0]['verified'] is False
    assert not cue_intervals([a,{**a,'timestamp':3.}])

def test_green_query_intersects_all_constraints_and_all_people():
    a=scene();a['attributes']={'intervals':[{'track_id':'c0-t2','color':'green','start':0.,'end':4.}]}
    p=make_plan('a person walking in green shirt',a)
    assert p['garment']=='shirt' and p['colors']==['green'] and p['action']=='walking'
    assert matched_indices(a['frames'][0],p,a)==[1]
    assert {e['person_track_id'] for e in p['intervals']}=={'c0-t2'}

def test_missing_clothing_cues_never_show_unfiltered_action_cards():
    a=scene();a['attributes']={'intervals':[]};p=make_plan('a person walking in green shirt',a)
    assert p['reasons'] and p['intervals']==[] and matched_indices(a['frames'][0],p,a)==[]
