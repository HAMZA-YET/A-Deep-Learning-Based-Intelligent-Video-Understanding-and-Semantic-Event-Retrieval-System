from copy import deepcopy
import pytest
from PIL import Image
from scenemind_mvp.highlighting.render import project_box,draw_frame,track_segment

def target():return {'source_frame_index':2,'class_id':0,'track_id':'c0-t1','bbox_xyxy':[10,10,30,50],'label':'Person 1'}
def frames(indices):return [dict(frame_index=i,detections=[dict(class_id=0,track_id='c0-t1',bbox_xyxy=[10+i,10,30+i,50])]) for i in indices]

def test_resize_projection_and_padding():
    assert project_box([10,10,30,50],(200,100),(100,50),(0,25))==(5,30,15,50)
    image=draw_frame(Image.new('RGB',(200,100),'white'),[target()],(100,100),True)
    assert image.size==(100,100) and image.getpixel((5,31))!=(255,255,255)

@pytest.mark.parametrize('box',[[-1,1,5,5],[1,1,500,5],[5,1,1,5],[float('nan'),1,5,5]])
def test_invalid_coordinates_rejected(box):
    with pytest.raises(ValueError):project_box(box,(200,100),(100,50))

def test_lost_track_never_resumes_or_interpolates():
    trace,summary=track_segment(frames([0,1,2,4,5]),target())
    assert set(trace)=={0,1,2} and summary['frames']==3

def test_untracked_target_only_anchor():
    t=target();t['track_id']=None
    assert set(track_segment(frames(range(6)),t)[0])=={2}

def test_ambiguous_track_or_large_jump_stops():
    f=frames(range(6));f[3]['detections']*=2
    assert set(track_segment(f,target())[0])=={0,1,2}
    f=frames(range(6));f[3]['detections'][0]['bbox_xyxy']=[150,10,170,50]
    assert 3 not in track_segment(f,target())[0]

def test_sparse_observations_not_propagated_to_unsampled_frames():
    trace,_=track_segment(frames([0,2,4,6]),target(),2)
    assert set(trace)=={0,2,4,6} and 3 not in trace
