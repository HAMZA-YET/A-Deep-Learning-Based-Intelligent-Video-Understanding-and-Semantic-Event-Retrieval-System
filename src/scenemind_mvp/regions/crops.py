"""No super-resolution or inferred identities: crops retain only observed source pixels."""
import math
from PIL import Image,ImageOps
from ..association.core import box_iou

def padded_box(box: list[float],width: int,height: int,padding: float=.2,min_side: int=48) -> list[int]:
    if len(box)!=4 or not all(math.isfinite(v) for v in box) or not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height:raise ValueError('Invalid source box')
    x1,y1,x2,y2=box;cx=(x1+x2)/2;cy=(y1+y2)/2;w=max(min_side,(x2-x1)*(1+2*padding));h=max(min_side,(y2-y1)*(1+2*padding))
    return [max(0,math.floor(cx-w/2)),max(0,math.floor(cy-h/2)),min(width,math.ceil(cx+w/2)),min(height,math.ceil(cy+h/2))]

def square_view(image: Image.Image) -> Image.Image:
    """Letterbox rather than center-crop away a narrow body/bag. No native upscale saved."""
    return ImageOps.pad(image.convert('RGB'),(max(image.size),)*2,color=(127,127,127),method=Image.Resampling.LANCZOS)

def merge_proposals(proposals: list[dict],threshold: float=.6) -> list[dict]:
    """Class-aware dedup, tracking metadata is resolved separately; no cross-class removal."""
    selected=[]
    for proposal in sorted(proposals,key=lambda p:p['confidence'],reverse=True):
        old=next((p for p in selected if p['class_id']==proposal['class_id'] and box_iou(p['bbox_xyxy'],proposal['bbox_xyxy'])>=threshold),None)
        if old is None:selected.append(dict(proposal))
        else:old['detector_sizes']=sorted(set(old['detector_sizes']+proposal['detector_sizes']))
    return selected

def track_reference(proposal: dict,observations: list[dict],threshold: float=.5) -> tuple[str|None,list[str]]:
    matches=[o for o in observations if o['class_id']==proposal['class_id'] and box_iou(proposal['bbox_xyxy'],o['bbox_xyxy'])>=threshold]
    ids=sorted({m['track_id'] for m in matches if m.get('track_id')})
    return (ids[0] if len(ids)==1 and len(matches)==1 else None,ids)

def context_boxes(width: int,height: int) -> list[list[int]]:
    """Four overlapping context views are visual fallback, not object detections."""
    return [[int(x*width),int(y*height),min(width,math.ceil((x+.6)*width)),min(height,math.ceil((y+.6)*height))] for y in (0.,.4) for x in (0.,.4)]


def center_crop_source_fov(source_size: tuple[int,int],resized_size: tuple[int,int],crop_size: tuple[int,int]) -> list[float]:
    """Map deterministic resize/center-crop visibility back to original source pixels."""
    sw,sh=source_size;rw,rh=resized_size;cw,ch=crop_size
    if min(sw,sh,rw,rh,cw,ch)<=0 or cw>rw or ch>rh:raise ValueError('Unsupported crop geometry')
    left=int(round((rw-cw)/2));top=int(round((rh-ch)/2))
    return [left*sw/rw,top*sh/rh,(left+cw)*sw/rw,(top+ch)*sh/rh]
