"""Manual annotation helpers. No detector, tracker or encoder is imported."""
from copy import deepcopy
import math
from scenemind_mvp.detection.schema import Observation

def add_box(review: dict, frame_index: int, fps: float, width: int, height: int, class_id: int, coordinates: list[float], reference: str='') -> dict:
    if not 0<=frame_index/fps<review['duration']:raise ValueError('Frame outside video')
    box=Observation(class_id,'manual',1.,tuple(coordinates)).to_dict()
    if coordinates[2]>width or coordinates[3]>height:raise ValueError('Box outside source dimensions')
    out=deepcopy(review);out['review_status']='pending';out['reviewer']=None;out['reviewed_utc']=None;out['independent_of_model_predictions']=None
    rows=out.setdefault('detection_frames',[]);row=next((r for r in rows if r['frame_index']==frame_index),None)
    if row is None:row={'frame_index':frame_index,'timestamp':frame_index/fps,'reviewed':False,'exhaustive_classes':[],'boxes':[]};rows.append(row)
    row['boxes'].append({'class_id':class_id,'bbox_xyxy':list(box['bbox_xyxy']),'person_reference':(reference.strip() or None) if class_id==0 else None,'object_reference':(reference.strip() or None) if class_id!=0 else None})
    return out

def add_interval(review: dict,kind: str,action: str,verdict: str,start: float,end: float,person: str='',object_reference: str='',object_category: str='backpack',notes: str='') -> dict:
    if kind not in ('events','relationships') or verdict not in ('positive','negative','ambiguous'):raise ValueError('Invalid manual interval type')
    if not all(math.isfinite(x) for x in (start,end)) or not 0<=start<end<=review['duration']:raise ValueError('Invalid interval')
    if kind=='events' and action not in ('approaching','moving_together','stationary_near_object','possible_pickup','possible_carrying','possible_departure'):raise ValueError('Unsupported temporal action')
    if kind=='relationships' and action not in ('nearby','possibly_carrying','unknown'):raise ValueError('Unsupported relationship')
    out=deepcopy(review);out['review_status']='pending';out['reviewer']=None;out['reviewed_utc']=None;out['independent_of_model_predictions']=None
    rows=out.setdefault(kind,[]);rows.append({'event_id':f'manual-{kind}-{len(rows)+1}','action':action,'verdict':verdict,'start_timestamp':start,'end_timestamp':end,'person_references':[person.strip()] if person.strip() else [],'object_reference':object_reference.strip() or None,'object_category':object_category,'notes':notes})
    return out


def label_query(review: dict,query_index: int,verdict: str,start: float=0.,end: float=0.,notes: str='') -> dict:
    if verdict not in ('positive','negative','ambiguous'):raise ValueError('Invalid verdict')
    if verdict=='positive' and (not all(math.isfinite(x) for x in (start,end)) or not 0<=start<end<=review['duration']):raise ValueError('Positive query requires a valid visible interval')
    out=deepcopy(review);row=out['queries'][query_index]
    previous=row.get('relevant_intervals') if row.get('verdict')=='positive' else None
    row.update(verdict=verdict,relevant_intervals=(previous or [])+[[start,end]] if verdict=='positive' else ([] if verdict=='negative' else None),notes=notes)
    out.update(review_status='pending',reviewer=None,reviewed_utc=None,independent_of_model_predictions=None)
    return out
