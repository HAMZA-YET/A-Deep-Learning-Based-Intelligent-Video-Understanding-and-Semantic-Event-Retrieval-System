"""Describe evidence without promoting detector/CLIP hypotheses to verified facts."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ResultLabel:
    status: str
    kind: str
    description: str
    color: str

def label_result(row: dict,query: dict) -> ResultLabel:
    warnings=query.get('unsupported_constraints',[])
    if any('identity' in w for w in warnings):return ResultLabel('Unsupported','Visual similarity',"This search cannot establish a person's identity.",'gray')
    if warnings:return ResultLabel('Uncertain','Visual similarity','Requested action or attributes are unverified.','orange')
    claim=row.get('event_claim')
    if claim and claim.get('status')=='observed':return ResultLabel('Candidate','Temporal evidence','Motion evidence spans several frames; physical interaction is unverified.','blue')
    if claim:return ResultLabel('Candidate','Possible event','A possible event has multi-frame support, with remaining uncertainty.','blue')
    evidence=row.get('evidence',{})
    if query.get('action')=='nearby':
        supported=[a for a in evidence.get('associations',[]) if a.get('relationship') in ('nearby','possibly_carrying')]
        if supported:return ResultLabel('Uncertain','Spatial evidence','Possible image-plane proximity; contact and ownership are unverified.','orange')
        return ResultLabel('Uncertain','Visual similarity','The person-object relationship is unverified.','orange')
    if query.get('action'):return ResultLabel('Uncertain','Visual similarity','Requested action or attributes are unverified.','orange')
    objects=evidence.get('objects',[]) or [r for r in row.get('regions',[]) if r.get('class_id') not in (0,None)]
    if objects and query.get('object_classes'):return ResultLabel('Candidate','Detected object','Detector suggestion; the object label may be incorrect.','blue')
    return ResultLabel('Candidate','Visual similarity','Similar appearance; no action or relationship is confirmed.','gray')

def timestamp(seconds: float) -> str:
    hours=int(seconds//3600);minutes=int(seconds%3600//60);value=seconds%60
    sec=f'{value:05.2f}'.rstrip('0').rstrip('.')
    if len(sec)<2:sec=sec.zfill(2)
    return (f'{hours:02d}:' if hours else '')+f'{minutes:02d}:'+sec

def friendly_error(error: Exception) -> str:
    message=str(error)
    if 'memory' in message.lower():return 'The GPU ran out of memory. Try a smaller batch or select CPU in Advanced Details.'
    if 'nonempty' in message.lower() or not message:return 'Enter a description of the moment you want to find.'
    if 'model' in message.lower() or 'checkpoint' in message.lower():return 'The model could not be loaded. Check the model settings and local cache in Advanced Details.'
    if 'changed' in message.lower() or 'mismatch' in message.lower() or 'corrupt' in message.lower():return 'The saved index no longer matches this video or its evidence. Prepare it again in Advanced Details.'
    if isinstance(error,(OSError,ValueError)):return 'This video or index could not be opened. Check the file and try preparing it again.'
    return 'The search could not finish. Try again; technical details are available below.'
