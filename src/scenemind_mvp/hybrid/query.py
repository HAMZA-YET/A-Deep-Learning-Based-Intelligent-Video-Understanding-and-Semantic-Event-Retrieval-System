"""Explicit, limited query parsing. Unmodelled constraints are never silently confirmed."""
from dataclasses import asdict,dataclass
import re

@dataclass(frozen=True)
class ParsedQuery:
    text: str
    action: str | None
    object_classes: tuple[int,...]
    colors: tuple[str,...]
    minimum_people: int
    all_people: bool
    person_track_ids: tuple[str,...]
    start: float | None
    end: float | None
    at: float | None
    start_inclusive: bool
    end_inclusive: bool
    unsupported_constraints: tuple[str,...]
    def to_dict(self): return asdict(self)

TIME=r'(?:\d{1,2}:\d{2}(?::\d{2})?|\d+(?:\.\d+)?)'
def seconds(value: str) -> float:
    parts=value.split(':')
    if len(parts)>1 and any(float(p)>=60 for p in parts[1:]):raise ValueError('Invalid clock timestamp')
    result=0.
    for part in parts:result=result*60+float(part)
    return result

def parse_query(text: str) -> ParsedQuery:
    if not text.strip():raise ValueError('Enter a nonempty query')
    q=text.lower();warnings=[];action=None
    if re.search(r'\b(?:at|after|before|between|from)\s+-',q):raise ValueError('Timestamps must be nonnegative')
    patterns=[('leaving_with_object',r'\b(?:leaving|leave|left|departing)\s+with\b'),
        ('possible_pickup',r'\b(?:pick(?:ing)?\s+up|pickup|picks\s+up)\b'),
        ('possible_carrying',r'\b(?:carry|carrying|carries)\b'),
        ('stationary_near_object',r'\b(?:stationary|staying still)\b'),
        ('moving_together',r'\b(?:moving together|move together|co-moving)\b'),
        ('approaching',r'\b(?:approach|approaching)\b'),
        ('possible_departure',r'\b(?:leaving|leave|left|departing|departure)\b'),
        ('nearby',r'\b(?:near|nearby|beside|next to)\b'),
        ('walking',r'\b(?:walk|walking|walks)\b'),
        ('standing',r'\b(?:stand|standing|stands)\b'),
        ('sitting',r'\b(?:sit|sitting|sits|seated)\b')]
    for name,pattern in patterns:
        if re.search(pattern,q):action=name;break
    if action in ('walking','standing','sitting','leaving_with_object'):warnings.append('action_not_modelled:'+action)
    if re.search(r'\b(?:not|without|never|no)\b',q):warnings.append('negation_not_supported')
    families=[]
    for pattern,classes in [(r'\bbackpacks?\b',(24,)),(r'\bhandbags?\b',(26,)),(r'\bsuitcases?\b',(28,)),(r'\bchairs?\b',(56,))]:
        if re.search(pattern,q):families.append(classes)
    if re.search(r'\bbags?\b',q) and not families:families.append((24,26,28))
    objects=tuple(sorted({c for family in families for c in family}))
    if len(families)>1:warnings.append('multiple_object_constraints_not_fully_supported')
    colors=tuple(re.findall(r'\b(?:black|white|red|blue|green|yellow|brown|gray|grey|orange|purple)\b',q))
    if colors:warnings.append('color_not_structurally_verified')
    if re.search(r'\b(?:foreground|background)\b',q):warnings.append('depth_or_foreground_not_verified')
    person=bool(re.search(r'\b(?:person|people|someone|somebody|persons|man|woman)\b',q))
    count=2 if re.search(r'\b(?:two|both|2)\s+(?:people|persons|men|women)\b',q) else int(person)
    all_people=bool(re.search(r'\ball\s+(?:people|persons|men|women)\b',q))
    ids=tuple(dict.fromkeys(re.findall(r'\bc0-t\d+\b',q)))
    if re.search(r'\b(?:identify|face recognition|named person|who is)\b',q):warnings.append('identity_inference_not_supported')
    start=end=at=None;start_inclusive=end_inclusive=True
    range_match=re.search(r'\b(?:between|from)\s+('+TIME+r')\s*(?:s|seconds?|minutes?)?\s+(?:and|to|through)\s+('+TIME+r')',q)
    if range_match:start,end=map(seconds,range_match.groups())
    else:
        after=re.search(r'\bafter\s+('+TIME+r')',q);before=re.search(r'\bbefore\s+('+TIME+r')',q)
        point=re.search(r'\bat\s+('+TIME+r')',q)
        if after:start=seconds(after.group(1));start_inclusive=False
        if before:end=seconds(before.group(1));end_inclusive=False
        if point:at=seconds(point.group(1))
    if start is not None and end is not None and start>end:raise ValueError('Time range is reversed')
    if re.search(r'\b(?:after|before)\s+(?:this|that|the event)\b',q):warnings.append('relative_event_reference_not_supported')
    if re.search(r'\b(?:minutes?|hours?)\b',q):raise ValueError('Use mm:ss or seconds for time constraints')
    return ParsedQuery(text.strip(),action,objects,colors,count,all_people,ids,start,end,at,start_inclusive,end_inclusive,tuple(warnings))

def accepts_timestamp(parsed: ParsedQuery, timestamp: float, sample_tolerance: float = 0.) -> bool:
    if parsed.start is not None and (timestamp<parsed.start-1e-6 or (not parsed.start_inclusive and timestamp<=parsed.start+1e-6)):return False
    if parsed.end is not None and (timestamp>parsed.end+1e-6 or (not parsed.end_inclusive and timestamp>=parsed.end-1e-6)):return False
    return parsed.at is None or abs(timestamp-parsed.at)<=sample_tolerance+1e-6
