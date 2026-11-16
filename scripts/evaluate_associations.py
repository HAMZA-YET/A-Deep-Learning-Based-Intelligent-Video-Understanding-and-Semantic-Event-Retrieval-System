"""Report selected visual-review cases without asserting carrying accuracy."""
from pathlib import Path
import argparse,json,hashlib
parser=argparse.ArgumentParser()
parser.add_argument('association_directory',type=Path)
parser.add_argument('--review',type=Path,default=Path('docs/association-visual-review.json'))
parser.add_argument('--output',type=Path,default=Path('docs/association-review-results.json'))
args=parser.parse_args()
review_bytes=args.review.read_bytes();review=json.loads(review_bytes)
manifest=json.loads((args.association_directory/'manifest.json').read_text())
assert manifest['identity']['video_sha256']==review['source_sha256']
data=json.loads((args.association_directory/'associations.json').read_text())
frames={r['frame_index']:r for r in data['frames']}
relations={}
for event in data['events']:
    for evidence in event['supporting_observations']:
        relations[(evidence['frame_index'],evidence['person_detection_index'],evidence['object_detection_index'])]=event['relationship']
spatial=[]
for case in review['spatial_reference_cases']:
    frame=frames[case['frame_index']];reference=case['reference_person_detection_index']
    indexes={c['person_detection_index'] for c in frame['candidates']}
    extras=[c for c in frame['candidates'] if c['person_detection_index']!=reference]
    spatial.append({'frame_index':case['frame_index'],'reference_candidate_retained':reference in indexes,
        'extra_candidate_person_indexes':sorted({c['person_detection_index'] for c in extras if c['person_detection_index'] is not None}),
        'extra_candidate_relationships':sorted({relations[(c['frame_index'],c['person_detection_index'],c['object_detection_index'])] for c in extras})})
missing=[{'frame_index':case['frame_index'],'visually_reviewed_bag_present':True,
          'association_candidates':len(frames[case['frame_index']]['candidates'])} for case in review['visible_bag_checks']]
report={'scope':review['purpose'],'spatial_review':spatial,'visible_bag_checks':missing,
    'reference_retained_cases':sum(c['reference_candidate_retained'] for c in spatial),
    'reviewed_spatial_cases':len(spatial),'extra_candidate_people_across_reviewed_frames':sum(len(c['extra_candidate_person_indexes']) for c in spatial),
    'missing_candidate_cases':sum(c['association_candidates']==0 for c in missing),
    'carrying_precision':None,'identity_switches':None,
    'limitations':[review['forbidden_interpretation'],'Extra spatial candidates are not asserted associations; unknown is intentional.'],
    'review_sha256':hashlib.sha256(review_bytes).hexdigest()}
assert args.review.read_bytes()==review_bytes
args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
