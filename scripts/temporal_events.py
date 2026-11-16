"""Build an event timeline or search its structured records; no language-model fabrication."""
from pathlib import Path
import argparse,json
from scenemind_mvp.temporal.core import TemporalConfig,search_events
from scenemind_mvp.temporal.pipeline import process_timeline
parser=argparse.ArgumentParser()
parser.add_argument('association_directory',type=Path)
parser.add_argument('--config',type=Path,default=Path('configs/temporal.json'))
parser.add_argument('--output',type=Path,default=Path('data/timelines'))
parser.add_argument('--action');parser.add_argument('--object',dest='object_category');parser.add_argument('--timestamp',type=float)
args=parser.parse_args()
result=process_timeline(args.association_directory,args.output,TemporalConfig(**json.loads(args.config.read_text())))
if args.action or args.object_category or args.timestamp is not None:
    timeline=json.loads((Path(result['directory'])/'timeline.json').read_text())
    result['search_results']=search_events(timeline,args.action,args.object_category,args.timestamp)
print(json.dumps(result,indent=2))
