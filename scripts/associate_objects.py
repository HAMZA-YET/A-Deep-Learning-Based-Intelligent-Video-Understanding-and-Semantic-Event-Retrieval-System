"""Generate cautious person-object evidence from an existing detection/tracking cache."""
from pathlib import Path
import argparse,json
from scenemind_mvp.association.core import AssociationConfig
from scenemind_mvp.association.pipeline import process_associations
parser=argparse.ArgumentParser()
parser.add_argument('detection_directory',type=Path)
parser.add_argument('--config',type=Path,default=Path('configs/association.json'))
parser.add_argument('--output',type=Path,default=Path('data/associations'))
args=parser.parse_args()
settings=AssociationConfig(**json.loads(args.config.read_text()))
print(json.dumps(process_associations(args.detection_directory,args.output,settings),indent=2))
