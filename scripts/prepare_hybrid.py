"""Prepare source-bound offline evidence for unified retrieval."""
from pathlib import Path
import argparse,json,contextlib,sys
from scenemind_mvp.config import Config
from scenemind_mvp.detection.schema import DetectionConfig
from scenemind_mvp.hybrid.prepare import prepare_bundle
from scenemind_mvp.association.core import AssociationConfig
from scenemind_mvp.temporal.core import TemporalConfig
parser=argparse.ArgumentParser();parser.add_argument('video',type=Path)
parser.add_argument('--data',type=Path,default=Path('data'))
parser.add_argument('--detection-config',type=Path,default=Path('configs/detection.json'))
parser.add_argument('--semantic-config',type=Path,default=Path('configs/default.json'))
parser.add_argument('--association-config',type=Path,default=Path('configs/association.json'))
parser.add_argument('--temporal-config',type=Path,default=Path('configs/temporal.json'))
args=parser.parse_args();d=json.loads(args.detection_config.read_text());d['classes']=tuple(d['classes'])
with contextlib.redirect_stdout(sys.stderr):result=prepare_bundle(args.video,args.data,Config.load(args.semantic_config),DetectionConfig(**d),
    association_config=AssociationConfig(**json.loads(args.association_config.read_text())),temporal_config=TemporalConfig(**json.loads(args.temporal_config.read_text())))
print(json.dumps(result,indent=2))
