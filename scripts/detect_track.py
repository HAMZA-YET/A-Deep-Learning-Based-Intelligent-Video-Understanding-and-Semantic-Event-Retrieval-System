"""Stage 2.1/2.2 CLI: no person-object association or event inference."""
from pathlib import Path
import argparse
import json
import contextlib
import sys
from scenemind_mvp.detection.schema import DetectionConfig
from scenemind_mvp.detection.pipeline import process_video
parser=argparse.ArgumentParser()
parser.add_argument('video',type=Path)
parser.add_argument('--config',type=Path,default=Path('configs/detection.json'))
parser.add_argument('--output',type=Path,default=Path('data/detection'))
parser.add_argument('--no-tracking',action='store_true')
args=parser.parse_args()
settings=json.loads(args.config.read_text())
settings['classes']=tuple(settings['classes'])
if args.no_tracking: settings['tracking']=False
with contextlib.redirect_stdout(sys.stderr):
    result=process_video(args.video,args.output,DetectionConfig(**settings))
print(json.dumps(result,indent=2))
