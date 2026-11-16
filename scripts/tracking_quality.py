"""Diagnostics for an existing run; no detector rerun or annotation change."""
from pathlib import Path
import argparse,json
from scenemind_mvp.association.diagnostics import diagnose
parser=argparse.ArgumentParser()
parser.add_argument('detection_directory',type=Path)
parser.add_argument('--output',type=Path,default=Path('docs/tracking-quality.json'))
args=parser.parse_args()
manifest=json.loads((args.detection_directory/'manifest.json').read_text())
frames=json.loads((args.detection_directory/'records.json').read_text())['frames']
report=diagnose(frames,manifest['metadata']['fps'],manifest['identity']['config']['every_n_frames'])
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='tracks'},indent=2))
