"""Evaluate a frozen real-video case set independently from the temporal engine."""
from pathlib import Path
import argparse,json,hashlib
from scenemind_mvp.temporal.evaluation import evaluate_cases
parser=argparse.ArgumentParser();parser.add_argument('timeline',type=Path)
parser.add_argument('--benchmark',type=Path,default=Path('benchmarks/events-real-v1.json'))
parser.add_argument('--output',type=Path,default=Path('docs/temporal-benchmark-result.json'))
args=parser.parse_args();original=args.benchmark.read_bytes()
report=evaluate_cases(json.loads(args.timeline.read_text()),json.loads(original))
assert args.benchmark.read_bytes()==original
report['benchmark_sha256']=hashlib.sha256(original).hexdigest()
args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
