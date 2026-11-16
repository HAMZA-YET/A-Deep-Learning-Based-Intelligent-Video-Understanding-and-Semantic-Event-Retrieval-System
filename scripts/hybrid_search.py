"""Execute real OpenCLIP text encoding and hybrid retrieval against prepared evidence."""
from pathlib import Path
from dataclasses import replace
import argparse,json,time
from scenemind_mvp.config import Config
from scenemind_mvp.model import Encoder
from scenemind_mvp.hybrid.schema import HybridConfig
from scenemind_mvp.hybrid.evidence import load_bundle
from scenemind_mvp.hybrid.engine import HybridEngine
parser=argparse.ArgumentParser();parser.add_argument('bundle',type=Path);parser.add_argument('query')
parser.add_argument('--mode',choices=['hybrid','clip_only'],default='hybrid');parser.add_argument('--config',type=Path,default=Path('configs/hybrid.json'))
parser.add_argument('--output',type=Path)
parser.add_argument('--strict-events',action='store_true')
args=parser.parse_args();index,evidence=load_bundle(args.bundle)
encoder=Encoder(Config(**index.manifest['identity']['config']),Path('data/models'))
started=time.perf_counter();vector=encoder.encode_text(args.query)
settings=HybridConfig(**json.loads(args.config.read_text()))
if args.strict_events:settings=replace(settings,strict_events=True)
result=HybridEngine(index,evidence,settings).search(args.query,vector,args.mode)
result['query_seconds_excluding_model_load']=time.perf_counter()-started
if args.output:args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
