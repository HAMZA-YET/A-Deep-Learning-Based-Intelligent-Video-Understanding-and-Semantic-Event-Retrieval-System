"""Six requested queries on the full real-video frame pool; no accuracy labels assumed."""
from pathlib import Path
from dataclasses import replace
import argparse,json,time
from scenemind_mvp.config import Config
from scenemind_mvp.model import Encoder
from scenemind_mvp.hybrid.schema import HybridConfig
from scenemind_mvp.hybrid.evidence import load_bundle
from scenemind_mvp.hybrid.engine import HybridEngine
parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path)
parser.add_argument('--output',type=Path,default=Path('docs/hybrid-examples.json'))
args=parser.parse_args();bundle=args.bundle or Path(json.loads(Path('docs/hybrid-real-preparation.json').read_text())['bundle'])
index,evidence=load_bundle(bundle);settings=replace(HybridConfig(**json.loads(Path('configs/hybrid.json').read_text())),min_gap_seconds=1.)
encoder=Encoder(Config(**index.manifest['identity']['config']),Path('data/models'));engine=HybridEngine(index,evidence,settings)
queries=['Show a person walking.','Find someone carrying a black bag.','Show two people standing together.',
         'Find a person picking up a bag.','Show someone leaving with a backpack.','Find all people near a bag.']
responses=[]
for query in queries:
    started=time.perf_counter();vector=encoder.encode_text(query);response=engine.search(query,vector)
    response['query_seconds_excluding_model_load']=time.perf_counter()-started;responses.append(response)
    print(query,[(row['timestamp'],round(row['clip_cosine'],4),round(row['rank_score'],4),row['state']) for row in response['results']])
args.output.write_text(json.dumps({'bundle':str(bundle.resolve()),'source':evidence.provenance['source'],
    'scope':'Full semantic frame pool; examples only, unverified labels are not scored.','responses':responses},indent=2),encoding='utf-8')
