from pathlib import Path
import argparse,json,contextlib,sys,time
from dataclasses import replace
from scenemind_mvp.regions.schema import RegionConfig
from scenemind_mvp.regions.pipeline import build_regions,load_regions
from scenemind_mvp.regions.engine import RegionEngine
from scenemind_mvp.hybrid.evidence import load_bundle
from scenemind_mvp.hybrid.schema import HybridConfig
from scenemind_mvp.model import Encoder
from scenemind_mvp.config import Config
p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('--config',type=Path,default=Path('configs/regions.json'));p.add_argument('--query');p.add_argument('--visual-candidates',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args()
d=json.loads(args.config.read_text(encoding='utf-8-sig'));d['detector_sizes']=tuple(d['detector_sizes']);config=RegionConfig(**d)
index,evidence=load_bundle(args.bundle);encoder=Encoder(Config(**index.manifest['identity']['config']),Path('data/models'))
if encoder.device=='cuda':encoder.torch.cuda.reset_peak_memory_stats()
with contextlib.redirect_stdout(sys.stderr):regions,cached=build_regions(args.bundle,Path('data/regions'),config,encoder)
report={'region_directory':str(regions.directory.resolve()),'cache_hit':cached,'manifest':regions.manifest}
if args.query:
    vector=encoder.encode_text(args.query);report['search']=RegionEngine(regions,evidence,config,HybridConfig(**json.loads(Path('configs/hybrid.json').read_text()))).search(args.query,vector,strict_actions=not args.visual_candidates)
report['peak_gpu_allocated_mib']=encoder.torch.cuda.max_memory_allocated()/1024**2 if encoder.device=='cuda' else None
report['peak_gpu_reserved_mib']=encoder.torch.cuda.max_memory_reserved()/1024**2 if encoder.device=='cuda' else None
if args.output:args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
