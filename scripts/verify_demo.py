"""Run the real demo pipeline and score point timestamps against immutable annotations.
No production model substitutes; errors remain explicit in machine-readable output.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import time
import traceback
from scenemind_mvp.config import Config
from scenemind_mvp.model import Encoder
from scenemind_mvp.pipeline import index_video
from scenemind_mvp.video import inspect_video, iter_frames


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,default=Path('data/demo/scenemind_demo.mp4'))
    parser.add_argument('--annotations',type=Path,default=Path('data/demo/scenemind_demo_ground_truth.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/demo-verification-current.json'))
    args=parser.parse_args()
    truth_bytes=args.annotations.read_bytes()
    truth=json.loads(truth_bytes)
    config=replace(Config.load(Path('configs/default.json')),top_k=5,min_similarity=-1.0)
    report={'video':str(args.video.resolve()),'ground_truth_sha256':hashlib.sha256(truth_bytes).hexdigest(),
            'status':'running','index_seconds':None,'model_load_seconds':None,
            'gpu_peak_allocated_mib':None,'gpu_peak_reserved_mib':None,
            'queries':[{'query':row['query'],'status':'not_executed','results':None,
                        'query_seconds':None,'hit_at_1':None,'hit_at_5':None}
                       for row in truth['queries']]}
    encoder=None
    try:
        info=inspect_video(args.video)
        sample_times=[frame.timestamp for frame in iter_frames(args.video,config.sampling_fps)]
        report['video_validation']={'duration':info.duration,'fps':info.fps,'samples':len(sample_times),
                                    'sample_timestamps':sample_times}
        load_start=time.perf_counter()
        encoder=Encoder(config,Path('data/models'))
        report['model_load_seconds']=time.perf_counter()-load_start
        report['device']=encoder.device
        torch=encoder.torch
        if encoder.device=='cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        indexed=index_video(args.video,Path('data/indexes'),config,lambda:encoder)
        if encoder.device=='cuda': torch.cuda.synchronize()
        report['index_seconds']=time.perf_counter()-started
        report['cache_hit']=indexed.cache_hit
        report['search_backend']=indexed.search_index.backend
        for row,annotation in zip(report['queries'],truth['queries']):
            try:
                started=time.perf_counter()
                vector=encoder.encode_text(row['query'])
                if encoder.device=='cuda': torch.cuda.synchronize()
                results=indexed.search_index.search(vector,5,config.min_gap_seconds,-1.0)
                row['query_seconds']=time.perf_counter()-started
                for result in results:
                    result['preview']=str(indexed.preview(result['frame_index']).resolve())
                    result['ground_truth_hit']=any(start<=result['timestamp']<end for start,end in annotation['intervals'])
                row['results']=results
                row['hit_at_1']=bool(results and results[0]['ground_truth_hit'])
                row['hit_at_5']=any(result['ground_truth_hit'] for result in results)
                row['precision_at_5']=sum(result['ground_truth_hit'] for result in results)/len(results) if results else 0
                row['status']='success' if row['hit_at_5'] else 'retrieval_failure'
            except Exception:
                row['status']='execution_error'
                row['error']=traceback.format_exc()
        report['device_after_inference']=encoder.device
        report['notices']=encoder.notices
        if encoder.device=='cuda':
            report['gpu_peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
            report['gpu_peak_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2
        report['status']='completed'
    except Exception:
        report['status']='execution_error'
        report['error']=traceback.format_exc()
        for row in report['queries']:
            row['reason']='Pipeline failed before query execution; see top-level error'
    finally:
        report['ground_truth_unchanged']=args.annotations.read_bytes()==truth_bytes
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='completed' else 1

if __name__=='__main__':
    raise SystemExit(main())
