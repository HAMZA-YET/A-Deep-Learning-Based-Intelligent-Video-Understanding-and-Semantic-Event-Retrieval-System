"""Actual student-video race and repeated-search verification. No new AI features."""
from pathlib import Path
import json,multiprocessing,uuid,time
from scenemind_mvp.unified.pipeline import load_analysis
from scenemind_mvp.unified.attributes import analyze_attributes
from scenemind_mvp.unified.core import make_plan,match_summary
from scenemind_mvp.unified.render import render
from scenemind_mvp.pipeline import digest
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/phase3/repair'

def worker(directory,root,barrier,queue):
    try:
        analysis=load_analysis(Path(directory));plan=make_plan('a person walking',analysis);barrier.wait(30)
        result=render(analysis,plan,Path(root),{'view':'Search Matches'});queue.put({'pid':__import__('os').getpid(),'cache_hit':result['cache_hit'],'directory':result['directory'],'sha256':digest(Path(result['video'])),'seconds':result['request_seconds']})
    except Exception as error:queue.put({'error':repr(error)})

def main():
    report=json.loads((ROOT/'docs/phase3/actual-analysis.json').read_text());row=next(r for r in report['runs'] if 'c242ea' in r['source']);a=load_analysis(Path(row['analysis_directory']));a={**a,'attributes':analyze_attributes(a,ROOT/'data/attributes')};source_before=digest(Path(a['source']))
    stress=ROOT/'data/unified-renders'/('repair-stress-'+uuid.uuid4().hex);ctx=multiprocessing.get_context('spawn');barrier=ctx.Barrier(3);queue=ctx.Queue();jobs=[ctx.Process(target=worker,args=(a['directory'],str(stress),barrier,queue)) for _ in range(3)]
    for j in jobs:j.start()
    concurrent=[queue.get(timeout=120) for _ in jobs]
    for j in jobs:j.join(10);assert j.exitcode==0
    assert all('error' not in r for r in concurrent),concurrent
    assert sum(not r['cache_hit'] for r in concurrent)==1 and len({r['sha256'] for r in concurrent})==1
    assert not list(stress.glob('overlay-*'))
    sequential=[];prior={}
    for round in range(3):
        for query in ['', 'a person walking','a person walking in green shirt','a person wearing a black shirt','']:
            started=time.perf_counter();plan=make_plan(query,a);summary=match_summary(plan,a);result=render(a,plan,ROOT/'data/unified-renders',{'view':'Search Matches' if query else 'Show All'})
            sha=digest(Path(result['video']))
            if query in prior:assert prior[query]==(result['video'],sha) and result['cache_hit']
            prior[query]=(result['video'],sha);sequential.append({'round':round,'query':query,'summary':summary,'render':result,'video_sha256':sha,'seconds':time.perf_counter()-started});print(round,query,summary['status'],summary['track_ids'],result['cache_hit'],flush=True)
    assert digest(Path(a['source']))==source_before
    result={'concurrent_processes':concurrent,'sequential':sequential,'source_unchanged':True,'winerror183':False,'notes':['Three synchronized processes race for one cold real-video cache. Exactly one publishes.','Fifteen requests run in one process without restarting or deleting indexes/tracks.','Color matches are uncertain torso cues, never verified shirt recognition.']}
    (OUT/'actual-repair.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
