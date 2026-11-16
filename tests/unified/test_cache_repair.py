"""Real ffmpeg/process races on explicit fixture video, not model accuracy."""
from pathlib import Path
import json,hashlib,multiprocessing
import cv2,numpy as np
from scenemind_mvp.pipeline import digest
from scenemind_mvp.unified.render import render,DEFAULTS
from scenemind_mvp.unified.core import make_plan
from scenemind_mvp.unified.cache import publish

def fixture(tmp_path):
    path=tmp_path/'source.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120));assert writer.isOpened()
    for _ in range(20):writer.write(np.zeros((120,160,3),dtype=np.uint8))
    writer.release();frames=[{'frame_index':i,'timestamp':i/10,'detections':[{'class_id':0,'label':'person','track_id':'c0-t1','bbox_xyxy':[20,10,50,100]}]} for i in range(20)]
    return {'source':str(path),'source_sha256':digest(path),'metadata':{'fps':10.,'width':160,'height':120,'frame_count':20,'duration':2.},'identity':{'source':digest(path),'fixture':True},'actions_sha256':'fixture','frames':frames,'actions':{'intervals':[]}}

def worker(analysis,root,barrier,queue):
    try:
        barrier.wait(20);result=render(analysis,make_plan('Show Person 1',analysis),Path(root),{'view':'Search Matches'})
        queue.put({'hit':result['cache_hit'],'directory':result['directory'],'sha':digest(Path(result['video']))})
    except Exception as error:queue.put({'error':repr(error)})

def test_cold_render_concurrent_processes_publish_once(tmp_path):
    # AppTest changes __main__.__file__ to app.py. Windows multiprocessing.spawn
    # would execute that UI in children. Dedicated interpreter entrypoints avoid
    # that test-harness side effect while still racing three real OS processes.
    import subprocess,sys,os,time
    a=fixture(tmp_path);before=digest(Path(a['source']));root=tmp_path/'renders';request=tmp_path/'input.json';request.write_text(json.dumps(a));gate=tmp_path/'start'
    code="""import json,sys,time
from pathlib import Path
from scenemind_mvp.unified.render import render
from scenemind_mvp.unified.core import make_plan
from scenemind_mvp.pipeline import digest
request,root,gate,output,ready=map(Path,sys.argv[1:])
a=json.loads(request.read_text());ready.write_text('ready')
start=time.monotonic()
while not gate.exists():
 if time.monotonic()-start>30:raise TimeoutError('Test start gate')
 time.sleep(.02)
r=render(a,make_plan('Show Person 1',a),root,{'view':'Search Matches'})
output.write_text(json.dumps({'hit':r['cache_hit'],'directory':r['directory'],'sha':digest(Path(r['video']))}))
"""
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[2]/'src')};jobs=[];outputs=[]
    try:
        for i in range(3):
            output=tmp_path/f'result-{i}.json';outputs.append(output);jobs.append(subprocess.Popen([sys.executable,'-c',code,str(request),str(root),str(gate),str(output),str(tmp_path/f'ready-{i}')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE))
        deadline=time.monotonic()+30
        while len(list(tmp_path.glob('ready-*')))<3:
            assert time.monotonic()<deadline,'Workers did not reach start gate';time.sleep(.02)
        gate.write_text('start')
        for job in jobs:
            out,err=job.communicate(timeout=40);assert job.returncode==0,err.decode(errors='replace')
        results=[json.loads(path.read_text()) for path in outputs]
        assert sum(not r['hit'] for r in results)==1 and len({r['directory'] for r in results})==1 and len({r['sha'] for r in results})==1
        assert not list(root.glob('overlay-*')) and digest(Path(a['source']))==before
    finally:
        for job in jobs:
            if job.poll() is None:job.terminate();job.wait(timeout=10)


def test_repeated_search_reuses_tuple_json_identity(tmp_path,monkeypatch):
    a=fixture(tmp_path);plan=make_plan('Show Person 1',a);out=render(a,plan,tmp_path/'renders',{'view':'Search Matches'})
    def forbid(*args,**kw):raise AssertionError('Completed cache must not re-encode')
    monkeypatch.setattr('scenemind_mvp.unified.render.subprocess.run',forbid)
    for _ in range(4):
        again=render(a,make_plan('Show Person 1',a),tmp_path/'renders',{'view':'Search Matches'});assert again['cache_hit'] and again['video']==out['video']

def test_existing_invalid_destination_preserved_and_recovery_reused(tmp_path):
    a=fixture(tmp_path);plan=make_plan('',a);options=DEFAULTS.copy();identity={'revision':'unified-player-v1','analysis':a['identity'],'actions':a['actions_sha256'],'options':options,'plan':plan};key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();root=tmp_path/'renders';invalid=root/key;invalid.mkdir(parents=True);sentinel=invalid/'preserve.txt';sentinel.write_text('not owned by this render')
    first=render(a,plan,root,{});again=render(a,plan,root,{})
    assert first['directory']!=str(invalid) and again['cache_hit'] and first['directory']==again['directory']
    assert sentinel.read_text()=='not owned by this render'

def test_publication_loser_keeps_completed_output(tmp_path):
    staging=tmp_path/'mine';staging.mkdir();(staging/'file').write_text('loser')
    destination=tmp_path/'winner';destination.mkdir();(destination/'file').write_text('valid winner')
    assert publish(staging,destination,lambda p:(p/'file').read_text()=='valid winner') is False
    assert (destination/'file').read_text()=='valid winner' and staging.exists()
