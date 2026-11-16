"""Official AVA weights, bounded parallel ranges, atomic publication."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,math
ROOT=Path(__file__).resolve().parents[2];CACHE=ROOT/'data/models/actions';CACHE.mkdir(parents=True,exist_ok=True)
URL='https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ava/SLOWFAST_8x8_R50_DETECTION.pyth'
def main():
    dest=CACHE/'SLOWFAST_8x8_R50_DETECTION.pyth'
    if dest.exists():print('Official checkpoint already cached');return
    with urllib.request.urlopen(urllib.request.Request(URL,method='HEAD'),timeout=30) as r:size=int(r.headers['Content-Length'])
    count=8;chunk=math.ceil(size/count)
    def get(i):
        start=i*chunk;end=min(size-1,(i+1)*chunk-1);path=CACHE/('ava-range-'+str(i)+'.partial')
        if path.exists() and path.stat().st_size==end-start+1:return path
        request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=180) as r:
            if r.status!=206:raise ValueError('Server did not honor byte range')
            with path.open('wb') as f:
                while block:=r.read(1024*1024):f.write(block)
        if path.stat().st_size!=end-start+1:raise ValueError('Incomplete model chunk')
        print('Completed official weight range',i,flush=True);return path
    with ThreadPoolExecutor(max_workers=count) as pool:paths=list(pool.map(get,range(count)))
    tmp=CACHE/'ava-complete.partial'
    with tmp.open('wb') as f:
        for path in paths:
            with path.open('rb') as src:
                while block:=src.read(1024*1024):f.write(block)
    if tmp.stat().st_size!=size:raise ValueError('Checkpoint size mismatch')
    tmp.replace(dest)
    for path in paths:path.unlink()
    urllib.request.urlretrieve('https://research.google.com/ava/download/ava_action_list_v2.2.pbtxt',CACHE/'ava_action_list_v2.2.pbtxt')
    print('Official model cached:',dest,flush=True)
if __name__=='__main__':main()
