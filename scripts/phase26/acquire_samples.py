"""Acquire openly licensed originals; create separate deterministic local evaluation copies."""
from pathlib import Path
import urllib.request,subprocess,json,hashlib
from dataclasses import asdict
from scenemind_mvp.detection.pipeline import find_ffmpeg
from scenemind_mvp.video import inspect_video
root=Path('data/phase26');root.mkdir(exist_ok=True)
items=[('fitwalking','Fit_walking.webmhd.webm','2/2e','Msrmesa','CC BY-SA 3.0','https://creativecommons.org/licenses/by-sa/3.0/',40,12),('winery','Buena_Vista_Winery_-_September_2024_-_Sarah_Stierch_02.webm','e/e1','Sarah Stierch (Missvain)','CC0 1.0','https://creativecommons.org/publicdomain/zero/1.0/',0,12)]
for key,name,folder,creator,license,license_url,start,duration in items:
    url=f'https://upload.wikimedia.org/wikipedia/commons/{folder}/{name}'
    source=root/f'{key}_original.webm';target=root/f'{key}.mp4'
    if not source.exists():
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SceneMind research evaluation/0.1'}),timeout=60) as response,source.open('wb') as out:
            while chunk:=response.read(1024*1024):out.write(chunk)
    if not target.exists():subprocess.run([find_ffmpeg(),'-v','error','-ss',str(start),'-i',str(source),'-t',str(duration),'-vf','scale=960:960:force_original_aspect_ratio=decrease:force_divisible_by=2','-r','25','-an','-c:v','libx264','-preset','fast','-crf','23',str(target)],check=True)
    record={'id':key,'path':str(target),'source_page':f'https://commons.wikimedia.org/wiki/File:{name}','download_url':url,'creator':creator,'license':license,'license_url':license_url,'changes':f'Source offset {start}s, first up to {duration}s thereafter, fit960x960, CFR25,H264,no audio. Original preserved.','source_offset_seconds':start,'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'metadata':asdict(inspect_video(target))}
    (root/f'{key}_attribution.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))
