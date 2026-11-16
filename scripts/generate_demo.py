"""Generate a deterministic cartoon demo, not photorealistic benchmark footage.
Never overwrites an existing video or annotation file.
"""
from pathlib import Path
import argparse
import json
import math
import shutil
import subprocess
import tempfile
import cv2
import numpy as np

FPS = 15
WIDTH, HEIGHT = 640, 360
QUERIES = [
    ('A person walking', [[0, 24]]),
    ('Someone carrying a black bag', [[12, 24], [56, 60]]),
    ('Two people standing together', [[24, 36]]),
    ('A person sitting on a chair', [[36, 48]]),
    ('Someone picking up a bag', [[49, 56]]),
]

def person(image, x, y=288, phase=0.0, walking=False, bag=False, sitting=False, bend=0.0, shirt=(175,85,40)):
    """Filled stylized human with head, torso, articulated limbs and optional bag."""
    skin=(125,177,218)
    trousers=(75,65,55)
    shoulder=(int(x+bend*35),int(y-116+bend*34))
    hip=(int(x),int(y-63))
    head=(shoulder[0]+int(bend*13),shoulder[1]-28)
    cv2.circle(image,head,17,skin,-1,cv2.LINE_AA)
    cv2.ellipse(image,(head[0],head[1]-8),(17,10),0,180,360,(42,38,34),-1,cv2.LINE_AA)
    cv2.line(image,shoulder,hip,shirt,33,cv2.LINE_AA)
    if sitting:
        knee=(int(x+45),int(y-63))
        cv2.line(image,hip,knee,trousers,17,cv2.LINE_AA)
        cv2.line(image,knee,(knee[0]+2,int(y)),trousers,15,cv2.LINE_AA)
        cv2.line(image,(int(x+2),int(y-56)),(int(x+36),int(y-52)),trousers,12,cv2.LINE_AA)
        cv2.line(image,(int(x+36),int(y-52)),(int(x+40),int(y)),trousers,12,cv2.LINE_AA)
    else:
        stride=22*math.sin(phase) if walking else 7
        for direction in (-1,1):
            foot=(int(x+direction*stride),int(y))
            knee=(int(x+direction*stride*.4),int(y-32))
            cv2.line(image,hip,knee,trousers,15,cv2.LINE_AA)
            cv2.line(image,knee,foot,trousers,13,cv2.LINE_AA)
            cv2.line(image,foot,(foot[0]+13,foot[1]),(30,30,30),10,cv2.LINE_AA)
    if sitting:
        hand=(int(x+38),int(y-75))
    elif bend:
        hand=(int(x+57),int(y-65+bend*40))
    else:
        hand=(int(x+22),int(y-57))
    elbow=((shoulder[0]+hand[0])//2+6,(shoulder[1]+hand[1])//2)
    cv2.line(image,shoulder,elbow,shirt,12,cv2.LINE_AA)
    cv2.line(image,elbow,hand,skin,10,cv2.LINE_AA)
    cv2.circle(image,hand,6,skin,-1,cv2.LINE_AA)
    if bag:
        draw_bag(image,hand[0],hand[1]+28)

def draw_bag(image,x,y):
    cv2.rectangle(image,(int(x-17),int(y-12)),(int(x+17),int(y+12)),(15,15,15),-1)
    cv2.ellipse(image,(int(x),int(y-12)),(9,9),0,180,360,(15,15,15),3)

def render(t):
    image=np.full((HEIGHT,WIDTH,3),(225,230,233),np.uint8)
    cv2.rectangle(image,(0,289),(WIDTH,HEIGHT),(165,177,182),-1)
    cv2.line(image,(0,289),(WIDTH,289),(110,120,130),2)
    # No scene captions or query text: avoid creating OCR label shortcuts.
    if t<12:
        person(image,130+(t/12)*370,phase=t*5,walking=True)
    elif t<24:
        person(image,125+((t-12)/12)*370,phase=t*5,walking=True,bag=True)
    elif t<36:
        person(image,260,shirt=(65,105,180))
        person(image,365,shirt=(120,155,55))
    elif t<48:
        cv2.line(image,(274,212),(274,292),(70,78,95),9)
        cv2.line(image,(270,226),(330,226),(70,78,95),12)
        cv2.line(image,(326,226),(326,292),(70,78,95),8)
        cv2.line(image,(270,158),(270,225),(70,78,95),10)
        person(image,290,sitting=True)
    else:
        local=t-48
        bend=min(1,max(0,(local-1)/3)) if local<4 else max(0,1-(local-4)/4)
        holding=local>=5
        person(image,290,bend=bend,bag=holding)
        if not holding:
            draw_bag(image,347,279)
    return image

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('data/demo'))
    parser.add_argument('--ffmpeg',default=shutil.which('ffmpeg'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    video=args.output/'scenemind_demo.mp4'
    truth=args.output/'scenemind_demo_ground_truth.json'
    if video.exists() or truth.exists():
        raise FileExistsError('Demo/annotations already exist; refusing to overwrite')
    if not args.ffmpeg:
        raise RuntimeError('FFmpeg required to create browser-compatible H.264 MP4; pass --ffmpeg')
    with tempfile.TemporaryDirectory(prefix='demo-build-',dir=args.output) as temporary:
        raw=Path(temporary)/'render.avi'
        writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'MJPG'),FPS,(WIDTH,HEIGHT))
        if not writer.isOpened():
            raise RuntimeError('OpenCV could not initialize MJPG encoder')
        try:
            for index in range(60*FPS):
                writer.write(render(index/FPS))
        finally:
            writer.release()
        subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-n','-i',str(raw),
            '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    annotations={'schema':1,'video':video.name,'duration_seconds':60,'fps':FPS,
        'synthetic':True,'style':'filled cartoon figures; no text labels',
        'interval_convention':'start inclusive, end exclusive',
        'limitations':['Designed actions are not a guarantee of visual recognizability.',
            'Picking up requires motion evidence; frame CLIP may match only pose/object appearance.',
            'Walking/carrying overlap is included; categories are not mutually exclusive.'],
        'queries':[{'query':query,'intervals':intervals} for query,intervals in QUERIES]}
    truth.write_text(json.dumps(annotations,indent=2),encoding='utf-8')
    capture=cv2.VideoCapture(str(video))
    count,last=0,-1.0
    try:
        while True:
            ok,_=capture.read()
            if not ok: break
            timestamp=capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if timestamp<=last:
                raise RuntimeError('Demo timestamps are not increasing')
            last=timestamp
            count+=1
    finally:
        capture.release()
    if count!=60*FPS:
        raise RuntimeError(f'Incomplete demo: {count} frames')
    # One representative image per scene for honest visual inspection.
    thumbnails=[cv2.resize(render(t),(320,180)) for t in (6,18,30,42,52)]
    sheet=np.vstack([np.hstack(thumbnails[:3]),np.hstack(thumbnails[3:]+[np.full((180,320,3),245,np.uint8)])])
    cv2.imwrite(str(args.output/'scene_contact_sheet.jpg'),sheet)
    print(json.dumps({'video':str(video),'annotations':str(truth),'frames_decoded':count,
        'duration':count/FPS,'last_timestamp':last,'codec':'H.264/yuv420p'},indent=2))

if __name__=='__main__':
    main()
