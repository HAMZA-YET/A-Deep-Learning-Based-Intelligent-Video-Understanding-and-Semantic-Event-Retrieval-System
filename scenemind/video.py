"""OpenCV decoding; duration is FPS-based, not exact VFR timing."""
from pathlib import Path
import math
import cv2
from PIL import Image

def metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video; check container and codec")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("Invalid video FPS/frame count")
        return {"fps": fps, "frame_count": count, "duration": count / fps,
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "timing": "OpenCV seek timestamps; FPS-based duration estimate"}
    finally:
        capture.release()

def read_samples(path: Path, times: list[float]):
    capture = cv2.VideoCapture(str(path))
    images, observed = [], []
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video")
        for timestamp in times:
            if not capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000):
                raise ValueError(f"Cannot seek to {timestamp:.3f}s")
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode sample at {timestamp:.3f}s")
            actual = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not math.isfinite(actual) or actual < 0:
                raise ValueError("Invalid decoded timestamp")
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            observed.append(actual)
        return images, observed
    finally:
        capture.release()
