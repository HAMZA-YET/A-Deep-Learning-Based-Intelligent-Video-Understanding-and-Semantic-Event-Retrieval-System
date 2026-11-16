"""Bounded-memory sequential decoding using decoded OpenCV timestamps."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import math
import cv2
from PIL import Image

@dataclass(frozen=True)
class VideoMetadata:
    duration: float
    fps: float
    width: int
    height: int
    frame_count: int
    duration_source: str = "frame_count / nominal_fps (estimate for VFR)"

@dataclass
class Frame:
    timestamp: float
    image: Image.Image


def inspect_video(path: Path) -> VideoMetadata:
    if not path.is_file() or path.suffix.lower() != ".mp4" or path.stat().st_size == 0:
        raise ValueError("Select a nonempty local MP4 file")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open MP4: invalid file or unsupported codec")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (int(capture.get(key)) for key in
                         (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or min(count, width, height) <= 0:
            raise ValueError("Invalid video metadata")
        ok, _ = capture.read()
        if not ok:
            raise ValueError("MP4 contains no decodable frames")
        return VideoMetadata(count / fps, fps, width, height, count)
    finally:
        capture.release()


def iter_frames(path: Path, sampling_fps: float) -> Iterator[Frame]:
    """Decode sequentially and retrieve only sampled frames; never hold the video in RAM."""
    info = inspect_video(path)
    if not math.isfinite(sampling_fps) or sampling_fps <= 0:
        raise ValueError("Sampling FPS must be positive")
    capture = cv2.VideoCapture(str(path))
    target, previous, decoded = 0.0, -1.0, 0
    try:
        while capture.grab():
            decoded += 1
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
                raise ValueError("Decoder did not expose increasing timestamps; transcode MP4 with FFmpeg")
            previous = timestamp
            if timestamp + 1e-6 < target:
                continue
            ok, image = capture.retrieve()
            if not ok:
                raise ValueError(f"Frame decode failed near {timestamp:.3f}s")
            yield Frame(timestamp, Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            target = (math.floor(timestamp * sampling_fps + 1e-6) + 1) / sampling_fps
        if decoded < info.frame_count:
            raise ValueError(f"Video ended early: decoded {decoded}/{info.frame_count} frames")
    finally:
        capture.release()
