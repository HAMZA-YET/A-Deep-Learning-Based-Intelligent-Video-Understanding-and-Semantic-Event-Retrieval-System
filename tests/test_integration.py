"""Real OpenCV/FAISS integration; deterministic encoder is an explicit test double."""
import json
import cv2
import numpy as np
import pytest
from scenemind.core import Config
from scenemind.video import metadata, read_samples

@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'colors.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    assert writer.isOpened(), 'MJPG test encoder unavailable'
    for i in range(40):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :, 2 if i < 20 else 0] = 255
        writer.write(frame)
    writer.release()
    return path

class ColorTestEncoder:
    device = 'test-double'
    def encode_images(self, images):
        return np.array([np.asarray(image).mean(axis=(0, 1)) for image in images], dtype='float32')

def test_metadata_and_actual_samples(video):
    info = metadata(video)
    assert info['duration'] == pytest.approx(4)
    images, timestamps = read_samples(video, [.5, 2.5])
    assert len(images) == 2
    assert timestamps == pytest.approx([.5, 2.5], abs=.11)
    assert np.asarray(images[0])[0, 0, 0] > 240
    assert np.asarray(images[1])[0, 0, 2] > 240

def test_corrupt_video(tmp_path):
    bad = tmp_path / 'bad.mp4'
    bad.write_bytes(b'not a video')
    with pytest.raises(ValueError):
        metadata(bad)

def test_index_roundtrip_cache_and_rejection(video, tmp_path):
    from scenemind.retrieval import build_index, load_index
    config = Config(window_seconds=2, stride_seconds=2, frames_per_window=2)
    result = build_index(video, tmp_path / 'indexes', config, ColorTestEncoder())
    loaded = load_index(result.directory, config)
    matches = loaded.search([[1, 0, 0]], 5, .8)
    assert len(matches) == 1 and matches[0]['start'] == 0
    assert loaded.search([[0, 1, 0]], 5, .8) == []
    cached = build_index(video, tmp_path / 'indexes', config, None)
    assert cached.directory == result.directory
    with pytest.raises(ValueError):
        loaded.search([[1, 0]], 1, .2)
    with pytest.raises(ValueError):
        load_index(result.directory, Config())

def test_no_commit_on_inference_failure(video, tmp_path):
    from scenemind.retrieval import build_index
    class Broken:
        def encode_images(self, images):
            raise RuntimeError('intentional test failure')
    root = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='intentional'):
        build_index(video, root, Config(), Broken())
    assert not list(root.glob('*/manifest.json'))

@pytest.mark.parametrize('values', [[[0, 0]], [[float('nan'), 1]], [1, 2]])
def test_invalid_vectors(values):
    from scenemind.retrieval import normalized
    with pytest.raises(ValueError):
        normalized(values)

def test_app_initial_state():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    assert not app.exception
    assert app.title[0].value.startswith('SceneMind')
