"""Phase 1 tests. ColorEncoder is a deterministic test double, never production AI."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import cv2
import pytest
from scenemind_mvp.config import Config
from scenemind_mvp.video import inspect_video, iter_frames
from scenemind_mvp.search import SearchIndex, normalize
from scenemind_mvp.pipeline import index_video, identity, cache_key, load_index

@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'sample.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
    assert writer.isOpened()
    for i in range(40):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :, 2 if i < 20 else 0] = 255
        writer.write(frame)
    writer.release()
    return path

class ColorEncoder:
    device = 'explicit-color-test-double'
    def __init__(self):
        self.max_batch = 0
    def encode_images(self, images):
        self.max_batch = max(self.max_batch, len(images))
        return np.array([np.asarray(image).mean(axis=(0,1)) for image in images], dtype=np.float32)

@pytest.mark.parametrize('kwargs', [dict(sampling_fps=0),dict(sampling_fps=float('nan')),
    dict(batch_size=0),dict(top_k=0),dict(device='bogus'),dict(min_similarity=2),dict(min_gap_seconds=-1)])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)

def test_timestamp_extraction_and_rgb(video):
    info = inspect_video(video)
    assert (info.width, info.height, info.fps, info.duration) == (64,48,10,4)
    frames = list(iter_frames(video,1))
    assert [frame.timestamp for frame in frames] == pytest.approx([0,1,2,3], abs=.01)
    assert np.asarray(frames[0].image)[0,0,0] > 240
    assert np.asarray(frames[-1].image)[0,0,2] > 240

def test_fractional_sampling(video):
    assert [f.timestamp for f in iter_frames(video,2)] == pytest.approx(np.arange(0,4,.5),abs=.01)

@pytest.mark.parametrize('content', [b'',b'not a video'])
def test_corrupt_video(tmp_path,content):
    path=tmp_path/'corrupt.mp4'
    path.write_bytes(content)
    with pytest.raises(ValueError):
        inspect_video(path)

def test_missing_video(tmp_path):
    with pytest.raises(ValueError):
        inspect_video(tmp_path/'missing.mp4')

@pytest.mark.parametrize('value',[np.empty((0,3)),np.zeros((1,3)),np.array([[float('nan'),1]]),np.array([1,2])])
def test_empty_invalid_embeddings(value):
    with pytest.raises(ValueError):
        normalize(value)

def test_shape_normalization():
    actual=normalize(np.array([[3,4],[5,12]]))
    assert actual.shape==(2,2) and actual.dtype==np.float32
    np.testing.assert_allclose(np.linalg.norm(actual,axis=1),[1,1])

def test_ranking_and_temporal_diversity():
    index=SearchIndex([[1,0],[.9,.1],[0,1]], [0,1,10],backend='numpy')
    rows=index.search([[1,0]],5,4,-1)
    assert [r['timestamp'] for r in rows]==[0,10]
    assert rows[0]['score']==pytest.approx(1)
    assert index.search([[-1,0]],5,0,.8)==[]

@pytest.mark.parametrize('query',[np.empty((0,2)),[[0,0]],[[1,0,0]]])
def test_bad_query_vectors(query):
    index=SearchIndex([[1,0]],[0],backend='numpy')
    with pytest.raises(ValueError):
        index.search(query)

def test_cache_batch_bound_shape_and_reindex(video,tmp_path):
    config=Config(batch_size=2)
    encoder=ColorEncoder()
    result=index_video(video,tmp_path/'cache',config,lambda:encoder)
    assert not result.cache_hit and encoder.max_batch<=2
    assert result.search_index.embeddings.shape==(4,3)
    assert result.manifest['sample_count']==4
    assert all(result.preview(i).is_file() for i in range(4))
    result2=index_video(video,tmp_path/'cache',config,lambda:pytest.fail('Cache must avoid model load'))
    assert result2.cache_hit
    np.testing.assert_array_equal(result.search_index.timestamps,result2.search_index.timestamps)
    assert cache_key(identity(video,replace(config,sampling_fps=2)))!=result.directory.name
    original=identity(video,config)
    video.write_bytes(video.read_bytes()+b'changed')
    assert cache_key(identity(video,config))!=cache_key(original)

def test_failed_index_not_published(video,tmp_path):
    class Broken:
        def encode_images(self,images):
            raise RuntimeError('intentional inference failure')
    with pytest.raises(RuntimeError,match='intentional'):
        index_video(video,tmp_path/'cache',Config(),lambda:Broken())
    assert not list((tmp_path/'cache').glob('*/manifest.json'))

def test_wrong_encoder_shape(video,tmp_path):
    class Wrong:
        def encode_images(self,images):
            return np.ones((1,3))
    with pytest.raises(ValueError,match='batch shape'):
        index_video(video,tmp_path/'cache',Config(),lambda:Wrong())

def test_query_settings_reuse_identity(video):
    assert identity(video,Config())==identity(video,Config(top_k=10,min_gap_seconds=1))

def test_empty_text_rejected_without_inference():
    from scenemind_mvp.model import Encoder
    encoder=Encoder.__new__(Encoder)
    with pytest.raises(ValueError,match='nonempty'):
        encoder.encode_text('   ')

def test_streamlit_startup():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).resolve().parents[2]/'app.py')).run()
    assert not app.exception
    assert app.title[0].value.startswith('SceneMind')
    assert app.button[0].disabled

def test_faiss_or_explicit_exact_fallback():
    index=SearchIndex([[1,0],[0,1]],[0,5])
    assert index.backend in ('faiss','numpy')
    if index.backend=='numpy':
        assert 'FAISS unavailable' in index.notice
    assert index.search([[1,0]],1)[0]['timestamp']==0

def test_gpu_oom_retries_on_cpu_with_explicit_notice():
    """Control-flow test with tensor/model doubles; not a real GPU test."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    from scenemind_mvp.model import Encoder
    class OOM(RuntimeError):
        pass
    class Tensor:
        def to(self,device): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return np.array([[1,0]],dtype=np.float32)
    class Model:
        def __init__(self): self.calls=0
        def to(self,device): return self
        def encode_image(self,tensor,normalize=True):
            self.calls+=1
            if self.calls==1: raise OOM('test-only')
            return Tensor()
    encoder=Encoder.__new__(Encoder)
    encoder.model=Model()
    encoder.device='cuda'
    encoder.notices=[]
    encoder.torch=SimpleNamespace(inference_mode=nullcontext,
        cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None))
    actual=encoder._infer(Tensor(),True)
    assert actual.shape==(1,2) and encoder.device=='cpu'
    assert encoder.model.calls==2 and 'GPU out of memory' in encoder.notices[0]

def test_search_parameters_validation():
    index=SearchIndex([[1,0]],[0],backend='numpy')
    with pytest.raises(ValueError): index.search([[1,0]],top_k=0)
    with pytest.raises(ValueError): index.search([[1,0]],min_gap=-1)

def test_pre_activation_fix_cache_is_rejected(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'identity':{'schema':2}}))
    with pytest.raises(ValueError,match='activation fix'):
        load_index(tmp_path)
