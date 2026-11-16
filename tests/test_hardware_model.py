"""Explicit hardware/model checks; these never substitute mock AI."""
import os
import pytest

@pytest.mark.skipif(os.getenv('SCENEMIND_RUN_GPU') != '1', reason='Set SCENEMIND_RUN_GPU=1 for real CUDA computation')
def test_cuda_matmul():
    import torch
    assert torch.cuda.is_available()
    torch.manual_seed(7)
    a = torch.randn(256, 256)
    b = torch.randn(256, 256)
    actual = a.to('cuda') @ b.to('cuda')
    torch.cuda.synchronize()
    assert actual.is_cuda
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), a @ b, rtol=1e-3, atol=1e-3)

@pytest.mark.skipif(os.getenv('SCENEMIND_RUN_MODEL') != '1', reason='Set SCENEMIND_RUN_MODEL=1; downloads pretrained weights')
def test_real_openclip_video_retrieval(tmp_path):
    import cv2
    import numpy as np
    from scenemind.encoder import OpenCLIPEncoder
    from scenemind.core import Config
    from scenemind.retrieval import build_index, load_index
    path = tmp_path / 'model-smoke.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.full((48, 64, 3), (0, 0, 255), dtype=np.uint8))
    writer.release()
    config = Config(frames_per_window=2)
    encoder = OpenCLIPEncoder(config, 'cuda', tmp_path / 'models')
    index = build_index(path, tmp_path / 'indexes', config, encoder)
    result = load_index(index.directory, config).search(encoder.encode_text('a red scene'), 1, -1)
    assert len(result) == 1 and np.isfinite(result[0]['score'])
    # Smoke test only: synthetic colors do not establish semantic event accuracy.
