"""Verify actual CUDA execution, synchronization and CPU agreement."""
import json
import torch

torch.manual_seed(7)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable')
a, b = torch.randn(256, 256), torch.randn(256, 256)
result = a.to('cuda') @ b.to('cuda')
torch.cuda.synchronize()
torch.testing.assert_close(result.cpu(), a @ b, rtol=1e-3, atol=1e-3)
assert result.is_cuda and torch.isfinite(result).all()
print(json.dumps({'status': 'PASS', 'torch': torch.__version__,
                  'gpu': torch.cuda.get_device_name(0), 'device': str(result.device),
                  'checksum': result.sum().item()}))
