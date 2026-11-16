"""Real pretrained OpenCLIP inference, with explicit device fallback reporting."""
from pathlib import Path
from typing import Sequence
import logging
import numpy as np
from PIL import Image
from .config import Config

logger = logging.getLogger(__name__)

class Encoder:
    def __init__(self, config: Config, cache_dir: Path):
        try:
            import torch
            import open_clip
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"PyTorch/OpenCLIP could not load native libraries: {exc}") from exc
        self.torch = torch
        self.batch_size = config.batch_size
        self.notices: list[str] = []
        self.device = "cuda" if config.device != "cpu" and torch.cuda.is_available() else "cpu"
        if config.device == "cuda" and self.device == "cpu":
            self.notices.append("CUDA unavailable; using CPU")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Load on CPU first so failed GPU initialization does not force a second download.
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                config.model, pretrained=config.pretrained, device="cpu", cache_dir=str(cache_dir),
                force_quick_gelu=(config.pretrained == "openai"))
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(config.model)
        except Exception as exc:
            raise RuntimeError(f"Cannot load pretrained model. Check model name, disk space and download access: {exc}") from exc
        try:
            self.model.to(self.device)
            if self.device == "cuda":
                probe = torch.ones((16, 16), device="cuda")
                _ = probe @ probe
                torch.cuda.synchronize()
        except RuntimeError as exc:
            self._cpu_fallback(f"CUDA initialization failed: {exc}")

    def _cpu_fallback(self, reason: str) -> None:
        self.model.to("cpu")
        self.device = "cpu"
        self.torch.cuda.empty_cache()
        self.notices.append(reason + "; using CPU")
        logger.warning(self.notices[-1])

    def _infer(self, tensor, image: bool) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            try:
                method = self.model.encode_image if image else self.model.encode_text
                output = method(tensor.to(self.device), normalize=True)
                return output.float().cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                if self.device != "cuda":
                    raise
        self._cpu_fallback("GPU out of memory")
        with torch.inference_mode():
            method = self.model.encode_image if image else self.model.encode_text
            return method(tensor.to("cpu"), normalize=True).float().cpu().numpy()

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("No images supplied")
        return self._infer(self.torch.stack([self.preprocess(image) for image in images]), True)

    def encode_text(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Enter a nonempty natural-language query")
        return self._infer(self.tokenizer([query.strip()]), False)
