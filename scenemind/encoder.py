"""Replaceable real OpenCLIP adapter; no fallback embeddings."""
from pathlib import Path
import numpy as np
import open_clip
import torch
from .core import Config

class OpenCLIPEncoder:
    def __init__(self, config: Config, device: str = "auto", cache_dir: Path | None = None):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.config = config
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.model, pretrained=config.pretrained, device=self.device,
            cache_dir=str(cache_dir) if cache_dir else None)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(config.model)

    @torch.inference_mode()
    def encode_images(self, images) -> np.ndarray:
        batches = []
        for start in range(0, len(images), self.config.batch_size):
            tensor = torch.stack([self.preprocess(image) for image in
                                  images[start:start + self.config.batch_size]]).to(self.device)
            batches.append(self.model.encode_image(tensor, normalize=True).float().cpu().numpy())
        return np.concatenate(batches)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Enter a nonempty query")
        tokens = self.tokenizer([text.strip()]).to(self.device)
        return self.model.encode_text(tokens, normalize=True).float().cpu().numpy()
