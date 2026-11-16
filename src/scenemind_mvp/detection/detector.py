"""Lightweight pretrained detector, with cached weights and no artificial detections."""
from pathlib import Path
import numpy as np
from .schema import DetectionConfig

class Detector:
    def __init__(self, config: DetectionConfig, model_cache: Path):
        import torch
        from ultralytics import YOLO
        self.torch=torch
        self.device='cuda:0' if config.device!='cpu' and torch.cuda.is_available() else 'cpu'
        if config.device=='cuda' and self.device=='cpu':
            raise RuntimeError('CUDA requested but unavailable; select CPU explicitly')
        model_cache.mkdir(parents=True,exist_ok=True)
        candidate=Path(config.model).expanduser()
        if candidate.is_file():
            self.weights=candidate.resolve()
        elif candidate.name=='yolo11n.pt' and str(candidate)==candidate.name:
            self.weights=model_cache/candidate.name
        else:
            raise ValueError('Use yolo11n.pt or an existing local compatible checkpoint')
        self.model=YOLO(str(self.weights))
        self.model.to(self.device)
        self.names=self.model.names
        self.config=config

    def infer(self, image: np.ndarray) -> np.ndarray:
        config=self.config
        threshold=min(config.confidence,config.low_threshold) if config.tracking else config.confidence
        try:
            result=self.model.predict(image,conf=threshold,classes=list(config.classes),
                imgsz=config.image_size,device=self.device,verbose=False)[0]
        except self.torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError('YOLO GPU out of memory. Select CPU or reduce image size; incomplete output was not published.') from exc
        return result.boxes.data.detach().cpu().numpy()[:,:6].astype(np.float32)
