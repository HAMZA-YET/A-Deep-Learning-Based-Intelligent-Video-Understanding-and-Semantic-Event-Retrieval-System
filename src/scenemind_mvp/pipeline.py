"""Content-addressed, transactional, bounded-frame-memory indexing."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
import time
import numpy as np
from PIL import Image
from .config import Config
from .search import SearchIndex, normalize
from .video import inspect_video, iter_frames

class ImageEncoder(Protocol):
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def identity(path: Path, config: Config) -> dict:
    weights = Path(config.pretrained)
    return {"schema": 2, "encoder_revision": "openai-quickgelu-v1", "video_sha256": digest(path), "config": config.embedding_config(),
            "weights_sha256": digest(weights) if weights.is_file() else None,
            "open_clip_version": importlib.metadata.version("open-clip-torch")}

def cache_key(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

@dataclass
class IndexedVideo:
    directory: Path
    manifest: dict
    search_index: SearchIndex
    cache_hit: bool = False

    def preview(self, index: int) -> Path:
        return self.directory / "previews" / f"{index:08d}.jpg"

def load_index(directory: Path) -> IndexedVideo:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["identity"].get("encoder_revision") != "openai-quickgelu-v1":
        raise ValueError("Index predates the encoder activation fix; reindex the video")
    if manifest["identity"]["schema"] != 2:
        raise ValueError("Unsupported cache schema")
    vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
    times = np.load(directory / "timestamps.npy", allow_pickle=False)
    if vectors.shape != (manifest["sample_count"], manifest["dimension"]):
        raise ValueError("Cache embedding dimensions do not match metadata")
    return IndexedVideo(directory, manifest, SearchIndex(vectors, times), True)

def index_video(path: Path, cache_root: Path, config: Config,
                encoder_factory: Callable[[], ImageEncoder],
                progress: Callable[[float], None] | None = None) -> IndexedVideo:
    info = inspect_video(path)
    signature = identity(path, config)
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / cache_key(signature)
    if (destination / "manifest.json").exists():
        cached = load_index(destination)
        if cached.manifest["identity"] != signature:
            raise ValueError("Cache identity mismatch")
        return cached
    encoder = encoder_factory()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="building-", dir=cache_root) as temporary:
        staging = Path(temporary)
        (staging / "previews").mkdir()
        embeddings, timestamps, batch = [], [], []
        def flush() -> None:
            if not batch:
                return
            vectors = normalize(encoder.encode_images([frame.image for frame in batch]))
            if len(vectors) != len(batch):
                raise ValueError("Encoder returned an incorrect batch shape")
            if embeddings and vectors.shape[1] != embeddings[0].shape[1]:
                raise ValueError("Encoder dimension changed during indexing")
            embeddings.append(vectors)
            for frame in batch:
                index = len(timestamps)
                timestamps.append(frame.timestamp)
                frame.image.thumbnail((320, 240))
                frame.image.save(staging / "previews" / f"{index:08d}.jpg", quality=80)
            if progress:
                progress(min(.99, timestamps[-1] / info.duration))
            batch.clear()
        for frame in iter_frames(path, config.sampling_fps):
            batch.append(frame)
            if len(batch) >= config.batch_size:
                flush()
        flush()
        if not embeddings:
            raise ValueError("No frames sampled")
        matrix = np.concatenate(embeddings)
        SearchIndex(matrix, np.array(timestamps), backend="numpy")
        np.save(staging / "embeddings.npy", matrix, allow_pickle=False)
        np.save(staging / "timestamps.npy", np.array(timestamps), allow_pickle=False)
        manifest = {"identity": signature, "source": str(path.resolve()), "metadata": asdict(info),
                    "sample_count": len(matrix), "dimension": matrix.shape[1],
                    "index_seconds": time.perf_counter() - started,
                    "device": getattr(encoder, "device", "external-test-encoder"),
                    "timestamp_source": "OpenCV decoded CAP_PROP_POS_MSEC"}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # A different session may have completed the same video concurrently.
        if not destination.exists():
            try:
                staging.rename(destination)
            except OSError:
                if not (destination / "manifest.json").exists():
                    raise
    if progress:
        progress(1.0)
    result = load_index(destination)
    result.cache_hit = False
    return result
