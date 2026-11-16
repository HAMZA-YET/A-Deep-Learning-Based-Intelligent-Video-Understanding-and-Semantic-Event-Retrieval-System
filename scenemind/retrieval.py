"""Persistent exact FAISS cosine search over local trusted artifacts."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import logging
import uuid
import numpy as np
from .core import Config, windows, sample_times, select_results
from .video import metadata, read_samples

logger = logging.getLogger(__name__)

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def normalized(values) -> np.ndarray:
    array = np.array(values, dtype="float32", order="C", copy=True)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Expected finite nonempty 2D embeddings")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Cannot search zero embeddings")
    return np.ascontiguousarray(array / norms)

@dataclass
class VideoIndex:
    index: object
    manifest: dict
    directory: Path

    def search(self, query, top_k: int = 5, threshold: float = 0.25) -> list[dict]:
        query = normalized(query)
        if query.shape != (1, self.index.d):
            raise ValueError("Query dimension does not match index")
        scores, ids = self.index.search(query, self.index.ntotal)
        candidates = [{**self.manifest["clips"][int(i)], "score": float(score)}
                      for score, i in zip(scores[0], ids[0]) if i >= 0]
        return select_results(candidates, top_k, threshold)

def load_index(directory: Path, config: Config | None = None) -> VideoIndex:
    import faiss
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported index schema")
    if config is not None and manifest["config"] != config.to_dict():
        raise ValueError("Index configuration mismatch; rebuild index")
    index = faiss.read_index(str(directory / manifest["index_file"]))
    if index.ntotal != len(manifest["clips"]) or index.d != manifest["dimension"]:
        raise ValueError("Index and metadata disagree")
    return VideoIndex(index, manifest, directory)

def build_index(path: Path, root: Path, config: Config, encoder, progress=None) -> VideoIndex:
    """Commit metadata last; failed builds never appear ready."""
    import faiss
    sha = file_hash(path)
    identity = {"sha256": sha, "config": config.to_dict(),
                "open_clip_version": importlib.metadata.version("open-clip-torch")}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = root / key
    if (directory / "manifest.json").exists():
        cached = load_index(directory, config)
        if Path(cached.manifest["source"]).is_file():
            return cached
    info = metadata(path)
    intervals = windows(info["duration"], config)
    vectors, clips = [], []
    for i, (start, end) in enumerate(intervals):
        requested = sample_times(start, end, config.frames_per_window)
        images, observed = read_samples(path, requested)
        features = normalized(encoder.encode_images(images))
        vectors.append(normalized(features.mean(axis=0, keepdims=True))[0])
        clips.append({"start": start, "end": end,
                      "requested_samples": requested, "observed_samples": observed})
        if progress:
            progress((i + 1) / len(intervals))
    matrix = normalized(vectors)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    directory.mkdir(parents=True, exist_ok=True)
    index_name = f"vectors-{uuid.uuid4().hex}.faiss"
    faiss.write_index(index, str(directory / index_name))
    manifest = {"schema": 1, **identity, "source": str(path.resolve()),
                "dimension": matrix.shape[1], "metadata": info, "clips": clips,
                "index_file": index_name,
                "representation": "mean-pooled frame embeddings; no temporal reasoning"}
    temporary = directory / f"manifest-{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    logger.info("Indexed video=%s windows=%d device=%s", sha, len(clips), encoder.device)
    return VideoIndex(index, manifest, directory)
