"""Exact normalized cosine search, with a visible NumPy equivalent if FAISS is unavailable."""
import logging
import numpy as np

logger = logging.getLogger(__name__)

def normalize(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=np.float32, order="C", copy=True)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Embeddings must be nonempty, finite 2D arrays")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero embeddings are invalid")
    return np.ascontiguousarray(values / norms)

class SearchIndex:
    def __init__(self, embeddings: np.ndarray, timestamps: np.ndarray, backend: str = "auto"):
        self.embeddings = normalize(embeddings)
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        if self.timestamps.shape != (len(self.embeddings),) or not np.isfinite(self.timestamps).all():
            raise ValueError("Timestamp/embedding count mismatch")
        if np.any(self.timestamps < 0) or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Timestamps must be nonnegative and increasing")
        if backend not in ("auto", "numpy", "faiss"):
            raise ValueError("Unknown search backend")
        self.backend, self.notice, self.index = "numpy", "", None
        if backend != "numpy":
            try:
                import faiss
                self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.index.add(self.embeddings)
                self.backend = "faiss"
            except (ImportError, OSError) as exc:
                if backend == "faiss":
                    raise
                self.notice = f"FAISS unavailable; using exact NumPy cosine search: {exc}"
                logger.warning(self.notice)

    def search(self, query: np.ndarray, top_k: int = 5, min_gap: float = 4,
               threshold: float = 0.0) -> list[dict]:
        query = normalize(query)
        if query.shape != (1, self.embeddings.shape[1]):
            raise ValueError("Query must contain one vector of the indexed model dimension")
        if top_k < 1 or not np.isfinite(min_gap) or min_gap < 0 or not np.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("Invalid retrieval parameters")
        if self.index is not None:
            scores, ids = self.index.search(query, len(self.timestamps))
            pairs = zip(ids[0], scores[0])
        else:
            scores = self.embeddings @ query[0]
            ids = np.argsort(-scores, kind="stable")
            pairs = ((i, scores[i]) for i in ids)
        results = []
        for index, score in pairs:
            timestamp = float(self.timestamps[index])
            if float(score) < threshold:
                continue
            if any(abs(timestamp - row["timestamp"]) < min_gap for row in results):
                continue
            results.append({"frame_index": int(index), "timestamp": timestamp,
                            "score": float(np.clip(score, -1, 1))})
            if len(results) == top_k:
                break
        return results
