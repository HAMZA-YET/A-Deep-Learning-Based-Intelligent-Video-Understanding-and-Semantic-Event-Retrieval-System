"""Actual-video indexing/search timing; no mocked encoder."""
from pathlib import Path
import argparse
import json
import time
from scenemind_mvp.config import Config
from scenemind_mvp.model import Encoder
from scenemind_mvp.pipeline import index_video

parser = argparse.ArgumentParser()
parser.add_argument("video", type=Path)
parser.add_argument("--query", default="a person")
parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
parser.add_argument("--data", type=Path, default=Path("data"))
args = parser.parse_args()
config = Config.load(args.config)
encoder = None
def factory():
    global encoder
    if encoder is None:
        encoder = Encoder(config, args.data / "models")
    return encoder
started = time.perf_counter()
indexed = index_video(args.video, args.data / "indexes", config, factory)
index_seconds = time.perf_counter() - started
encoder = factory()
started = time.perf_counter()
results = indexed.search_index.search(encoder.encode_text(args.query), config.top_k,
                                     config.min_gap_seconds, config.min_similarity)
print(json.dumps({"device": encoder.device, "cache_hit": indexed.cache_hit,
                  "index_seconds_including_load": index_seconds,
                  "query_seconds": time.perf_counter() - started,
                  "samples": indexed.manifest["sample_count"],
                  "backend": indexed.search_index.backend, "results": results}, indent=2))
