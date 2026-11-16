# SceneMind

**A Deep Learning Based Intelligent Video Understanding and Semantic Event Retrieval System**

University FYP prototype for local semantic video search and evidence inspection.

## Current functionality
- OpenCLIP full-frame, region and hybrid retrieval; OpenCV sampling and FAISS cosine ranking.
- YOLO detection, ByteTrack temporary track references, candidate person-object associations and conservative temporal evidence.
- Unified Original / Annotated playback, multiple-person boxes, query-specific highlighting and timestamp navigation.
- Stable search form: type without reruns, then press Search or Enter. Clear search or an empty submission restores Show All.

Similarity scores are not probabilities or proof of actions. Track IDs are temporary tracklets, not real-world identities. Clothing uses uncertain torso-color cues and does not reliably distinguish shirts from dresses. Proximity does not establish carrying, contact or ownership. No facial recognition.

## Setup
Use Python 3.11, FFmpeg on PATH and a compatible PyTorch/torchvision pair installed using the official PyTorch instructions. Preserve a working CUDA environment. From the project directory in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
# Optional temporal model dependencies:
.\.venv\Scripts\python.exe -m pip install --no-deps pytorchvideo==0.1.5 fvcore==0.1.5.post20221221 iopath==0.1.10 yacs==0.1.8 termcolor portalocker parameterized tabulate
.\.venv\Scripts\python.exe scripts/verify_gpu.py
.\.venv\Scripts\python.exe scripts/phase3/download_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501 and upload an MP4 or select a local video. First processing downloads models/builds indexes. Runtime files stay in ignored `data/`. Model weights and source videos are not redistributed; upstream licensing terms apply. `requirements.lock.txt` records the development environment, not a portable CUDA installer.

## Tests
```powershell
# Self-contained baseline:
.\.venv\Scripts\python.exe -m pytest tests/phase1 -q
# Full development suite, requiring local videos/model/evaluation caches:
.\.venv\Scripts\python.exe -m pytest tests -q
```

The public checkout excludes sample videos and generated evaluation artifacts. Some integration tests require these local fixtures and will fail or skip until recreated.

Latest local search-form regression run: **46 passed in 139.30 seconds**. Actual browser verification executed five searches in one session with **138 individually typed characters**, maintaining focus in Original and Annotated playback. These validate UI/pipeline behavior, not clothing or action accuracy. See [verification notes](docs/search-focus-verification.md).

## Layout
- `app.py`: Streamlit application.
- `src/scenemind_mvp/`: retrieval, detection, tracking, associations, temporal reasoning and visualization.
- `configs/`: inference/ranking configurations.
- `scripts/`: indexing, evaluation and model preparation.
- `tests/`: unit and integration tests.
- `scenemind/`, `legacy/`: preserved earlier implementations.

## Limitations
Small or occluded objects can be missed; tracks fragment; temporal models can miss plausible actions. Ambiguous evidence remains uncertain. Local development results are not a held-out generalization study. This publication makes no new accuracy claims.
