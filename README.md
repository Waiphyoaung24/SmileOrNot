---
title: Vision
emoji: 😀
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
---

# Vision

Real-time browser-based object detection demo (smile + can detectors), end-to-end with the Ultralytics ecosystem:
data sourced from CelebA, auto-labeled with a pretrained face detector, refined
on Ultralytics Platform's annotation tooling, trained as a YOLO26n model on
Platform cloud GPUs, and self-hosted as a FastAPI service behind a static
Astro-built frontend, deployed to a free Hugging Face Docker Space.

[![Template CI](https://github.com/Waiphyoaung24/SmileOrNot/actions/workflows/ci.yml/badge.svg)](https://github.com/Waiphyoaung24/SmileOrNot/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Live demo

→ **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/<your-username>/smileornot)**

Click "Start Camera," grant webcam permission, and you'll see live bounding
boxes labeled `smiling` (green) or `neutral` (slate) on every detected face.

## How it was built

1. **Data** — Pulled the CelebA face dataset from Hugging Face. CelebA ships
   with a `Smiling` binary attribute per image.
2. **Auto-labeling** — Ran a pretrained `yolo11n-face` detector on each image
   to generate bounding boxes; combined with CelebA's `Smiling` attribute to
   produce YOLO-format labels (class 0 = smiling, 1 = neutral) for a balanced
   10K subset.
3. **Smart Annotation** — Uploaded the dataset to Ultralytics Platform and
   used Smart Annotation to refine misaligned boxes and wrong-class labels.
   See `docs/training.md` for screenshots.
4. **Training** — Fine-tuned YOLO26n (released January 2026, NMS-free) on
   Platform cloud GPU for 50 epochs at imgsz=640. Hit mAP50 > 0.9.
5. **Inference** — FastAPI service loads `weights/best.pt` once at startup;
   the `/predict` endpoint accepts JPEG frames and returns normalized bounding
   boxes as JSON.
6. **Frontend** — Astro project with `output: 'static'` builds plain
   HTML/CSS/JS — zero framework runtime. Vanilla JS handles `getUserMedia`,
   adaptive frame capture, and canvas overlay drawing.
7. **Deploy** — Multi-stage Dockerfile (Node build + slim Python runtime,
   CPU-only torch) pushed to Hugging Face as a Docker Space.

## Architecture

See [`docs/superpowers/specs/2026-04-27-smileornot-design.md`](docs/superpowers/specs/2026-04-27-smileornot-design.md)
for the full design document.

```
[Browser]                                  [HF Space — FastAPI]
  <video> webcam frame                       /predict (JPEG multipart)
       ↓                                            ↓
  capture → JPEG (480w) → POST ─────────→     YOLO26n inference (CPU)
       ↓                                            ↓
  draw normalized boxes ←──────────────  JSON: { boxes, inference_ms }
       ↓
  loop (adaptive: next frame after response)
```

## Local development

Two terminals:

```bash
# Backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
uvicorn smileornot.app:app --reload --port 8000
```

```bash
# Frontend
cd frontend && npm ci && npm run dev
# Open http://localhost:4321
```

## Reproduce the data + training pipeline

```bash
pip install huggingface_hub
python scripts/01_download_celeba.py    # ~1.4 GB
python scripts/02_autolabel.py          # ~10-20 min on CPU
python scripts/03_split.py
# Then upload data/yolo + scripts/dataset.yaml to Ultralytics Platform,
# run Smart Annotation, train YOLO26n (50 epochs, imgsz=640),
# and download best.pt to weights/.
```

## Test

```bash
cd frontend && npm run build && cd ..
pytest -v
```

## License

AGPL-3.0 (matching the Ultralytics ecosystem).
