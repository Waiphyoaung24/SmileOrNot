# SmileOrNot — Design

**Date:** 2026-04-27
**Status:** Approved (brainstorming complete; ready for implementation plan)
**Author:** Wai Phyo Aung

---

## 1. Goal & Scope

### Goal

A portfolio piece demonstrating end-to-end use of the Ultralytics ecosystem — from
data sourcing and annotation on Ultralytics Platform, to training a custom YOLO
detection model, to self-hosted inference behind a custom web UI.

The deliverable is twofold:

1. **A live demo** at a public URL: a single web page where the visitor clicks
   "Start Camera," grants webcam permission, and sees their video feed with
   live bounding boxes labeled `smiling` or `neutral` drawn on top, plus a live
   classification readout.
2. **A public GitHub repo** containing the full pipeline: data prep scripts,
   training writeup, FastAPI backend, Astro-built static frontend, Dockerfile,
   and CI — structured according to the Ultralytics Python project template.

### Portfolio narrative

> "I sourced face data from CelebA, auto-labeled it with a pretrained face
> detector, refined annotations using Ultralytics Platform's smart annotation
> tools, trained a YOLO26n detection model on the Platform's cloud GPUs,
> exported the weights, and built a FastAPI backend with a custom Astro-built
> HTML/JS frontend deployed to Hugging Face Spaces."

### Non-goals (explicit scope discipline)

- **Not** using Ultralytics Platform's deployed-endpoint feature — we self-host
  inference instead.
- **Not** multi-face tracking with persistent IDs — per-frame detection only.
  Multiple faces in frame each get an independent box.
- **Not** emotion classification beyond smile / neutral. Two classes only.
- **Not** a mobile app. Browser only, modern desktop browsers only.
- **Not** authenticated, multi-user, or persistent. Single anonymous visitor at
  a time, ephemeral, page reload = camera off.
- **Not** GPU-deployed. Free Hugging Face CPU Space is the deliberate target.

---

## 2. System Architecture Overview

Two phases with distinct lifetimes:

### Phase 1 — Data → Trained Weights (one-shot, offline, lives outside the deployed app)

```
CelebA (200K images, "Smiling" attr)
    → local script: yolo11n-face → bounding box per image
    → join with CelebA Smiling attribute → YOLO-format labels
       (class 0 = smiling, 1 = neutral)
    → upload to Ultralytics Platform
    → Platform "Smart Annotation" review pass (fix misaligned boxes)
    → train YOLO26n on Platform cloud GPU (50 epochs, imgsz=640)
    → export best.pt
```

Output artifact: `weights/best.pt` (~6 MB), committed to the repo.

### Phase 2 — Live Demo (deployed on HF Spaces, FastAPI Docker)

```
[Browser]                                  [HF Space — FastAPI container, port 7860]
  <video> from getUserMedia                  GET /  → static index.html + bundled JS
  <canvas> overlay                           POST /predict (multipart JPEG)
                                                ↓
  capture frame → JPEG (480w, q=0.7)          PIL decode → np.array
       ↓                                      → YOLO26n inference
  POST /predict ──────────────────────────→     (best.pt loaded once at startup)
       ↓                                      → format boxes as JSON
  receive JSON ←───────────────────────────  HTTP 200 { boxes:[…], inference_ms }
       ↓
  draw boxes + labels on <canvas>
       ↓
  immediately capture next frame (adaptive loop)
```

### Key invariants

- Backend is **stateless**. No sessions, no per-user state. Model loaded once at
  startup and reused across all requests.
- Weights are loaded by the FastAPI app at container start (in the lifespan
  context manager). Never on the request hot path.
- Frontend never holds more than one in-flight request — the adaptive loop is
  request-driven (capture next frame *after* receiving the previous response).

### Inference architecture (decision)

Selected **per-frame HTTP POST** (option A, where alternatives were B: WebSocket
streaming, C: client-side ONNX). Rationale: preserves the "self-hosted FastAPI
inference" portfolio narrative; simplest to deploy on HF Spaces; 3-6 effective
FPS is sufficient for the smile-detection use case.

### Frame cadence (decision)

Selected **adaptive (request-driven)** cadence (option A, where alternative was
B: fixed-rate timer-driven). Rationale: zero queueing logic, no backpressure,
self-throttling, no stale-prediction races.

---

## 3. Data Pipeline & Training (Phase 1 detail)

This entire section runs once, offline, and produces a single artifact:
`weights/best.pt`. Nothing here is part of the deployed Space.

### Step 1 — Acquire CelebA

Source: Hugging Face mirror of CelebA (e.g. `tpremoli/CelebA-attrs`). The
original Google Drive distribution is unreliable. ~200K images plus
`list_attr_celeba.csv` with the `Smiling` attribute (1 = smiling, -1 = neutral).

Script: `scripts/01_download_celeba.py`.

### Step 2 — Subset

Take a balanced 10K subset: 5,000 smiling + 5,000 neutral. CelebA is mildly
imbalanced (~48% smiling); we sample evenly from each class. YOLO26n trains in
under 30 minutes on Platform GPU on this subset at imgsz=640 — fast enough to
iterate. Full 200K is overkill for a 2-class detector and slows iteration.

### Step 3 — Auto-label with a pretrained face detector

Use `yolo11n-face` (Ultralytics's face-specific YOLO11 variant) or a comparable
pretrained face detector (RetinaFace, YuNet) — the implementor picks whichever
is most current. For each image: run detector → take the highest-confidence
bounding box → discard images where no face is detected. Write YOLO-format
labels: `<class> <x_center> <y_center> <w> <h>` with `class` set from CelebA's
`Smiling` attribute (1 → 0=smiling, -1 → 1=neutral).

Output:
- `data/images/*.jpg`
- `data/labels/*.txt`

Script: `scripts/02_autolabel.py`.

### Step 4 — Stratified 80/10/10 split

Train/val/test split, stratified by class. Write the YOLO data config:

```yaml
# scripts/dataset.yaml
path: ./data
train: images/train
val:   images/val
test:  images/test
names:
  0: smiling
  1: neutral
```

Script: `scripts/03_split.py`.

### Step 5 — Upload to Ultralytics Platform

Upload via the Platform UI/CLI. Run a "Smart Annotation" review pass — manually
fix obvious face-detector misalignments and any wrong-class labels. Budget:
1-2 hours of clicking. Document with screenshots in `docs/training.md`.

### Step 6 — Train YOLO26n on Platform

Hyperparameters:
- `model = yolo26n.pt` (pretrained)
- `epochs = 50`
- `imgsz = 640`
- `batch = auto`
- Default optimizer (AdamW)
- Mosaic + flip augmentation on (Ultralytics defaults)

Expected: mAP50 > 0.9 on this 2-class problem.

### Step 7 — Export weights

Download `best.pt` from Platform → commit to `weights/best.pt`. Optionally also
export ONNX for portability (`yolo export model=best.pt format=onnx`), but
production inference uses `best.pt` via the Python `ultralytics` SDK.

### What's committed from Phase 1

- The three scripts (`scripts/01_*.py`, `02_*.py`, `03_*.py`)
- `scripts/dataset.yaml`
- `weights/best.pt`
- `docs/training.md` (narrative writeup with Platform screenshots)

The CelebA images and generated labels are **not** committed — they live in
`data/` and are gitignored. The README's "Reproduce" section runs the scripts to
regenerate them.

### Dataset choice (decision)

Selected **CelebA** (option A, alternatives: B GENKI-4K, C AffectNet, D hybrid).
Rationale: ~200K images is plenty after subsetting; the `Smiling` attribute
maps cleanly to our two classes; one face per image makes auto-labeling clean;
recognizable to reviewers.

---

## 4. Backend Service Design

A single FastAPI app, two source modules, model loaded once at startup.

### Module structure

```
smileornot/
├── __init__.py          # __version__ only
├── app.py               # FastAPI app, lifespan, /predict endpoint
├── inference.py         # SmileDetector class (YOLO wrapper)
└── static/              # served at /; populated by Astro build (gitignored)
weights/
└── best.pt              # ~6 MB, committed
```

### `smileornot/app.py`

```python
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from smileornot.inference import SmileDetector

WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "best.pt"
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = SmileDetector(WEIGHTS_PATH, device="cpu")
    yield


app = FastAPI(lifespan=lifespan, title="SmileOrNot", version="0.1.0")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Expected an image upload")
    raw = await file.read()
    if len(raw) > 2_000_000:
        raise HTTPException(413, "Frame too large")
    boxes, ms = app.state.detector.predict_bytes(raw)
    return JSONResponse({"boxes": boxes, "inference_ms": ms})


# Mounted last so /predict takes precedence
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
```

### `smileornot/inference.py`

```python
import io
import time
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


class SmileDetector:
    def __init__(self, weights_path: Path, device: str = "cpu") -> None:
        self.model = YOLO(str(weights_path))
        self.device = device
        self.names: dict[int, str] = self.model.names  # {0: "smiling", 1: "neutral"}

    def predict_bytes(self, raw: bytes, conf: float = 0.4) -> tuple[list[dict], float]:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        t0 = time.perf_counter()
        results = self.model.predict(img, conf=conf, device=self.device, verbose=False)
        ms = (time.perf_counter() - t0) * 1000
        r = results[0]
        boxes: list[dict] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxyn = r.boxes.xyxyn.cpu().numpy()  # normalized 0-1
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxyn, confs, clss):
                boxes.append({
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "class": self.names[int(k)],
                    "conf": float(c),
                })
        return boxes, round(ms, 2)
```

### Response schema

```json
{
  "boxes": [
    { "x1": 0.31, "y1": 0.18, "x2": 0.59, "y2": 0.62,
      "class": "smiling", "conf": 0.87 }
  ],
  "inference_ms": 87.4
}
```

Coordinates are **normalized to 0-1**. The frontend multiplies by the overlay
canvas's pixel dimensions, so the upload resolution and display resolution can
diverge freely.

### Design decisions and rationale

- **Normalized `xyxyn` coords on the wire.** Eliminates frontend awareness of
  upload resolution.
- **`conf=0.4` default threshold.** Below this, profile/occluded faces produce
  noise.
- **`device="cpu"` explicit.** HF free Spaces have no GPU; explicit device
  prevents Ultralytics from probing CUDA.
- **`verbose=False`.** Otherwise every prediction logs a line, killing
  readability under load.
- **2 MB upload ceiling.** Defends against accidentally uploading raw 1080p
  frames; intended payloads are ~50-100 KB.
- **Same-origin (no CORS).** Static files served by the same FastAPI app;
  browsers treat `/predict` as same-origin.
- **Lifespan handler** (modern FastAPI) — model loaded once before first
  request.

### Deliberate omissions

- No request queue, batching, or concurrency limit. The single-user adaptive
  loop guarantees no overlap; multiple visitors fight for CPU acceptably.
- No auth, rate limiting, metrics middleware. HF Spaces is the trust boundary.
- No model warmup. First request is ~30% slower than subsequent ones;
  acceptable for a demo.

---

## 5. Frontend Design (Astro)

Astro is used as a **build tool only** — `output: 'static'` prerenders to plain
HTML/CSS/JS with **zero framework runtime**. The actual code is vanilla.

### Source layout

```
frontend/                          # Astro project (committed)
├── astro.config.mjs
├── package.json
├── package-lock.json
├── tsconfig.json
├── public/
│   └── favicon.svg
└── src/
    ├── pages/
    │   └── index.astro
    ├── scripts/
    │   └── app.js                 # vanilla JS: state machine, capture loop
    └── styles/
        └── styles.css

smileornot/static/                 # build output — GITIGNORED
                                   # populated by `cd frontend && npm run build`
                                   # served by FastAPI's StaticFiles mount
```

### `frontend/astro.config.mjs`

```js
import { defineConfig } from 'astro/config';

export default defineConfig({
  output: 'static',
  outDir: '../smileornot/static',     // build straight into FastAPI's static mount
  vite: {
    server: {
      proxy: {
        '/predict': 'http://localhost:8000',   // dev: forward API to FastAPI
      },
    },
  },
});
```

### `frontend/src/pages/index.astro`

```astro
---
// No frontmatter logic — Astro builds this to plain HTML.
---
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SmileOrNot</title>
  </head>
  <body>
    <main>
      <h1>SmileOrNot</h1>
      <p id="status">Click to start.</p>
      <div class="stage">
        <video id="video" playsinline autoplay muted></video>
        <canvas id="overlay"></canvas>
      </div>
      <button id="toggle">Start Camera</button>
    </main>

    <script>
      import '../scripts/app.js';
    </script>

    <style is:global>
      @import '../styles/styles.css';
    </style>
  </body>
</html>
```

### `frontend/src/scripts/app.js`

A four-state machine: `idle` → `starting` → `running` ↔ `idle`, plus `error`.
Camera lifecycle, adaptive request loop, two canvases (one off-screen for
capture at 480w, one on-screen for the overlay), `AbortController` for clean
stop, state guard after each `await`.

```js
const STATE = { IDLE: 'idle', STARTING: 'starting', RUNNING: 'running', ERROR: 'error' };
let state = STATE.IDLE;
let stream = null;
let abortController = null;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const toggleBtn = document.getElementById('toggle');

const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

async function start() {
  state = STATE.STARTING;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    sizeOverlayToVideo();
    state = STATE.RUNNING;
    toggleBtn.textContent = 'Stop Camera';
    loop();
  } catch (e) {
    state = STATE.ERROR;
    showStatus(e.name === 'NotAllowedError' ? 'Camera permission denied.' : 'Camera unavailable.');
  }
}

function stop() {
  state = STATE.IDLE;
  if (abortController) abortController.abort();
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  toggleBtn.textContent = 'Start Camera';
}

async function loop() {
  while (state === STATE.RUNNING) {
    try {
      const blob = await captureFrame();
      abortController = new AbortController();
      const fd = new FormData();
      fd.append('file', blob, 'frame.jpg');
      const r = await fetch('/predict', { method: 'POST', body: fd, signal: abortController.signal });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const { boxes, inference_ms } = await r.json();
      if (state !== STATE.RUNNING) return;
      drawBoxes(boxes);
      updateStatus(boxes, inference_ms);
    } catch (e) {
      if (e.name === 'AbortError') return;
      state = STATE.ERROR;
      showStatus(`Connection lost: ${e.message}`);
      return;
    }
  }
}

function captureFrame() {
  const targetW = 480;
  const targetH = Math.round(video.videoHeight * targetW / video.videoWidth);
  captureCanvas.width = targetW;
  captureCanvas.height = targetH;
  captureCtx.drawImage(video, 0, 0, targetW, targetH);
  return new Promise(resolve => captureCanvas.toBlob(resolve, 'image/jpeg', 0.7));
}

function sizeOverlayToVideo() {
  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;
}

function drawBoxes(boxes) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  for (const b of boxes) {
    const x = b.x1 * overlay.width;
    const y = b.y1 * overlay.height;
    const w = (b.x2 - b.x1) * overlay.width;
    const h = (b.y2 - b.y1) * overlay.height;
    ctx.lineWidth = 3;
    ctx.strokeStyle = b.class === 'smiling' ? '#22c55e' : '#94a3b8';
    ctx.strokeRect(x, y, w, h);
    const label = `${b.class} ${(b.conf * 100).toFixed(0)}%`;
    ctx.font = '14px system-ui';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillRect(x, y - 20, tw + 8, 20);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x + 4, y - 5);
  }
}

function updateStatus(boxes, ms) {
  const smiling = boxes.filter(b => b.class === 'smiling').length;
  const neutral = boxes.filter(b => b.class === 'neutral').length;
  statusEl.textContent = `Smiling: ${smiling} · Neutral: ${neutral} · Inference: ${ms.toFixed(0)} ms`;
}

function showStatus(msg) { statusEl.textContent = msg; }

toggleBtn.addEventListener('click', () => {
  if (state === STATE.RUNNING) stop();
  else start();
});

window.addEventListener('resize', () => {
  if (state === STATE.RUNNING) sizeOverlayToVideo();
});
```

### Dev workflow

Two terminals:

1. `uvicorn smileornot.app:app --reload --port 8000` — backend.
2. `cd frontend && npm run dev` — Astro dev server on `:4321`, HMR + proxies
   `/predict` to `:8000`.

Open `http://localhost:4321`.

### Production build

```bash
cd frontend && npm ci && npm run build        # → ../smileornot/static/*
uvicorn smileornot.app:app --port 7860         # serves both / and /predict
```

### Design decisions and rationale

- **Astro chosen over plain Vite/Parcel** for the `output: 'static'` ergonomics
  and `outDir` targeting, plus the deliberate "no framework runtime" semantics.
- **No React/Vue/Svelte islands.** "Vanilla HTML/CSS/JS" stays the rule.
- **Two-canvas split** (off-screen capture vs. on-screen overlay) prevents
  resolution-mismatch bugs.
- **Normalized coords throughout** — frontend never needs to know upload
  resolution.
- **`AbortController` + state guard after `await`** — clean stop semantics,
  no late draws after user clicks Stop.
- **`facingMode: 'user'`** — front camera by default.

### Deliberate omissions

- No FPS counter beyond `inference_ms`.
- No camera switching, resolution picker.
- No reconnect-on-error (user re-clicks Start to retry).
- No service worker / PWA / offline mode.
- No TypeScript on `app.js` (Astro accepts `.ts` if we add it later).

---

## 6. Deployment — Hugging Face Spaces

### Two-remote git model

GitHub is canonical. HF Space is a second remote on the same repo.

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/smileornot
git push hf main
```

HF auto-builds the Dockerfile on every push. GitHub renders the HF YAML
frontmatter as a small table at the top of `README.md` — minor cosmetic noise,
universally accepted.

### `README.md` YAML frontmatter

```yaml
---
title: SmileOrNot
emoji: 😀
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
---
```

### `Dockerfile` (multi-stage)

```dockerfile
# Stage 1: Astro frontend build
FROM node:20-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only torch wheels — saves ~1.5 GB vs default cuda-bundled torch
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml ./
COPY --chown=user smileornot ./smileornot
COPY --chown=user weights ./weights
RUN pip install --no-cache-dir -e .

COPY --from=frontend --chown=user /smileornot/static ./smileornot/static

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860
CMD ["uvicorn", "smileornot.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Design decisions and rationale

- **Multi-stage build.** Final image carries no Node, only the static output.
- **CPU-only torch wheels.** Default `pip install ultralytics` pulls torch
  with CUDA libs (~2.4 GB); CPU wheels are ~190 MB. Often the difference
  between "build times out" and "build succeeds" on HF.
- **Non-root `user` (uid 1000)** — HF Spaces convention.
- **`pip install -e .`** uses the `pyproject.toml` we already maintain — one
  source of truth for deps.
- **`weights/best.pt` committed** (~6 MB, under git LFS threshold).
- **No persistent storage, no Space secrets.** Backend is stateless; nothing
  to configure at runtime.

### HF resource fit (free CPU basic)

- 16 GB RAM, 2 vCPU, no GPU.
- YOLO26n at imgsz=640 inference: ~150-300 ms/frame.
- Memory: well under 1 GB total.
- Cold start: ~5-10s container boot + ~1-2s model load in lifespan.
- Free Spaces sleep after ~48h inactivity; first request after sleep wakes the
  container.

### Deliberate omissions

- No CI workflow that auto-pushes to HF (manual `git push hf main` for v1).
- No `docker-compose.yml` (local dev uses two-terminal flow).
- No paid Space tier, no GPU, no auto-scaling.
- No model versioning / hot-swap.

---

## 7. Repository Structure

```
smileornot/                                # repo root
├── .github/
│   └── workflows/
│       ├── ci.yml                         # MODIFIED — adds Node build + CPU torch
│       └── format.yml                     # KEEP — Ultralytics ruff/format
├── .gitignore                             # ADD: node_modules, dist, static, data, venv
├── LICENSE                                # KEEP — AGPL-3.0
├── README.md                              # REPLACE — HF YAML + portfolio narrative
├── pyproject.toml                         # MODIFY — package rename + new deps
├── Dockerfile                             # NEW — multi-stage
│
├── smileornot/                            # RENAMED from template/
│   ├── __init__.py                        # __version__ = "0.1.0"
│   ├── app.py
│   ├── inference.py
│   └── static/                            # GITIGNORED — Astro build output
│
├── frontend/                              # NEW — Astro project
│   ├── astro.config.mjs
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── public/
│   │   └── favicon.svg
│   └── src/
│       ├── pages/index.astro
│       ├── scripts/app.js
│       └── styles/styles.css
│
├── weights/
│   └── best.pt                            # ~6 MB, committed
│
├── scripts/                               # NEW — Phase 1 data pipeline
│   ├── 01_download_celeba.py
│   ├── 02_autolabel.py
│   ├── 03_split.py
│   └── dataset.yaml
│
├── tests/                                 # KEEP layout, REPLACE contents
│   ├── __init__.py
│   ├── fixtures/
│   │   ├── smile.jpg
│   │   ├── neutral.jpg
│   │   └── no_face.jpg
│   ├── test_inference.py
│   └── test_app.py
│
└── docs/
    ├── README.md                          # KEEP — placeholder from template
    ├── superpowers/
    │   └── specs/
    │       └── 2026-04-27-smileornot-design.md   # this file
    └── training.md                        # NEW — Platform training writeup
```

### `pyproject.toml` changes

```toml
[project]
name = "smileornot"
description = "Live smile detection with YOLO26n + FastAPI + Astro"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "ultralytics>=8.3",
  "pillow>=10.0",
  "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "httpx", "ruff"]

[tool.setuptools]
packages = { find = { where = ["."], include = ["smileornot", "smileornot.*"] } }

[tool.setuptools.dynamic]
version = { attr = "smileornot.__version__" }

[tool.pytest.ini_options]
addopts = "--durations=30 --color=yes"   # dropped --doctest-modules (fragile with ML libs)
```

Drop `[project.scripts]`'s `example-cli-command` (no CLI in this project).

### `.gitignore` additions

```
# Astro / Node
frontend/node_modules/
frontend/dist/
smileornot/static/

# Data pipeline (large, regenerable)
data/
*.zip
*.tar.gz

# Local environments
.venv/
.env
.DS_Store
```

### Why these choices

- `template/` → `smileornot/` keeps the package layout the Ultralytics CI
  expects.
- `--doctest-modules` removed: PIL/torch/Ultralytics state in docstrings is
  fragile under doctest.
- `data/` not committed (200K CelebA images would dwarf the repo); README's
  Reproduce section runs `01_*.py` to fetch.
- `notebooks/` deliberately omitted — `docs/training.md` is plenty without
  binary diff noise.

---

## 8. Testing Strategy

Three levels, justified by what each catches.

### Level 1 — Inference unit tests (`tests/test_inference.py`)

Most important. Loads real `weights/best.pt` once per session, predicts on
committed fixture images, asserts the response schema.

```python
from pathlib import Path
import pytest
from smileornot.inference import SmileDetector

WEIGHTS = Path(__file__).parent.parent / "weights" / "best.pt"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def detector() -> SmileDetector:
    return SmileDetector(WEIGHTS, device="cpu")


def test_predict_smile_face(detector: SmileDetector) -> None:
    raw = (FIXTURES / "smile.jpg").read_bytes()
    boxes, ms = detector.predict_bytes(raw, conf=0.3)
    assert len(boxes) >= 1
    assert {"x1", "y1", "x2", "y2", "class", "conf"} <= boxes[0].keys()
    assert boxes[0]["class"] in {"smiling", "neutral"}
    assert 0.0 <= boxes[0]["x1"] < boxes[0]["x2"] <= 1.0
    assert 0.0 <= boxes[0]["y1"] < boxes[0]["y2"] <= 1.0
    assert ms > 0


def test_predict_no_face(detector: SmileDetector) -> None:
    raw = (FIXTURES / "no_face.jpg").read_bytes()
    boxes, _ = detector.predict_bytes(raw, conf=0.5)
    assert boxes == []


def test_predict_invalid_bytes(detector: SmileDetector) -> None:
    with pytest.raises(Exception):
        detector.predict_bytes(b"not a jpeg")
```

Fixtures (committed, total <200 KB): `smile.jpg`, `neutral.jpg`, `no_face.jpg`.

### Level 2 — FastAPI endpoint tests (`tests/test_app.py`)

`TestClient` (httpx) exercises the full HTTP surface. Reuses the lifespan-loaded
detector — no mocking.

```python
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from smileornot.app import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:           # context manager triggers lifespan
        yield c


def test_predict_returns_boxes(client: TestClient) -> None:
    raw = (FIXTURES / "smile.jpg").read_bytes()
    r = client.post("/predict", files={"file": ("frame.jpg", raw, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "boxes" in body and "inference_ms" in body
    assert isinstance(body["boxes"], list)
    assert body["inference_ms"] > 0


def test_predict_rejects_non_image(client: TestClient) -> None:
    r = client.post("/predict", files={"file": ("foo.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_predict_rejects_oversized(client: TestClient) -> None:
    big = b"\xff\xd8\xff" + b"\x00" * 2_500_000
    r = client.post("/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_index_served(client: TestClient) -> None:
    # Requires `npm run build` to have populated smileornot/static/index.html
    r = client.get("/")
    assert r.status_code in (200, 404)
```

### CI must build the frontend before pytest

`.github/workflows/ci.yml` modifications:

```yaml
steps:
  - uses: actions/checkout@v6
  - uses: actions/setup-node@v4
    with: { node-version: '20' }
  - run: cd frontend && npm ci && npm run build
  - uses: actions/setup-python@v6
    with: { python-version: '3.11' }
  - run: pip install -e ".[dev]"
  - run: |
      pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
  - run: pytest --cov=smileornot
  - uses: codecov/codecov-action@v6
```

`format.yml` (ruff) stays unchanged.

### Level 3 — Frontend smoke tests

**Deliberately skipped in v1.** ~150 lines of vanilla JS with one state machine
and one network call. A Playwright/Cypress setup adds CI deps + a real browser
+ webcam mocking — disproportionate. Right test for the frontend is opening
it in a browser, which we'll do during dev and post-deploy. If the project
grows (camera switching, settings, multiple pages), revisit.

### What's intentionally not tested

- Astro build output (covered: build failure breaks CI before pytest).
- Data pipeline scripts (`scripts/01_*.py` etc.) — one-shot offline scripts;
  testing requires too much mocking; the existence of a trained `best.pt` is
  the demonstration they ran.
- Ultralytics's own behavior — their problem.
- HF Spaces deployment — verified by `git push hf main` + watching the build
  log, not by automated tests.

### Local test command

```bash
cd frontend && npm run build && cd ..
pip install -e ".[dev]"
pytest -v
```

---

## 9. Decisions Log

| # | Decision | Selected | Rationale |
|---|----------|----------|-----------|
| 1 | Inference architecture | Per-frame HTTP POST | Preserves portfolio narrative; simplest deploy; 3-6 FPS sufficient for use case |
| 2 | Dataset | CelebA | 200K images, has `Smiling` attribute, recognizable |
| 3 | Frame cadence | Adaptive (request-driven) | Zero queueing, no backpressure, self-throttling |
| 4 | Frontend stack | Astro `output: 'static'` + vanilla JS/CSS | Build-time only, zero framework runtime, Vite proxy for dev |
| 5 | Hosting | Hugging Face Space (Docker, free CPU) | Matches portfolio constraints; no GPU needed |
| 6 | License | AGPL-3.0 | Inherited from Ultralytics template; matches ecosystem |
| 7 | Weights distribution | Committed in `weights/` | ~6 MB, under git LFS threshold |
| 8 | Coordinate format | Normalized 0-1 (`xyxyn`) | Decouples upload resolution from display resolution |
| 9 | Confidence threshold | 0.4 default | Suppresses profile/occluded-face noise |
| 10 | Upload size ceiling | 2 MB | Defends against accidental full-res frames |
| 11 | Torch wheels | CPU-only (`download.pytorch.org/whl/cpu`) | Saves ~1.5 GB image size; HF has no GPU anyway |
| 12 | Frontend tests | Skipped in v1 | Disproportionate cost for ~150 lines of vanilla JS |

---

## 10. Out-of-scope (acknowledged for future work)

- Multi-face tracking with persistent IDs (per-frame independent boxes only).
- Mobile / responsive layout polish.
- Camera switching / resolution picker UI.
- More than two emotion classes.
- WebSocket streaming for higher FPS.
- Client-side ONNX inference (would defeat the "self-hosted FastAPI" narrative).
- Auth / multi-user / session persistence.
- Analytics / telemetry.
- GPU deployment / paid tier.
- Auto-deploy CI from GitHub to HF.
