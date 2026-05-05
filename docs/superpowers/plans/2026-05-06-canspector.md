# Canspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second YOLO model (4-class can-condition detector) and a mobile-first `/can` page sharing a GSAP-animated navbar with the existing smile demo, all inside the existing SmileOrNot project.

**Architecture:** Generalize the existing `SmileDetector` into `YoloDetector`. Load both smile and can detectors at FastAPI startup. New endpoint `/predict/can` mirrors `/predict`. New Astro page `/can` reuses extracted detector JS (live mode + photo upload). Shared `Base.astro` layout + `Navbar.astro` with Astro `<ClientRouter />` View Transitions plus GSAP animations.

**Tech Stack:** FastAPI, Ultralytics YOLO, Astro 5 (static output), vanilla JS, GSAP, pytest, Roboflow Python SDK.

**Spec:** `docs/superpowers/specs/2026-05-06-canspector-design.md`

---

## File Structure

**Backend**
- Modify: `smileornot/inference.py` — generalize to `YoloDetector`, keep `SmileDetector` as alias.
- Modify: `smileornot/app.py` — load both detectors, add `/predict/can`.
- Modify: `tests/test_app.py` — add `/predict/can` happy-path test.
- Modify: `tests/test_inference.py` — adapt to new class.

**Data scripts**
- Create: `scripts/can_01_download.py`
- Create: `scripts/can_02_remap.py`
- Create: `scripts/can_03_split.py`
- Create: `scripts/can_dataset.yaml`
- Modify: `pyproject.toml` — add `roboflow` dep.

**Frontend**
- Create: `frontend/src/scripts/detector.js` — extracted parameterized detector.
- Create: `frontend/src/scripts/can.js` — entry point for `/can` page.
- Modify: `frontend/src/scripts/app.js` — slim down to a thin entry that calls `detector.js`.
- Create: `frontend/src/layouts/Base.astro` — shared HTML shell + `<ClientRouter />`.
- Create: `frontend/src/components/Navbar.astro` — links + GSAP mount/swap.
- Create: `frontend/src/pages/can.astro` — Live + Upload modes.
- Modify: `frontend/src/pages/index.astro` — switch to `Base.astro`.
- Modify: `frontend/src/styles/styles.css` — navbar + `/can` styles.
- Modify: `frontend/package.json` — add `gsap` dep.

**Deploy**
- Modify: `Dockerfile` — bundle `weights/can_best.pt` if present (no behavior change required; current `COPY weights/` already includes it).

---

## Phase 1 — Backend refactor

### Task 1: Generalize `SmileDetector` into `YoloDetector`

**Files:**
- Modify: `smileornot/inference.py`
- Modify: `tests/test_inference.py`

- [ ] **Step 1: Read current inference test to know what API to preserve**

Run: `cat tests/test_inference.py`

- [ ] **Step 2: Write failing test for `YoloDetector` with explicit class names**

Add to `tests/test_inference.py`:

```python
from pathlib import Path

import pytest

from smileornot.inference import YoloDetector

WEIGHTS = Path(__file__).parent.parent / "weights" / "best.pt"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="weights/best.pt absent")
def test_yolo_detector_uses_supplied_class_names() -> None:
    det = YoloDetector(WEIGHTS, class_names=["alpha", "beta"])
    raw = (FIXTURES / "smile.jpg").read_bytes()
    boxes, ms = det.predict_bytes(raw)
    assert ms > 0
    for b in boxes:
        assert b["class"] in {"alpha", "beta"}
```

- [ ] **Step 3: Run the test, expect failure**

Run: `pytest tests/test_inference.py::test_yolo_detector_uses_supplied_class_names -v`
Expected: ImportError — `YoloDetector` not defined.

- [ ] **Step 4: Refactor `smileornot/inference.py`**

Replace file contents with:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YoloDetector — thin wrapper around Ultralytics YOLO for the SmileOrNot demos."""

from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


class YoloDetector:
    """Run object detection on JPEG/PNG bytes using a trained YOLO model.

    Args:
        weights_path: Path to the .pt weights file.
        class_names: Ordered list of class names; index = class id from the model.
            Overrides the names baked into the weights file so the API is
            independent of training-time class ordering.
        device: Torch device. Use ``"cpu"`` on Hugging Face Spaces free tier.
    """

    def __init__(
        self,
        weights_path: Path,
        class_names: list[str],
        device: str = "cpu",
    ) -> None:
        self.model = YOLO(str(weights_path))
        self.device = device
        self.class_names = class_names

    def predict_bytes(self, raw: bytes, conf: float = 0.4) -> tuple[list[dict], float]:
        """Run inference on encoded image bytes.

        Returns:
            (boxes, inference_ms) where each box is a dict with normalized
            xyxy coords, ``class`` (str), and ``conf`` (float).
        """
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        t0 = time.perf_counter()
        results = self.model.predict(img, conf=conf, device=self.device, verbose=False)
        ms = (time.perf_counter() - t0) * 1000
        r = results[0]
        boxes: list[dict] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxyn = r.boxes.xyxyn.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxyn, confs, clss):
                boxes.append(
                    {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "class": self.class_names[int(k)],
                        "conf": float(c),
                    }
                )
        return boxes, round(ms, 2)


class SmileDetector(YoloDetector):
    """Backwards-compat shim: smile/neutral detector with fixed class names."""

    def __init__(self, weights_path: Path, device: str = "cpu") -> None:
        super().__init__(weights_path, class_names=["smiling", "neutral"], device=device)
```

- [ ] **Step 5: Run all inference tests, expect pass**

Run: `pytest tests/test_inference.py -v`
Expected: PASS (or skipped if weights absent — both acceptable).

- [ ] **Step 6: Run full backend test suite to confirm no regressions**

Run: `pytest -v`
Expected: same pass count as before plus the new test.

- [ ] **Step 7: Commit**

```bash
git add smileornot/inference.py tests/test_inference.py
git commit -m "refactor(inference): generalize SmileDetector → YoloDetector"
```

---

### Task 2: Add `/predict/can` endpoint with optional weights

**Files:**
- Modify: `smileornot/app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write failing test for `/predict/can` 503-when-missing**

Add to `tests/test_app.py`:

```python
CAN_WEIGHTS = Path(__file__).parent.parent / "weights" / "can_best.pt"


def test_predict_can_returns_503_if_weights_missing(client: TestClient) -> None:
    if CAN_WEIGHTS.exists():
        pytest.skip("can weights present; this test only runs without them")
    raw = (FIXTURES / "smile.jpg").read_bytes()
    r = client.post("/predict/can", files={"file": ("frame.jpg", raw, "image/jpeg")})
    assert r.status_code == 503


@pytest.mark.skipif(not CAN_WEIGHTS.exists(), reason="weights/can_best.pt absent")
def test_predict_can_returns_boxes(client: TestClient) -> None:
    raw = (FIXTURES / "smile.jpg").read_bytes()  # any image; we only check the schema
    r = client.post("/predict/can", files={"file": ("frame.jpg", raw, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "boxes" in body and "inference_ms" in body
```

- [ ] **Step 2: Run the new tests, expect failure**

Run: `pytest tests/test_app.py::test_predict_can_returns_503_if_weights_missing -v`
Expected: 404 (route not defined yet).

- [ ] **Step 3: Update `smileornot/app.py`**

Replace contents with:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""FastAPI app: smile + can detectors, /predict + /predict/can, static assets."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from smileornot.inference import YoloDetector

ROOT = Path(__file__).parent.parent
SMILE_WEIGHTS = ROOT / "weights" / "best.pt"
CAN_WEIGHTS = ROOT / "weights" / "can_best.pt"
STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 2_000_000

CAN_CLASSES = ["intact_labeled", "intact_unlabeled", "damaged_labeled", "damaged_unlabeled"]

log = logging.getLogger("smileornot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.smile_detector = YoloDetector(
        SMILE_WEIGHTS, class_names=["smiling", "neutral"], device="cpu"
    )
    if CAN_WEIGHTS.exists():
        app.state.can_detector = YoloDetector(CAN_WEIGHTS, class_names=CAN_CLASSES, device="cpu")
    else:
        log.warning("weights/can_best.pt missing; /predict/can will return 503")
        app.state.can_detector = None
    yield


app = FastAPI(lifespan=lifespan, title="SmileOrNot", version="0.2.0")


async def _read_image(file: UploadFile) -> bytes:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Expected an image upload")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Frame too large")
    return raw


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    raw = await _read_image(file)
    boxes, ms = app.state.smile_detector.predict_bytes(raw)
    return JSONResponse({"boxes": boxes, "inference_ms": ms})


@app.post("/predict/can")
async def predict_can(file: UploadFile = File(...)) -> JSONResponse:
    if app.state.can_detector is None:
        raise HTTPException(status_code=503, detail="Can detector not loaded")
    raw = await _read_image(file)
    boxes, ms = app.state.can_detector.predict_bytes(raw)
    return JSONResponse({"boxes": boxes, "inference_ms": ms})


STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
```

- [ ] **Step 4: Run all backend tests, expect pass**

Run: `pytest -v`
Expected: all green; the can-503 test passes when weights absent, can-200 test skips.

- [ ] **Step 5: Commit**

```bash
git add smileornot/app.py tests/test_app.py
git commit -m "feat(api): add /predict/can endpoint, dual-model startup"
```

---

## Phase 2 — Data pipeline

### Task 3: Add `roboflow` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Locate the dependencies array**

Run: `grep -n "dependencies" pyproject.toml`

- [ ] **Step 2: Add `roboflow` to the runtime deps array**

Edit `pyproject.toml`. Inside `[project] dependencies = [ ... ]` (or the equivalent block), add the line `"roboflow>=1.1",`.

- [ ] **Step 3: Install the new dep into the local venv**

Run: `pip install -e ".[dev]"`
Expected: `roboflow` installs successfully.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add roboflow for can dataset download"
```

---

### Task 4: `scripts/can_01_download.py`

**Files:**
- Create: `scripts/can_01_download.py`

- [ ] **Step 1: Write the script**

```python
"""Download a Roboflow can-damage dataset in YOLOv8 format.

Edit the three constants below after picking a dataset on Roboflow Universe.
The script is idempotent: it checks for an existing download before re-fetching.

Outputs:
    data/can/raw/{train,valid,test}/{images,labels}/...
    data/can/raw/data.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

from roboflow import Roboflow

# TODO: replace with the chosen Roboflow workspace/project/version.
ROBOFLOW_WORKSPACE = "REPLACE_ME_WORKSPACE"
ROBOFLOW_PROJECT = "REPLACE_ME_PROJECT"
ROBOFLOW_VERSION = 1

DEST = Path(__file__).parent.parent / "data" / "can" / "raw"


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in your environment.")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if (DEST / "data.yaml").exists():
        print(f"Existing dataset at {DEST}; skipping download.")
        return
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    project.version(ROBOFLOW_VERSION).download("yolov8", location=str(DEST))
    print(f"Downloaded to {DEST}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity check — script imports cleanly**

Run: `python -c "import ast; ast.parse(open('scripts/can_01_download.py').read())"`
Expected: no output (parse OK).

- [ ] **Step 3: Commit**

```bash
git add scripts/can_01_download.py
git commit -m "feat(data): can_01_download — Roboflow dataset fetcher"
```

---

### Task 5: `scripts/can_02_remap.py`

**Files:**
- Create: `scripts/can_02_remap.py`

- [ ] **Step 1: Write the script**

```python
"""Remap a Roboflow can dataset's class ids to the 4 canspector classes.

Edit CLASS_MAP after inspecting data/can/raw/data.yaml. Keys are source class
names (as they appear in data.yaml), values are target class ids:

    0 = intact_labeled
    1 = intact_unlabeled
    2 = damaged_labeled
    3 = damaged_unlabeled

Images whose labels contain only unmapped classes are skipped.

Inputs:
    data/can/raw/{train,valid,test}/{images,labels}/...
    data/can/raw/data.yaml
Outputs:
    data/can/yolo/images/<basename>.jpg
    data/can/yolo/labels/<basename>.txt
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

RAW = Path(__file__).parent.parent / "data" / "can" / "raw"
DST = Path(__file__).parent.parent / "data" / "can" / "yolo"

# TODO: fill in after picking the Roboflow dataset and reading raw/data.yaml.
# Example shape — actual keys depend on the source dataset's class names.
CLASS_MAP: dict[str, int] = {
    # "can_intact_with_label": 0,
    # "can_intact_no_label": 1,
    # "can_damaged_with_label": 2,
    # "can_damaged_no_label": 3,
}


def load_source_names() -> list[str]:
    cfg = yaml.safe_load((RAW / "data.yaml").read_text())
    return cfg["names"]


def main() -> None:
    if not CLASS_MAP:
        raise SystemExit("CLASS_MAP is empty; edit scripts/can_02_remap.py first.")
    src_names = load_source_names()
    src_id_to_target: dict[int, int] = {}
    for i, name in enumerate(src_names):
        if name in CLASS_MAP:
            src_id_to_target[i] = CLASS_MAP[name]

    (DST / "images").mkdir(parents=True, exist_ok=True)
    (DST / "labels").mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    for split in ("train", "valid", "test"):
        labels_dir = RAW / split / "labels"
        images_dir = RAW / split / "images"
        if not labels_dir.exists():
            continue
        for lp in labels_dir.glob("*.txt"):
            new_lines: list[str] = []
            for line in lp.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                src_cls = int(parts[0])
                if src_cls not in src_id_to_target:
                    continue
                parts[0] = str(src_id_to_target[src_cls])
                new_lines.append(" ".join(parts))
            if not new_lines:
                dropped += 1
                continue
            stem = lp.stem
            ip = images_dir / f"{stem}.jpg"
            if not ip.exists():
                # try .png / .jpeg
                alt = next(images_dir.glob(f"{stem}.*"), None)
                if alt is None:
                    dropped += 1
                    continue
                ip = alt
            (DST / "labels" / f"{stem}.txt").write_text("\n".join(new_lines) + "\n")
            shutil.copy2(ip, DST / "images" / f"{stem}.jpg")
            kept += 1

    print(f"Kept {kept} images, dropped {dropped}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity check parse**

Run: `python -c "import ast; ast.parse(open('scripts/can_02_remap.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add scripts/can_02_remap.py
git commit -m "feat(data): can_02_remap — Roboflow → 4-class YOLO labels"
```

---

### Task 6: `scripts/can_03_split.py`

**Files:**
- Create: `scripts/can_03_split.py`

- [ ] **Step 1: Write the script (stratified 80/10/10)**

```python
"""Stratified 80/10/10 split for the canspector dataset (4 classes).

Reads:  data/can/yolo/images/*.jpg + data/can/yolo/labels/*.txt
Writes: data/can/yolo/{images,labels}/{train,val,test}/
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent / "data" / "can" / "yolo"
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
SEED = 42
NUM_CLASSES = 4


def primary_class(label_path: Path) -> int:
    """Use the first row's class as the stratification key."""
    return int(label_path.read_text().strip().splitlines()[0].split()[0])


def main() -> None:
    label_files = [p for p in (ROOT / "labels").glob("*.txt") if p.parent.name == "labels"]
    by_class: dict[int, list[Path]] = {i: [] for i in range(NUM_CLASSES)}
    for lp in label_files:
        c = primary_class(lp)
        if c in by_class:
            by_class[c].append(lp)

    rng = random.Random(SEED)
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for paths in by_class.values():
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        splits["train"].extend(paths[:n_train])
        splits["val"].extend(paths[n_train : n_train + n_val])
        splits["test"].extend(paths[n_train + n_val :])

    for split, paths in splits.items():
        (ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for lp in paths:
            stem = lp.stem
            ip = ROOT / "images" / f"{stem}.jpg"
            shutil.move(str(ip), str(ROOT / "images" / split / f"{stem}.jpg"))
            shutil.move(str(lp), str(ROOT / "labels" / split / f"{stem}.txt"))
        print(f"{split}: {len(paths)} samples")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity check parse**

Run: `python -c "import ast; ast.parse(open('scripts/can_03_split.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add scripts/can_03_split.py
git commit -m "feat(data): can_03_split — stratified 80/10/10"
```

---

### Task 7: `scripts/can_dataset.yaml`

**Files:**
- Create: `scripts/can_dataset.yaml`

- [ ] **Step 1: Write the YAML**

```yaml
# Ultralytics dataset config for canspector.
# Upload data/can/yolo/ + this file to Ultralytics Platform; train YOLO26n.
path: ../data/can/yolo
train: images/train
val: images/val
test: images/test

names:
  0: intact_labeled
  1: intact_unlabeled
  2: damaged_labeled
  3: damaged_unlabeled
```

- [ ] **Step 2: Commit**

```bash
git add scripts/can_dataset.yaml
git commit -m "feat(data): can_dataset.yaml — 4-class Ultralytics config"
```

---

## Phase 3 — Frontend

### Task 8: Add GSAP dependency

**Files:**
- Modify: `frontend/package.json`

- [ ] **Step 1: Install gsap**

Run: `cd frontend && npm install gsap@^3.12 && cd ..`
Expected: package.json updated with `gsap` under dependencies; lockfile updated.

- [ ] **Step 2: Verify install**

Run: `grep gsap frontend/package.json`
Expected: matches `"gsap": "^3.12..."`.

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/package-lock.json
git commit -m "deps(frontend): add gsap"
```

---

### Task 9: Extract `detector.js` (parameterized live + still detector)

**Files:**
- Create: `frontend/src/scripts/detector.js`
- Modify: `frontend/src/scripts/app.js`

- [ ] **Step 1: Create `detector.js`**

```js
// Parameterized detector controller. Used by the smile and can pages.
//
// Usage (live mode):
//   initLiveDetector({
//     endpoint: '/predict',
//     classColors: { smiling: '#22c55e', neutral: '#94a3b8' },
//     elements: { video, overlay, status, toggle },
//   });
//
// Usage (still mode):
//   detectStill({ endpoint: '/predict/can', classColors, image, overlay, status });

const STATE = { IDLE: 'idle', RUNNING: 'running', ERROR: 'error' };

export function initLiveDetector({ endpoint, classColors, elements }) {
  const { video, overlay, status, toggle } = elements;
  const ctx = overlay.getContext('2d');
  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');
  let state = STATE.IDLE;
  let stream = null;
  let abortController = null;

  async function start() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, facingMode: { ideal: 'environment' } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      sizeOverlay();
      state = STATE.RUNNING;
      toggle.textContent = 'Stop Camera';
      loop();
    } catch (e) {
      state = STATE.ERROR;
      status.textContent = e.name === 'NotAllowedError' ? 'Camera permission denied.' : 'Camera unavailable.';
    }
  }

  function stop() {
    state = STATE.IDLE;
    abortController?.abort();
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    toggle.textContent = 'Start Camera';
    status.textContent = 'Stopped.';
  }

  async function loop() {
    while (state === STATE.RUNNING) {
      try {
        const blob = await captureFrame();
        abortController = new AbortController();
        const fd = new FormData();
        fd.append('file', blob, 'frame.jpg');
        const r = await fetch(endpoint, { method: 'POST', body: fd, signal: abortController.signal });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const { boxes, inference_ms } = await r.json();
        if (state !== STATE.RUNNING) return;
        drawBoxes(ctx, overlay, boxes, classColors);
        renderStatus(status, boxes, inference_ms);
      } catch (e) {
        if (e.name === 'AbortError') return;
        state = STATE.ERROR;
        status.textContent = `Connection lost: ${e.message}`;
        return;
      }
    }
  }

  function captureFrame() {
    const targetW = 480;
    const targetH = Math.round((video.videoHeight * targetW) / video.videoWidth);
    captureCanvas.width = targetW;
    captureCanvas.height = targetH;
    captureCtx.drawImage(video, 0, 0, targetW, targetH);
    return new Promise((resolve) => captureCanvas.toBlob(resolve, 'image/jpeg', 0.7));
  }

  function sizeOverlay() {
    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
  }

  toggle.addEventListener('click', () => (state === STATE.RUNNING ? stop() : start()));
  window.addEventListener('resize', () => state === STATE.RUNNING && sizeOverlay());
}

export async function detectStill({ endpoint, classColors, file, overlay, image, status }) {
  const ctx = overlay.getContext('2d');
  const url = URL.createObjectURL(file);
  await new Promise((res) => {
    image.onload = res;
    image.src = url;
  });
  overlay.width = image.clientWidth;
  overlay.height = image.clientHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  status.textContent = 'Predicting…';
  const fd = new FormData();
  fd.append('file', file, file.name);
  const r = await fetch(endpoint, { method: 'POST', body: fd });
  if (!r.ok) {
    status.textContent = `Error: HTTP ${r.status}`;
    return;
  }
  const { boxes, inference_ms } = await r.json();
  drawBoxes(ctx, overlay, boxes, classColors);
  renderStatus(status, boxes, inference_ms);
}

function drawBoxes(ctx, overlay, boxes, classColors) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  for (const b of boxes) {
    const x = b.x1 * overlay.width;
    const y = b.y1 * overlay.height;
    const w = (b.x2 - b.x1) * overlay.width;
    const h = (b.y2 - b.y1) * overlay.height;
    const color = classColors[b.class] ?? '#ffffff';
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    const label = `${b.class} ${(b.conf * 100).toFixed(0)}%`;
    ctx.font = '14px system-ui';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 20, tw + 8, 20);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x + 4, y - 5);
  }
}

function renderStatus(statusEl, boxes, ms) {
  const counts = boxes.reduce((acc, b) => ((acc[b.class] = (acc[b.class] || 0) + 1), acc), {});
  const parts = Object.entries(counts).map(([k, v]) => `${k}: ${v}`);
  statusEl.textContent = `${parts.join(' · ') || 'no detections'} · ${ms.toFixed(0)} ms`;
}
```

- [ ] **Step 2: Replace `frontend/src/scripts/app.js` with thin entry**

```js
import { initLiveDetector } from './detector.js';

initLiveDetector({
  endpoint: '/predict',
  classColors: { smiling: '#22c55e', neutral: '#94a3b8' },
  elements: {
    video: document.getElementById('video'),
    overlay: document.getElementById('overlay'),
    status: document.getElementById('status'),
    toggle: document.getElementById('toggle'),
  },
});
```

- [ ] **Step 3: Build the frontend to confirm no syntax errors**

Run: `cd frontend && npm run build && cd ..`
Expected: build succeeds; output in `smileornot/static/`.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/scripts/detector.js frontend/src/scripts/app.js
git commit -m "refactor(frontend): extract reusable detector controller"
```

---

### Task 10: Shared `Base.astro` layout + `Navbar.astro` with GSAP

**Files:**
- Create: `frontend/src/layouts/Base.astro`
- Create: `frontend/src/components/Navbar.astro`
- Modify: `frontend/src/pages/index.astro`
- Modify: `frontend/src/styles/styles.css`

- [ ] **Step 1: Create `Base.astro`**

```astro
---
import { ClientRouter } from 'astro:transitions';
import Navbar from '../components/Navbar.astro';

interface Props {
  title: string;
}
const { title } = Astro.props;
---
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <title>{title}</title>
    <ClientRouter />
  </head>
  <body>
    <Navbar />
    <main>
      <slot />
    </main>

    <script>
      import { gsap } from 'gsap';
      function animateMain() {
        const el = document.querySelector('main');
        if (el) gsap.fromTo(el, { opacity: 0, y: 12 }, { opacity: 1, y: 0, duration: 0.3 });
      }
      animateMain();
      document.addEventListener('astro:after-swap', animateMain);
    </script>

    <style is:global>
      @import '../styles/styles.css';
    </style>
  </body>
</html>
```

- [ ] **Step 2: Create `Navbar.astro`**

```astro
---
const path = Astro.url.pathname;
const links = [
  { href: '/', label: 'Smile' },
  { href: '/can', label: 'Can' },
];
---
<nav class="nav" aria-label="Primary">
  <button class="nav-toggle" aria-controls="nav-list" aria-expanded="false">☰</button>
  <ul id="nav-list" class="nav-list">
    {links.map((l) => (
      <li>
        <a
          href={l.href}
          aria-current={path === l.href ? 'page' : undefined}
        >{l.label}</a>
      </li>
    ))}
  </ul>
</nav>

<script>
  import { gsap } from 'gsap';

  function animateNav() {
    gsap.fromTo(
      '.nav-list li',
      { y: -20, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.4, stagger: 0.08, ease: 'power2.out' },
    );
  }

  function wireToggle() {
    const btn = document.querySelector('.nav-toggle');
    const list = document.getElementById('nav-list');
    if (!btn || !list) return;
    btn.addEventListener('click', () => {
      const open = list.classList.toggle('open');
      btn.setAttribute('aria-expanded', String(open));
      gsap.fromTo(
        list,
        { height: open ? 0 : 'auto', opacity: open ? 0 : 1 },
        { height: open ? 'auto' : 0, opacity: open ? 1 : 0, duration: 0.25 },
      );
    });
  }

  function init() {
    animateNav();
    wireToggle();
  }
  init();
  document.addEventListener('astro:after-swap', init);
</script>
```

- [ ] **Step 3: Update `frontend/src/pages/index.astro` to use `Base.astro`**

```astro
---
import Base from '../layouts/Base.astro';
---
<Base title="SmileOrNot">
  <h1>SmileOrNot</h1>
  <p id="status">Click to start.</p>
  <div class="stage">
    <video id="video" playsinline autoplay muted></video>
    <canvas id="overlay"></canvas>
  </div>
  <button id="toggle">Start Camera</button>

  <script>
    import '../scripts/app.js';
  </script>
</Base>
```

- [ ] **Step 4: Append navbar styles to `frontend/src/styles/styles.css`**

Append:

```css
.nav {
  display: flex;
  align-items: center;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid #1f2937;
  background: #0b1020;
  position: sticky;
  top: 0;
  z-index: 10;
}
.nav-toggle {
  display: none;
  background: transparent;
  color: inherit;
  font-size: 1.25rem;
  border: 0;
  padding: 0.25rem 0.5rem;
}
.nav-list {
  display: flex;
  gap: 1rem;
  list-style: none;
  margin: 0;
  padding: 0;
}
.nav-list a {
  color: #cbd5e1;
  text-decoration: none;
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
}
.nav-list a[aria-current='page'] {
  color: #ffffff;
  background: #1e293b;
}

@media (max-width: 600px) {
  .nav { flex-direction: column; align-items: stretch; }
  .nav-toggle { display: block; align-self: flex-end; }
  .nav-list { flex-direction: column; overflow: hidden; height: 0; opacity: 0; }
  .nav-list.open { height: auto; opacity: 1; }
}
```

- [ ] **Step 5: Build to verify**

Run: `cd frontend && npm run build && cd ..`
Expected: build succeeds; smile demo still works in the built output.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/layouts/Base.astro frontend/src/components/Navbar.astro \
        frontend/src/pages/index.astro frontend/src/styles/styles.css
git commit -m "feat(frontend): shared Base layout + GSAP navbar"
```

---

### Task 11: `/can` page with Live + Upload modes

**Files:**
- Create: `frontend/src/pages/can.astro`
- Create: `frontend/src/scripts/can.js`
- Modify: `frontend/src/styles/styles.css`

- [ ] **Step 1: Create `can.astro`**

```astro
---
import Base from '../layouts/Base.astro';
---
<Base title="Canspector">
  <h1>Can condition</h1>

  <div class="mode-toggle" role="tablist">
    <button role="tab" id="mode-live" aria-selected="true">Live</button>
    <button role="tab" id="mode-upload" aria-selected="false">Upload</button>
  </div>

  <p id="status">Pick a mode.</p>

  <section id="live-pane" class="stage">
    <video id="video" playsinline autoplay muted></video>
    <canvas id="overlay-live" class="overlay"></canvas>
    <button id="toggle">Start Camera</button>
  </section>

  <section id="upload-pane" hidden>
    <label class="upload-label">
      Photo
      <input id="file-input" type="file" accept="image/*" capture="environment" />
    </label>
    <div class="stage">
      <img id="upload-image" alt="" />
      <canvas id="overlay-upload" class="overlay"></canvas>
    </div>
  </section>

  <script>
    import '../scripts/can.js';
  </script>
</Base>
```

- [ ] **Step 2: Create `can.js`**

```js
import { initLiveDetector, detectStill } from './detector.js';

const CAN_COLORS = {
  intact_labeled: '#22c55e',   // green
  intact_unlabeled: '#84cc16', // lime
  damaged_labeled: '#f59e0b',  // amber
  damaged_unlabeled: '#ef4444',// red
};

const status = document.getElementById('status');
const livePane = document.getElementById('live-pane');
const uploadPane = document.getElementById('upload-pane');
const liveBtn = document.getElementById('mode-live');
const uploadBtn = document.getElementById('mode-upload');

initLiveDetector({
  endpoint: '/predict/can',
  classColors: CAN_COLORS,
  elements: {
    video: document.getElementById('video'),
    overlay: document.getElementById('overlay-live'),
    status,
    toggle: document.getElementById('toggle'),
  },
});

const fileInput = document.getElementById('file-input');
const image = document.getElementById('upload-image');
const overlayUpload = document.getElementById('overlay-upload');

fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  await detectStill({
    endpoint: '/predict/can',
    classColors: CAN_COLORS,
    file,
    overlay: overlayUpload,
    image,
    status,
  });
});

function setMode(mode) {
  const live = mode === 'live';
  livePane.hidden = !live;
  uploadPane.hidden = live;
  liveBtn.setAttribute('aria-selected', String(live));
  uploadBtn.setAttribute('aria-selected', String(!live));
}

liveBtn.addEventListener('click', () => setMode('live'));
uploadBtn.addEventListener('click', () => setMode('upload'));
```

- [ ] **Step 3: Append upload-pane styles to `styles.css`**

Append:

```css
.mode-toggle {
  display: inline-flex;
  border: 1px solid #1f2937;
  border-radius: 0.5rem;
  overflow: hidden;
  margin: 0.5rem 0 1rem;
}
.mode-toggle button {
  background: transparent;
  color: #cbd5e1;
  border: 0;
  padding: 0.5rem 1rem;
  cursor: pointer;
}
.mode-toggle button[aria-selected='true'] {
  background: #1e293b;
  color: #fff;
}
.upload-label {
  display: block;
  margin: 0.5rem 0 1rem;
}
#upload-image { width: 100%; display: block; }
.stage { position: relative; }
.overlay { position: absolute; inset: 0; pointer-events: none; }
```

- [ ] **Step 4: Build and verify**

Run: `cd frontend && npm run build && cd ..`
Expected: build succeeds; `smileornot/static/can/index.html` exists.

Run: `ls smileornot/static/can/`
Expected: `index.html` listed.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/can.astro frontend/src/scripts/can.js frontend/src/styles/styles.css
git commit -m "feat(frontend): /can page with Live + Upload modes"
```

---

## Phase 4 — End-to-end verification

### Task 12: Run full test suite + manual smoke

**Files:** none

- [ ] **Step 1: Run pytest**

Run: `pytest -v`
Expected: all green. `/predict/can` 503 test passes if `weights/can_best.pt` is absent; the 200 test skips. If `can_best.pt` is present, both pass.

- [ ] **Step 2: Run Astro build**

Run: `cd frontend && npm run build && cd ..`
Expected: success.

- [ ] **Step 3: Start the backend and load both pages**

Run: `uvicorn smileornot.app:app --port 8000`
Then in a browser open:
- `http://localhost:8000/` — smile demo, navbar visible, GSAP intro plays.
- `http://localhost:8000/can` — Can page, mode toggle, navbar active state on `/can`. If `can_best.pt` is missing, expect a 503 status message when you start the camera or upload — that's intentional.

Stop the server with Ctrl-C.

- [ ] **Step 4: Final commit if any tweaks**

If smoke test required no changes, no commit. Otherwise stage and commit fixes.

---

## Self-Review Notes (post-write)

- **Spec coverage:** all spec sections covered — class list (Task 2), data scripts (Tasks 4–7), backend (Tasks 1–2), frontend pages + navbar + GSAP (Tasks 8–11), tests (Tasks 1–2), deploy (no change required, noted in Phase 4).
- **Placeholders:** the only `TODO` items are in `can_01_download.py` and `can_02_remap.py` for the Roboflow slug + `CLASS_MAP`. These are explicit user-edit points the spec calls out and cannot be filled in until a Roboflow dataset is chosen — they are intentional, not plan placeholders.
- **Type consistency:** `YoloDetector(weights_path, class_names, device)` signature is consistent across Tasks 1, 2. `initLiveDetector({ endpoint, classColors, elements })` and `detectStill({ endpoint, classColors, file, overlay, image, status })` signatures match between `detector.js` (Task 9), `app.js` (Task 9), and `can.js` (Task 11).
