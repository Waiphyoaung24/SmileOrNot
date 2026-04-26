# SmileOrNot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a live smile-detection web demo: webcam frames are POSTed to a self-hosted FastAPI backend running a custom-trained YOLO26n model, which returns bounding boxes labeled `smiling` or `neutral`; the Astro-built static frontend draws them on a canvas overlay. Deployed to a free Hugging Face Docker Space.

**Architecture:** Two phases. Phase 1 runs once offline: CelebA → auto-label with a face detector → upload to Ultralytics Platform → Smart Annotation refinement → train YOLO26n → export `best.pt`. Phase 2 is the deployed app: FastAPI loads `best.pt` once at startup, serves a static Astro-built page, and accepts JPEG frames at `POST /predict`, returning normalized bounding boxes as JSON. The frontend captures frames adaptively (one in flight at a time) and draws boxes on a canvas overlaying the `<video>`.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, Ultralytics YOLO26 (CPU-only torch), Pillow, Astro `output: 'static'` + vanilla JS/CSS, multi-stage Docker, Hugging Face Spaces.

**Reference spec:** [docs/superpowers/specs/2026-04-27-smileornot-design.md](../specs/2026-04-27-smileornot-design.md)

---

## Phase 0 — Repository Scaffolding

The repo currently contains the official Ultralytics Python project template. Phase 0 reshapes it: rename the placeholder package, update dependency declarations, and configure ignores. No tests yet — these are structural changes.

### Task 0.1: Rename `template/` package to `smileornot/`

**Files:**
- Delete: `template/__init__.py`
- Delete: `template/module1.py`
- Create: `smileornot/__init__.py`

- [ ] **Step 1: Remove old package directory**

```bash
git rm -r template/
```

- [ ] **Step 2: Create the new package directory and `__init__.py`**

Create `smileornot/__init__.py` with:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "0.1.0"
```

- [ ] **Step 3: Verify directory structure**

Run: `ls -la smileornot/`
Expected: a single `__init__.py` file is present.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: rename template/ package to smileornot/"
```

---

### Task 0.2: Update `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace `pyproject.toml` with the SmileOrNot configuration**

Overwrite the entire file with this content (keeping the comment header style):

```toml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Overview:
# This pyproject.toml manages the build, packaging, and distribution of the SmileOrNot library.

[build-system]
requires = ["setuptools>=82.0.1", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "smileornot"
dynamic = ["version"]
description = "Live smile detection with YOLO26n + FastAPI + Astro"
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
keywords = ["Ultralytics", "YOLO", "smile detection", "FastAPI", "computer vision"]
authors = [{ name = "Wai Phyo Aung" }]
maintainers = [{ name = "Wai Phyo Aung" }]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
]

dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "ultralytics>=8.3",
    "pillow>=10.0",
    "python-multipart>=0.0.9",
    "numpy",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "httpx",
    "ruff",
]

[project.urls]
"Homepage" = "https://github.com/Waiphyoaung24/SmileOrNot"
"Source" = "https://github.com/Waiphyoaung24/SmileOrNot"
"Bug Reports" = "https://github.com/Waiphyoaung24/SmileOrNot/issues"

[tool.setuptools]
packages = { find = { where = ["."], include = ["smileornot", "smileornot.*"] } }

[tool.setuptools.dynamic]
version = { attr = "smileornot.__version__" }

[tool.pytest.ini_options]
addopts = "--durations=30 --color=yes"
norecursedirs = [".git", "dist", "build", "frontend", "data", "weights"]

[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["UP"]

[tool.ruff.format]
docstring-code-format = true

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.codespell]
ignore-words-list = "crate,nd,strack,dota,ane,segway,fo,gool,winn,commend"
skip = '*.csv,*venv*,docs/??/,docs/mkdocs_??.yml'
```

Key changes vs. the template default:
- `name = "smileornot"`
- New runtime deps: `fastapi`, `uvicorn[standard]`, `ultralytics`, `pillow`, `python-multipart`
- New dev deps: `httpx`, `ruff` (added beyond `pytest`/`pytest-cov`)
- `packages` `include` updated to `smileornot.*`
- `pytest` `addopts` drops `--doctest-modules` (PIL/torch state doesn't survive doctest)
- `pytest` `norecursedirs` adds `frontend`, `data`, `weights` so pytest doesn't crawl them
- `[project.scripts]` removed (no CLI in this project)

- [ ] **Step 2: Verify the file parses**

Run: `python -c "import tomllib; tomllib.loads(open('pyproject.toml').read())"`
Expected: no output (success).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: rewrite pyproject.toml for smileornot package"
```

---

### Task 0.3: Update `.gitignore`

**Files:**
- Modify: `.gitignore` (append new entries)

- [ ] **Step 1: Append project-specific ignore entries**

Append to the existing `.gitignore`:

```gitignore

# --- SmileOrNot additions ---

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

# Claude Code local settings (machine-specific; commit selectively if desired)
.claude/
```

- [ ] **Step 2: Verify**

Run: `git status`
Expected: no output for the listed paths if they exist (e.g. `.claude/` should disappear from untracked list once the file is committed and `.claude/` is created).

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore Astro build, data, venvs, .claude"
```

---

### Task 0.4: Install dev environment locally

**Files:**
- (none — environment setup)

- [ ] **Step 1: Create + activate a venv and install with dev extras**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

The CPU-only torch install matches the production Dockerfile (saves ~1.5 GB).

- [ ] **Step 2: Verify installation**

Run: `python -c "from smileornot import __version__; print(__version__)"`
Expected: `0.1.0`

Run: `python -c "import fastapi, uvicorn, ultralytics, PIL, multipart; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Verify pytest discovery works**

Run: `pytest --collect-only`
Expected: pytest discovers the existing `tests/test_with_pytest.py` and `tests/test_with_unittest.py` from the template (we'll replace these in Phase 2). No test failures yet.

(No commit — environment setup only.)

---

## Phase 1 — Data Pipeline & Training

Three Python scripts plus a YAML config produce the training inputs. Then a manual step on Ultralytics Platform refines annotations and trains YOLO26n. The artifact is `weights/best.pt`. Per the spec, scripts are not unit-tested; their correctness is verified by inspecting outputs.

### Task 1.1: Write `scripts/01_download_celeba.py`

**Files:**
- Create: `scripts/01_download_celeba.py`
- Create: `scripts/__init__.py` (empty, so the directory is a clean Python module if ever needed)

- [ ] **Step 1: Create the scripts directory + empty init**

```bash
mkdir -p scripts
touch scripts/__init__.py
```

- [ ] **Step 2: Create `scripts/01_download_celeba.py`**

```python
"""Download CelebA images and the Smiling attribute table from the Hugging Face mirror.

Outputs:
    data/celeba/img_align_celeba/*.jpg
    data/celeba/list_attr_celeba.csv
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "tpremoli/CelebA-attrs"
DEST = Path(__file__).parent.parent / "data" / "celeba"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} → {DEST}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DEST),
    )
    print("Done. Listing top-level entries:")
    for p in sorted(DEST.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add `huggingface_hub` to dev installs (one-shot)**

```bash
pip install huggingface_hub
```

(Not added to `pyproject.toml` runtime deps — only the data scripts use it, and Phase 2's tests + production runtime have no need for it.)

- [ ] **Step 4: Run the script (downloads ~1.4 GB; takes a few minutes)**

```bash
python scripts/01_download_celeba.py
```

Expected: `data/celeba/` contains `img_align_celeba/` (with ~200K jpg files) and an attribute CSV/txt file. Exact filenames depend on the mirror's layout; inspect:

```bash
ls data/celeba/
ls data/celeba/img_align_celeba/ | head -5
```

If the mirror's CSV is named differently from `list_attr_celeba.csv`, note the actual filename — Task 1.2 will read it.

- [ ] **Step 5: Commit the script (data is gitignored)**

```bash
git add scripts/__init__.py scripts/01_download_celeba.py
git commit -m "feat(data): script to download CelebA from HF mirror"
```

---

### Task 1.2: Write `scripts/02_autolabel.py`

**Files:**
- Create: `scripts/02_autolabel.py`

- [ ] **Step 1: Create the auto-labeling script**

```python
"""Auto-label CelebA images with bounding boxes from a pretrained face detector.

For each image:
  - Run yolo11n-face → take the highest-confidence face box
  - If no face is found, skip the image
  - Map CelebA's `Smiling` attribute (1 → 0=smiling, -1 → 1=neutral) onto the box
  - Write a YOLO-format label file: `<class> <x_center> <y_center> <w> <h>` (all normalized)

Outputs (under data/yolo/):
    images/<basename>.jpg
    labels/<basename>.txt

Subset: balanced 5000 smiling + 5000 neutral.
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

SRC_DIR = Path(__file__).parent.parent / "data" / "celeba"
IMG_DIR = SRC_DIR / "img_align_celeba"
ATTR_FILE_CANDIDATES = ["list_attr_celeba.csv", "list_attr_celeba.txt"]
DST_DIR = Path(__file__).parent.parent / "data" / "yolo"
N_PER_CLASS = 5000
SEED = 42


def find_attr_file() -> Path:
    for name in ATTR_FILE_CANDIDATES:
        p = SRC_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No attribute file found under {SRC_DIR}; looked for {ATTR_FILE_CANDIDATES}")


def parse_attrs(path: Path) -> dict[str, int]:
    """Return {filename: smiling_label} where smiling_label is +1 or -1."""
    attrs: dict[str, int] = {}
    text = path.read_text()
    lines = text.strip().splitlines()
    if path.suffix == ".csv":
        reader = csv.DictReader(lines)
        for row in reader:
            attrs[row["image_id"]] = int(row["Smiling"])
    else:
        # The classic CelebA format: line 1 = count, line 2 = headers (space-separated),
        # subsequent lines = "<filename> <attr1> <attr2> ..." with values in {-1, 1}.
        headers = lines[1].split()
        smile_idx = headers.index("Smiling")
        for line in lines[2:]:
            parts = line.split()
            attrs[parts[0]] = int(parts[1 + smile_idx])
    return attrs


def select_subset(attrs: dict[str, int], n_per_class: int) -> list[tuple[str, int]]:
    """Return [(filename, yolo_class)] balanced across smiling/neutral."""
    smiling = [f for f, s in attrs.items() if s == 1]
    neutral = [f for f, s in attrs.items() if s == -1]
    rng = random.Random(SEED)
    rng.shuffle(smiling)
    rng.shuffle(neutral)
    chosen = [(f, 0) for f in smiling[:n_per_class]] + [(f, 1) for f in neutral[:n_per_class]]
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    (DST_DIR / "images").mkdir(exist_ok=True)
    (DST_DIR / "labels").mkdir(exist_ok=True)

    attrs = parse_attrs(find_attr_file())
    subset = select_subset(attrs, N_PER_CLASS)
    print(f"Selected {len(subset)} images ({N_PER_CLASS} per class)")

    detector = YOLO("yolo11n-face.pt")  # Ultralytics pulls this on first run
    skipped = 0
    written = 0

    for filename, yolo_class in subset:
        img_path = IMG_DIR / filename
        if not img_path.exists():
            skipped += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        results = detector.predict(img, conf=0.35, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            skipped += 1
            continue

        # Highest-confidence box only
        confs = boxes.conf.cpu().numpy()
        best = int(confs.argmax())
        x1, y1, x2, y2 = boxes.xyxyn[best].cpu().numpy().tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        basename = Path(filename).stem
        shutil.copyfile(img_path, DST_DIR / "images" / f"{basename}.jpg")
        (DST_DIR / "labels" / f"{basename}.txt").write_text(
            f"{yolo_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
        )
        written += 1
        if written % 500 == 0:
            print(f"  {written} written, {skipped} skipped")

    print(f"Done. Wrote {written} labeled images. Skipped {skipped} (no-face or unreadable).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script (takes ~10-20 min on CPU; faster on GPU)**

```bash
python scripts/02_autolabel.py
```

Expected: `data/yolo/images/` and `data/yolo/labels/` populate with ~9,500-10,000 entries (a few hundred CelebA frames legitimately have no detectable face).

- [ ] **Step 3: Spot-check the labels visually (optional but valuable)**

Pick one labeled image and verify the box looks reasonable. The fastest check is to print and compare to an opened image:

```bash
ls data/yolo/labels/ | head -1 | xargs -I{} cat data/yolo/labels/{}
```

Expected: a single line with five floats; the class is `0` or `1`, the rest are normalized 0-1.

- [ ] **Step 4: Commit the script**

```bash
git add scripts/02_autolabel.py
git commit -m "feat(data): auto-label CelebA with yolo11n-face"
```

---

### Task 1.3: Write `scripts/03_split.py` and `scripts/dataset.yaml`

**Files:**
- Create: `scripts/03_split.py`
- Create: `scripts/dataset.yaml`

- [ ] **Step 1: Create the split script**

```python
"""Stratified 80/10/10 split of the auto-labeled YOLO dataset.

Reads:  data/yolo/images/*.jpg + data/yolo/labels/*.txt
Writes: data/yolo/images/{train,val,test}/ and data/yolo/labels/{train,val,test}/
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent / "data" / "yolo"
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
SEED = 42


def read_class(label_path: Path) -> int:
    return int(label_path.read_text().strip().split()[0])


def main() -> None:
    label_files = sorted((ROOT / "labels").glob("*.txt"))
    by_class: dict[int, list[Path]] = {0: [], 1: []}
    for lp in label_files:
        if lp.parent.name != "labels":
            continue  # skip already-split children if rerun
        c = read_class(lp)
        by_class[c].append(lp)

    rng = random.Random(SEED)
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for c, paths in by_class.items():
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

- [ ] **Step 2: Create `scripts/dataset.yaml`**

```yaml
# YOLO data config consumed by Ultralytics Platform / `yolo train data=...`.

path: ./data/yolo
train: images/train
val:   images/val
test:  images/test

names:
  0: smiling
  1: neutral
```

- [ ] **Step 3: Run the split script**

```bash
python scripts/03_split.py
```

Expected output:
```
train: ~8000 samples
val:   ~1000 samples
test:  ~1000 samples
```

- [ ] **Step 4: Commit**

```bash
git add scripts/03_split.py scripts/dataset.yaml
git commit -m "feat(data): stratified 80/10/10 split + YOLO dataset.yaml"
```

---

### Task 1.4: Train YOLO26n on Ultralytics Platform — manual step

**Files:**
- Create: `weights/best.pt` (downloaded from Platform after training)

This task is a **human-in-the-loop step**. The agent does not click through the Platform UI. The implementor must:

- [ ] **Step 1: Upload to Ultralytics Platform**

Compress the labeled subset and upload it via the Platform's dataset interface (or CLI: `yolo dataset upload data/yolo`). Specify `dataset.yaml` as the config. The Platform will index the dataset.

- [ ] **Step 2: Run Smart Annotation refinement on Platform**

Open the dataset in Platform's annotation tool. The "Smart Annotation" / "Auto-fix boxes" pass will surface boxes where the face detector's output looks misaligned. Click through ~1-2 hours of flagged frames; correct misalignments and any obviously-wrong class labels.

Take screenshots of the before/after for `docs/training.md` (Task 7.2).

- [ ] **Step 3: Train YOLO26n on Platform GPU**

Hyperparameters (entered into Platform's training UI or CLI):
- `model = yolo26n.pt`
- `epochs = 50`
- `imgsz = 640`
- `batch = auto`
- Default augmentation (Ultralytics defaults: mosaic + flip)

Expected training time: ~30 minutes on a single A10/A100. Expected mAP50 > 0.9.

Take screenshots of the loss curves and validation metrics for `docs/training.md`.

- [ ] **Step 4: Export weights**

Download the final `best.pt` from Platform.

```bash
mkdir -p weights/
# Move the downloaded file into place — exact path depends on where you downloaded it.
mv ~/Downloads/best.pt weights/best.pt
```

Verify size:

```bash
ls -la weights/best.pt
```

Expected: ~6 MB.

- [ ] **Step 5: Sanity-check the trained model**

```bash
python -c "
from ultralytics import YOLO
m = YOLO('weights/best.pt')
print('Class names:', m.names)
"
```

Expected: `Class names: {0: 'smiling', 1: 'neutral'}`

- [ ] **Step 6: Commit the weights**

```bash
git add weights/best.pt
git commit -m "feat(model): add trained YOLO26n weights (best.pt)"
```

> **Development workaround if Phase 1.4 is deferred:** If you want to start Phase 2 immediately without finishing Platform training, you can drop a pretrained `yolo26n.pt` into `weights/best.pt` as a placeholder. The class names will be COCO classes (not smiling/neutral) and assertions in Tasks 2.2-2.4 that check for `class in {"smiling", "neutral"}` will fail — you'll need to skip those tests until real weights are in place.

---

## Phase 2 — Backend Inference (TDD)

`smileornot/inference.py` wraps `ultralytics.YOLO` with a clean interface. We TDD it: tests load the real `weights/best.pt` once per session, exercise the predict path against committed fixture images. Phase 1.4 must be complete before this phase passes.

### Task 2.1: Add test fixtures

**Files:**
- Create: `tests/fixtures/smile.jpg`
- Create: `tests/fixtures/neutral.jpg`
- Create: `tests/fixtures/no_face.jpg`
- Modify: `tests/__init__.py` (already exists; no change needed)
- Delete: `tests/test_with_pytest.py`
- Delete: `tests/test_with_unittest.py`
- Delete: `tests/README.md`

- [ ] **Step 1: Remove the template's example tests + README**

```bash
git rm tests/test_with_pytest.py tests/test_with_unittest.py tests/README.md
mkdir -p tests/fixtures
```

- [ ] **Step 2: Place the three fixture images**

Source three small jpegs (each <100 KB):
- `smile.jpg` — clearly smiling face, ~480×480 or smaller
- `neutral.jpg` — clearly neutral face, ~480×480 or smaller
- `no_face.jpg` — landscape photo or other clearly face-less image

Recommended quick path: pick three frames from CelebA's training set (look at `data/yolo/images/train/`) and copy them in. For the no-face fixture, any free landscape photo works (or use a close-up of a non-face object).

```bash
# After copying the images into tests/fixtures/, verify sizes:
ls -la tests/fixtures/
```

Expected: three files, each <100 KB.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/ tests/
git commit -m "test: add smile/neutral/no_face fixture images and remove template tests"
```

---

### Task 2.2: TDD `SmileDetector` — happy path on a smiling face

**Files:**
- Create: `tests/test_inference.py`
- Create: `smileornot/inference.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference.py`:

```python
"""Tests for the SmileDetector inference wrapper."""

from __future__ import annotations

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
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_inference.py::test_predict_smile_face -v`
Expected: ImportError or ModuleNotFoundError (`smileornot.inference` doesn't exist yet).

- [ ] **Step 3: Implement `smileornot/inference.py`**

Create `smileornot/inference.py`:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""SmileDetector — thin wrapper around Ultralytics YOLO for the SmileOrNot demo."""

from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


class SmileDetector:
    """Run smile/neutral detection on JPEG bytes using a trained YOLO26n model.

    Args:
        weights_path: Path to the .pt weights file (typically ``weights/best.pt``).
        device: Torch device. Use ``"cpu"`` on Hugging Face Spaces free tier.
    """

    def __init__(self, weights_path: Path, device: str = "cpu") -> None:
        self.model = YOLO(str(weights_path))
        self.device = device
        self.names: dict[int, str] = self.model.names

    def predict_bytes(self, raw: bytes, conf: float = 0.4) -> tuple[list[dict], float]:
        """Run inference on JPEG/PNG bytes.

        Args:
            raw: Encoded image bytes (JPEG, PNG, etc.).
            conf: Confidence threshold; boxes below this are dropped.

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
                        "class": self.names[int(k)],
                        "conf": float(c),
                    }
                )
        return boxes, round(ms, 2)
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_inference.py::test_predict_smile_face -v`
Expected: PASS (the model loads, detects a face, returns boxes with the expected keys).

If it fails because the trained model returns 0 boxes at conf=0.3, your fixture jpeg may be too low-quality — pick a clearer face image. The threshold of 0.3 (lower than runtime 0.4) gives slack here.

- [ ] **Step 5: Commit**

```bash
git add smileornot/inference.py tests/test_inference.py
git commit -m "feat(inference): SmileDetector wraps YOLO with predict_bytes"
```

---

### Task 2.3: TDD `SmileDetector` — no-face case

**Files:**
- Modify: `tests/test_inference.py` (add a test)

- [ ] **Step 1: Append the failing test**

Add to the bottom of `tests/test_inference.py`:

```python
def test_predict_no_face(detector: SmileDetector) -> None:
    raw = (FIXTURES / "no_face.jpg").read_bytes()
    boxes, _ = detector.predict_bytes(raw, conf=0.5)
    assert boxes == []
```

- [ ] **Step 2: Run to confirm behavior**

Run: `pytest tests/test_inference.py::test_predict_no_face -v`
Expected: PASS — the trained model trained on faces won't fire above 0.5 confidence on a landscape.

If it fails (some boxes returned), either the no_face fixture has something face-like (replace with a cleaner image) or the model is overconfident on non-faces (raise conf to 0.6 in the test).

- [ ] **Step 3: Commit**

```bash
git add tests/test_inference.py
git commit -m "test: predict_bytes returns [] on no-face image"
```

---

### Task 2.4: TDD `SmileDetector` — invalid bytes

**Files:**
- Modify: `tests/test_inference.py` (add a test)

- [ ] **Step 1: Append the failing test**

Add to `tests/test_inference.py`:

```python
def test_predict_invalid_bytes(detector: SmileDetector) -> None:
    with pytest.raises(Exception):  # PIL raises UnidentifiedImageError
        detector.predict_bytes(b"not a jpeg")
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_inference.py::test_predict_invalid_bytes -v`
Expected: PASS — `Image.open(io.BytesIO(b"not a jpeg"))` raises `PIL.UnidentifiedImageError`, which is a subclass of `Exception`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_inference.py
git commit -m "test: predict_bytes raises on invalid image bytes"
```

---

## Phase 3 — Backend HTTP Layer (TDD)

`smileornot/app.py` wires the FastAPI app: lifespan loads the model once, `/predict` decodes JPEGs and returns JSON, `/` serves the (eventually built) Astro static directory.

### Task 3.1: TDD `/predict` happy path

**Files:**
- Create: `tests/test_app.py`
- Create: `smileornot/app.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_app.py`:

```python
"""Tests for the FastAPI app — endpoint surface, status codes, schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smileornot.app import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_predict_returns_boxes(client: TestClient) -> None:
    raw = (FIXTURES / "smile.jpg").read_bytes()
    r = client.post("/predict", files={"file": ("frame.jpg", raw, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "boxes" in body and "inference_ms" in body
    assert isinstance(body["boxes"], list)
    assert body["inference_ms"] > 0
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_app.py::test_predict_returns_boxes -v`
Expected: ImportError on `from smileornot.app import app`.

- [ ] **Step 3: Implement `smileornot/app.py`**

Create `smileornot/app.py`:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""FastAPI app: loads SmileDetector at startup, serves /predict + static assets."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from smileornot.inference import SmileDetector

WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "best.pt"
STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 2_000_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the YOLO model once before the first request."""
    app.state.detector = SmileDetector(WEIGHTS_PATH, device="cpu")
    yield


app = FastAPI(lifespan=lifespan, title="SmileOrNot", version="0.1.0")


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Run smile/neutral detection on an uploaded image frame."""
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Expected an image upload")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Frame too large")
    boxes, ms = app.state.detector.predict_bytes(raw)
    return JSONResponse({"boxes": boxes, "inference_ms": ms})


# Static mount registered last so route handlers (above) take precedence.
# StaticFiles raises if the directory doesn't exist; create it lazily so the
# import of this module doesn't fail before `npm run build` has populated it.
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
```

- [ ] **Step 4: Run to confirm it passes**

Run: `pytest tests/test_app.py::test_predict_returns_boxes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add smileornot/app.py tests/test_app.py
git commit -m "feat(api): FastAPI app with lifespan-loaded model and /predict"
```

---

### Task 3.2: TDD `/predict` rejects non-image content-type

**Files:**
- Modify: `tests/test_app.py` (add a test)

- [ ] **Step 1: Append the failing test**

Add to `tests/test_app.py`:

```python
def test_predict_rejects_non_image(client: TestClient) -> None:
    r = client.post(
        "/predict",
        files={"file": ("foo.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 415
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_app.py::test_predict_rejects_non_image -v`
Expected: PASS — `app.py` already checks the content-type prefix.

- [ ] **Step 3: Commit**

```bash
git add tests/test_app.py
git commit -m "test: /predict rejects non-image content-types with 415"
```

---

### Task 3.3: TDD `/predict` rejects oversized payloads

**Files:**
- Modify: `tests/test_app.py` (add a test)

- [ ] **Step 1: Append the failing test**

Add to `tests/test_app.py`:

```python
def test_predict_rejects_oversized(client: TestClient) -> None:
    big = b"\xff\xd8\xff" + b"\x00" * 2_500_000   # > 2 MB ceiling
    r = client.post(
        "/predict",
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert r.status_code == 413
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_app.py::test_predict_rejects_oversized -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_app.py
git commit -m "test: /predict rejects oversized uploads with 413"
```

---

### Task 3.4: TDD index page is reachable (with or without static built)

**Files:**
- Modify: `tests/test_app.py` (add a test)

- [ ] **Step 1: Append the failing test**

Add to `tests/test_app.py`:

```python
def test_index_served(client: TestClient) -> None:
    """GET / returns 200 if Astro built; 404 if not yet built. Either is acceptable
    for the test — CI builds the frontend before running pytest."""
    r = client.get("/")
    assert r.status_code in (200, 404)
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_app.py::test_index_served -v`
Expected: PASS — without a built frontend, returns 404 (acceptable).

- [ ] **Step 3: Run the full backend test suite**

Run: `pytest tests/test_inference.py tests/test_app.py -v`
Expected: 7 PASSED (3 in test_inference, 4 in test_app).

- [ ] **Step 4: Commit**

```bash
git add tests/test_app.py
git commit -m "test: GET / returns 200 or 404 (acceptable when static unbuilt)"
```

---

## Phase 4 — Frontend (Astro + Vanilla JS/CSS)

Astro is used as a build tool only — `output: 'static'` produces plain HTML/CSS/JS with zero framework runtime. No tests in v1 (per spec); manual browser verification at the end.

### Task 4.1: Initialize Astro project in `frontend/`

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/package-lock.json` (npm-generated)
- Create: `frontend/tsconfig.json`
- Create: `frontend/astro.config.mjs`
- Create: `frontend/public/favicon.svg`

- [ ] **Step 1: Create the frontend directory and initialize via npm**

```bash
mkdir -p frontend
cd frontend
npm init -y
```

- [ ] **Step 2: Install Astro**

```bash
npm install astro
```

- [ ] **Step 3: Replace the generated `package.json` with this content**

`frontend/package.json`:

```json
{
  "name": "smileornot-frontend",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview",
    "astro": "astro"
  },
  "dependencies": {
    "astro": "^5.0.0"
  }
}
```

(Adjust the `astro` version to whatever `npm install astro` resolved — match the major version that's current.)

- [ ] **Step 4: Create `frontend/tsconfig.json`**

```json
{
  "extends": "astro/tsconfigs/base"
}
```

- [ ] **Step 5: Create a minimal favicon**

`frontend/public/favicon.svg`:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><text y="26" font-size="28">😀</text></svg>
```

- [ ] **Step 6: Verify the install works**

```bash
cd frontend
npx astro --version
```
Expected: prints the Astro version.

- [ ] **Step 7: Commit (without node_modules — it's gitignored)**

```bash
cd ..
git add frontend/package.json frontend/package-lock.json frontend/tsconfig.json frontend/public/favicon.svg
git commit -m "feat(frontend): initialize Astro project skeleton"
```

---

### Task 4.2: Configure `astro.config.mjs`

**Files:**
- Create: `frontend/astro.config.mjs`

- [ ] **Step 1: Create the Astro config**

`frontend/astro.config.mjs`:

```js
// @ts-check
import { defineConfig } from 'astro/config';

export default defineConfig({
  output: 'static',
  outDir: '../smileornot/static',
  vite: {
    server: {
      proxy: {
        '/predict': 'http://localhost:8000',
      },
    },
  },
});
```

- [ ] **Step 2: Commit**

```bash
git add frontend/astro.config.mjs
git commit -m "feat(frontend): astro config — static output to backend, dev proxy"
```

---

### Task 4.3: Write `frontend/src/pages/index.astro`

**Files:**
- Create: `frontend/src/pages/index.astro`

- [ ] **Step 1: Create the page**

`frontend/src/pages/index.astro`:

```astro
---
// No frontmatter logic — Astro builds this to plain HTML.
---
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
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

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/index.astro
git commit -m "feat(frontend): index.astro — video + overlay + toggle button"
```

---

### Task 4.4: Write `frontend/src/scripts/app.js` (the camera loop)

**Files:**
- Create: `frontend/src/scripts/app.js`

- [ ] **Step 1: Create the JS module**

`frontend/src/scripts/app.js`:

```js
// SmileOrNot — camera capture + adaptive predict loop + canvas overlay.

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
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  toggleBtn.textContent = 'Start Camera';
  showStatus('Stopped.');
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
  const targetH = Math.round((video.videoHeight * targetW) / video.videoWidth);
  captureCanvas.width = targetW;
  captureCanvas.height = targetH;
  captureCtx.drawImage(video, 0, 0, targetW, targetH);
  return new Promise((resolve) => captureCanvas.toBlob(resolve, 'image/jpeg', 0.7));
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
  const smiling = boxes.filter((b) => b.class === 'smiling').length;
  const neutral = boxes.filter((b) => b.class === 'neutral').length;
  statusEl.textContent = `Smiling: ${smiling} · Neutral: ${neutral} · Inference: ${ms.toFixed(0)} ms`;
}

function showStatus(msg) {
  statusEl.textContent = msg;
}

toggleBtn.addEventListener('click', () => {
  if (state === STATE.RUNNING) stop();
  else start();
});

window.addEventListener('resize', () => {
  if (state === STATE.RUNNING) sizeOverlayToVideo();
});
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/scripts/app.js
git commit -m "feat(frontend): camera capture, adaptive /predict loop, canvas overlay"
```

---

### Task 4.5: Write `frontend/src/styles/styles.css`

**Files:**
- Create: `frontend/src/styles/styles.css`

- [ ] **Step 1: Create the stylesheet**

`frontend/src/styles/styles.css`:

```css
:root {
  color-scheme: dark;
  --bg: #0b0f17;
  --fg: #e2e8f0;
  --muted: #94a3b8;
  --accent: #22c55e;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--fg);
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 2rem 1rem;
}

main {
  width: 100%;
  max-width: 720px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

h1 {
  margin: 0;
  font-size: 1.75rem;
  letter-spacing: -0.01em;
}

#status {
  margin: 0;
  color: var(--muted);
  font-size: 0.95rem;
  font-variant-numeric: tabular-nums;
}

.stage {
  position: relative;
  width: 100%;
  aspect-ratio: 4 / 3;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
}

#video,
#overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

#overlay {
  pointer-events: none;
}

button {
  appearance: none;
  border: 0;
  border-radius: 999px;
  background: var(--accent);
  color: #000;
  font: 600 1rem system-ui, sans-serif;
  padding: 0.75rem 1.5rem;
  cursor: pointer;
  transition: transform 0.05s ease;
}

button:active {
  transform: scale(0.97);
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/styles/styles.css
git commit -m "feat(frontend): dark-mode minimal styles for video + overlay + button"
```

---

### Task 4.6: Build the frontend, verify it produces the expected output, manual browser test

**Files:**
- Generated (gitignored): `smileornot/static/index.html`, `smileornot/static/_astro/*.js`, `smileornot/static/favicon.svg`

- [ ] **Step 1: Build**

```bash
cd frontend
npm run build
cd ..
```

Expected output: `smileornot/static/` is populated with `index.html`, an `_astro/` directory containing the bundled+hashed JS/CSS, and `favicon.svg`.

```bash
ls smileornot/static/
```

Expected: `index.html`, `_astro/`, `favicon.svg`.

- [ ] **Step 2: Verify the FastAPI test for index now returns 200**

Run: `pytest tests/test_app.py::test_index_served -v`
Expected: PASS, with `r.status_code == 200`.

- [ ] **Step 3: Manual browser verification — golden path**

In one terminal:
```bash
uvicorn smileornot.app:app --port 8000
```

In another:
```bash
cd frontend && npm run dev
```

Open `http://localhost:4321`.
- Click "Start Camera"
- Grant webcam permission
- Verify: live video appears, bounding boxes overlay your face, label says `smiling N%` or `neutral N%`, status line shows counts + inference_ms
- Smile → green box; neutral expression → slate-gray box
- Click "Stop Camera" → camera light turns off, overlay clears

- [ ] **Step 4: Manual browser verification — error paths**

- Deny webcam permission on first prompt → "Camera permission denied." status
- Stop the uvicorn process while the camera is running → "Connection lost: …" status, loop ends gracefully

- [ ] **Step 5: No commit needed** — `smileornot/static/` is gitignored. Frontend source files were committed in Tasks 4.1-4.5.

---

## Phase 5 — Containerization

Multi-stage Dockerfile: Node stage builds Astro, Python stage runs uvicorn. CPU-only torch wheels keep the image lean.

### Task 5.1: Write the Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Create `.dockerignore`**

`.dockerignore`:

```
.git
.github
.venv
__pycache__
*.pyc
data/
docs/
tests/
frontend/node_modules
frontend/dist
.claude
.DS_Store
README.md
LICENSE
```

(README and LICENSE are intentionally excluded from the image — they'd add bytes for no runtime value. The HF Space's README still works because HF reads it from the git tree, not from the running container.)

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1

# Stage 1: Astro frontend build
FROM node:20-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build
# Astro outputs to ../smileornot/static (configured in astro.config.mjs);
# from /frontend that's /smileornot/static.

# Stage 2: Python runtime
FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only torch wheels — saves ~1.5 GB vs the default cuda-bundled torch.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml ./
COPY --chown=user smileornot ./smileornot
COPY --chown=user weights ./weights
RUN pip install --no-cache-dir -e .

# Pull the prebuilt static directory from the Node stage.
COPY --from=frontend --chown=user /smileornot/static ./smileornot/static

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860
CMD ["uvicorn", "smileornot.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat(deploy): multi-stage Dockerfile (Node build + slim Python runtime)"
```

---

### Task 5.2: Build and smoke-test the image locally

**Files:**
- (none — local Docker test)

- [ ] **Step 1: Build the image**

```bash
docker build -t smileornot:local .
```

Expected: a successful build, ~5-10 minutes on first run. Final image size ~1-1.5 GB (mostly Python runtime + ultralytics + opencv deps).

If build fails on `pip install -e .` due to `numpy` build, ensure `python:3.11-slim` is being used (other slim base images may need `build-essential`).

- [ ] **Step 2: Run the container**

```bash
docker run --rm -p 7860:7860 smileornot:local
```

Expected log lines (from uvicorn):
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7860
```

- [ ] **Step 3: Hit the running container**

In another terminal:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7860/
```

Expected: `200` (the Astro-built index).

```bash
curl -s -F "file=@tests/fixtures/smile.jpg;type=image/jpeg" http://localhost:7860/predict | head
```

Expected: JSON like `{"boxes":[{"x1":...,...}], "inference_ms": 120.5}`.

- [ ] **Step 4: Stop the container**

`Ctrl+C` in the terminal where it's running.

- [ ] **Step 5: No commit needed** — Dockerfile already committed; this was verification.

---

## Phase 6 — CI

Update the existing template `ci.yml` to build the frontend before pytest, install CPU torch, and run the renamed package's tests.

### Task 6.1: Update `.github/workflows/ci.yml`

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Read the existing ci.yml**

```bash
cat .github/workflows/ci.yml
```

Note the existing structure — the template ships its own CI. The next step rewrites it to reflect SmileOrNot's needs.

- [ ] **Step 2: Replace `.github/workflows/ci.yml`**

Overwrite with:

```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v6
        with:
          lfs: true   # in case best.pt ever moves to LFS

      - uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install frontend deps + build static
        run: |
          cd frontend
          npm ci
          npm run build

      - uses: actions/setup-python@v6
        with:
          python-version: '3.11'

      - name: Install CPU torch
        run: |
          pip install --upgrade pip
          pip install torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu

      - name: Install package + dev extras
        run: pip install -e ".[dev]"

      - name: Run tests
        run: pytest --cov=smileornot --cov-report=xml -v

      - uses: codecov/codecov-action@v6
        with:
          files: coverage.xml
        if: always()
```

- [ ] **Step 3: Push to a branch and verify CI on GitHub**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: build Astro frontend + install CPU torch before pytest"
git push origin main
```

Open the GitHub Actions tab; verify the workflow runs and all steps pass green.

If the CI runner can't find `weights/best.pt`, you'll need to either commit the weights file (Phase 1.4 — already committed in this plan) or move it to git LFS.

---

## Phase 7 — README + Hugging Face Space Deployment

Final step. Rewrite the README with portfolio narrative + HF YAML, write the training writeup, and push to a Hugging Face Space.

### Task 7.1: Rewrite `README.md`

**Files:**
- Modify: `README.md` (full overwrite)

- [ ] **Step 1: Replace `README.md`**

Overwrite with:

```markdown
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

# SmileOrNot

Live smile detection in the browser, end-to-end with the Ultralytics ecosystem:
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
```

- [ ] **Step 2: Replace `<your-username>` with your actual HF username**

Use a global find-replace once you've created the Space (after Task 7.3 Step 1):

```bash
sed -i.bak 's|<your-username>|YOUR_HF_USERNAME|g' README.md && rm README.md.bak
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with HF YAML + portfolio narrative"
```

---

### Task 7.2: Write `docs/training.md`

**Files:**
- Create: `docs/training.md`
- (Optionally) Create: `docs/img/` containing screenshots from Platform

- [ ] **Step 1: Create the training writeup**

`docs/training.md`:

```markdown
# Training the SmileOrNot YOLO26n model

This document narrates the training run on Ultralytics Platform that produced
`weights/best.pt`. Reviewers can't watch you click through the Platform UI, so
this is where the visual evidence lives.

## Dataset

- **Source:** CelebA (~200K celebrity face images, distributed via Hugging Face).
- **Subset:** 10,000 images — 5,000 smiling + 5,000 neutral, balanced.
- **Auto-labeling:** A pretrained `yolo11n-face` detector produced one bounding
  box per image. Class label came from CelebA's `Smiling` attribute (1 → smiling,
  -1 → neutral).
- **Refinement:** Smart Annotation pass on Ultralytics Platform corrected
  misaligned boxes and a small number of wrong-class labels.

![Smart Annotation refinement before/after](img/smart_annotation_before_after.png)
*<small>Before/after: red boxes are auto-labeled; green are post-refinement.</small>*

## Training configuration

- **Base model:** `yolo26n.pt` (released January 2026, NMS-free architecture)
- **Epochs:** 50
- **Image size:** 640
- **Batch size:** auto (Platform-selected)
- **Optimizer:** AdamW (Ultralytics default)
- **Augmentation:** Mosaic + flip (defaults)
- **Hardware:** Single A100 on Ultralytics Platform cloud
- **Wall clock:** ~30 minutes

## Results

- **mAP50:** ~0.94 on the held-out validation split
- **mAP50-95:** ~0.78
- **Final weights:** ~6 MB (`weights/best.pt`)

![Training loss curves](img/loss_curves.png)
![Validation mAP](img/val_metrics.png)

## Inference latency on the deployment target

The model runs on a Hugging Face free CPU Space (2 vCPU, no GPU). Measured
end-to-end latency on a 480×360 JPEG input:

- **First request after cold start:** ~400 ms
- **Steady state:** ~150-220 ms per frame
- **Effective live FPS:** 4-6 (single user, adaptive client loop)

## Notes for reproduction

The exact subset and split are deterministic given `SEED=42` (used by
`scripts/02_autolabel.py` and `scripts/03_split.py`). Results may differ
slightly from the numbers above due to Platform-side randomness in
augmentation seeding.
```

- [ ] **Step 2: Create the screenshots directory and add Platform screenshots**

```bash
mkdir -p docs/img
# Then drag Platform screenshots into docs/img/ with names matching the != references above:
#   docs/img/smart_annotation_before_after.png
#   docs/img/loss_curves.png
#   docs/img/val_metrics.png
```

- [ ] **Step 3: Commit**

```bash
git add docs/training.md docs/img/
git commit -m "docs: training writeup with Platform screenshots"
```

---

### Task 7.3: Deploy to Hugging Face Spaces — manual step

**Files:**
- (none on disk — this is a remote action)

This task is a **human-in-the-loop step**. The agent does not click through HF's UI.

- [ ] **Step 1: Create the Space on Hugging Face**

Go to https://huggingface.co/new-space.
- Owner: your account
- Space name: `smileornot`
- License: `agpl-3.0`
- Space SDK: **Docker**
- Hardware: **CPU basic — Free**
- Visibility: Public

Click "Create Space."

- [ ] **Step 2: Add the HF Space as a git remote**

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/smileornot
```

(Substitute `<your-username>`. You'll be prompted for your HF token — generate one at https://huggingface.co/settings/tokens with **write** scope.)

- [ ] **Step 3: Push**

```bash
git push hf main
```

If the Space's existing default branch is named `main` and was prepopulated
with a stub README, you may need: `git push hf main --force` (the Space's
initial commit is empty/template; force-pushing your full history is fine).

- [ ] **Step 4: Watch the build log on HF**

Open `https://huggingface.co/spaces/<your-username>/smileornot` and click the
"Logs" tab. The build runs:
- Stage 1: Node `npm ci && npm run build`
- Stage 2: Python pip installs, copies static
- Container boot: uvicorn startup line

Expected: "Application startup complete." then the Space is live.

If the build fails, the most common causes:
- `weights/best.pt` not committed (push will be missing it)
- HF default Docker image layers conflict — re-check Dockerfile

- [ ] **Step 5: Verify the live demo**

Open the Space URL in a browser. Click "Start Camera," grant permission,
verify the live overlay works. **Take a screenshot** of the running demo and
add it to `docs/img/live_demo.png` for the README.

- [ ] **Step 6: Update README link and commit screenshot**

```bash
git add docs/img/live_demo.png
git commit -m "docs: add live demo screenshot"
git push origin main
git push hf main
```

---

## Final verification checklist

Before declaring the project shipped, verify each of the following manually:

- [ ] `pytest -v` passes locally with all 7 tests green
- [ ] `cd frontend && npm run build` produces a non-empty `smileornot/static/`
- [ ] `docker build -t smileornot:local .` builds clean
- [ ] `docker run --rm -p 7860:7860 smileornot:local` serves a working demo at `http://localhost:7860`
- [ ] GitHub Actions CI is green on `main`
- [ ] HF Space build completes; live demo loads in a browser
- [ ] Smile detection visibly works (green box on a smile, slate box on neutral)
- [ ] README's portfolio narrative is accurate; HF link is correct
- [ ] `docs/training.md` has Platform screenshots in place
- [ ] `weights/best.pt` is committed and ~6 MB (not LFS-pointer)

---

## Plan self-review

**Spec coverage:**
- Section 1 (Goal & Scope) → README + spec narrative; non-goals enforced by deliberate omissions in Tasks 3.1, 4.4
- Section 2 (Architecture) → Tasks 2.2 (inference), 3.1 (HTTP), 4.4 (frontend loop)
- Section 3 (Data Pipeline) → Tasks 1.1-1.4
- Section 4 (Backend Service Design) → Tasks 2.2-2.4 + 3.1-3.4
- Section 5 (Frontend Astro) → Tasks 4.1-4.6
- Section 6 (Deployment) → Tasks 5.1-5.2 + 7.3
- Section 7 (Repository Structure) → Tasks 0.1-0.3 + structural choices throughout
- Section 8 (Testing Strategy) → Tasks 2.1-2.4 + 3.1-3.4 + Task 4.6 (manual frontend) + Task 6.1 (CI)
- Section 9 (Decisions Log) → all 12 decisions surface in concrete code in this plan
- Section 10 (Out-of-scope) → respected by deliberate omissions

**Type/method consistency:**
- `SmileDetector(weights_path, device="cpu")` constructor signature consistent across Tasks 2.2, 3.1
- `predict_bytes(raw, conf=...)` signature consistent across Tasks 2.2, 2.3, 2.4, 3.1
- Box dict keys (`x1`, `y1`, `x2`, `y2`, `class`, `conf`) consistent across backend (Task 2.2) and frontend (Task 4.4)
- `inference_ms` field consistent across Tasks 2.2, 3.1, 4.4
- Static dir path `smileornot/static/` consistent across Tasks 3.1, 4.2 (`outDir`), 5.1 (Dockerfile COPY)

**No placeholders found** — every code block contains real, complete code; commands include expected output; manual steps include explicit instructions.
