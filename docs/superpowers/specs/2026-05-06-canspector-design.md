# Canspector — Can Condition Detector

**Status:** Draft
**Date:** 2026-05-06
**Scope:** Add a second YOLO model and demo page to the existing SmileOrNot
project. Detects beverage cans and classifies each as one of four conditions.

## Goal

Extend SmileOrNot with a mobile-first can-condition detector that mirrors the
existing smile workflow: Roboflow dataset → Ultralytics Platform training →
`weights/can_best.pt` → FastAPI `/predict/can` → Astro static page at `/can`.

Same container, same HF Space, two demos.

**Out of scope (future spec):** any user-facing data upload or model retraining
platform. This spec is inference-only on a model trained by the maintainer.

## Classes

Four mutually exclusive labels, one box per can:

| id | name                  |
|----|-----------------------|
| 0  | `intact_labeled`      |
| 1  | `intact_unlabeled`    |
| 2  | `damaged_labeled`     |
| 3  | `damaged_unlabeled`   |

`damaged` collapses dented / crushed / punctured. `labeled` means the can's
printed label/wrapper is present and recognizable.

## Data pipeline

New scripts under `scripts/`, prefixed `can_` so they can't be confused with
smile scripts:

- **`scripts/can_01_download.py`** — pulls a Roboflow can-damage dataset in
  YOLOv8 format into `data/can/raw/`. The Roboflow workspace/project/version
  are constants (`ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT`, `ROBOFLOW_VERSION`)
  at the top of the file. The maintainer edits these after picking a dataset.
- **`scripts/can_02_remap.py`** — applies a `CLASS_MAP: dict[str, int]` (also
  a top-of-file constant) to convert the source dataset's class names into the
  4 target ids. Drops images whose labels can't be mapped. Writes
  `data/can/yolo/{images,labels}/`.
- **`scripts/can_03_split.py`** — stratified 80/10/10 split, mirroring
  `03_split.py`. Output: `data/can/yolo/{images,labels}/{train,val,test}/`.
- **`scripts/can_dataset.yaml`** — Ultralytics Platform dataset config.

After running these locally, the maintainer uploads `data/can/yolo/` to
Ultralytics Platform, runs Smart Annotation if needed, trains YOLO26n for 50
epochs at imgsz=640, and downloads the weights to `weights/can_best.pt`.

**Open variables (TODO before first run):**
- The Roboflow dataset slug.
- The exact `CLASS_MAP`, which depends on the chosen dataset's class names.

## Backend

Refactor `smileornot/inference.py`:
- Generalize `SmileDetector` into `YoloDetector(weights_path: Path,
  class_names: list[str], device: str = "cpu")`. Same `predict_bytes()` API.
- Keep `SmileDetector` as a thin subclass or alias for backwards compat with
  existing tests.

Update `smileornot/app.py`:
- At startup load both detectors:
  - `app.state.smile_detector` ← `weights/best.pt`, `["smiling", "neutral"]`
  - `app.state.can_detector` ← `weights/can_best.pt`,
    `["intact_labeled", "intact_unlabeled", "damaged_labeled", "damaged_unlabeled"]`
- If `weights/can_best.pt` is missing, log a warning and leave
  `app.state.can_detector = None`. Do **not** crash startup — the smile demo
  must still work.
- Endpoints:
  - `POST /predict` — unchanged.
  - `POST /predict/can` — same request shape (multipart `file`), same
    response shape `{boxes: [...], inference_ms: float}`. Returns **503** if
    `app.state.can_detector is None`.

Tests:
- Extend `tests/test_app.py` with one happy-path test for `/predict/can`
  using a fixture image. Skip if `weights/can_best.pt` is absent.

## Frontend

Two pages share a layout and a navbar with GSAP transitions.

**Pages**
- `/` — smile demo (existing). Migrate to use the new shared layout.
- `/can` — new. Mobile-first single-column layout. A segmented toggle at the
  top selects the input mode:
  - **Live mode** — `getUserMedia({ video: { facingMode: { ideal:
    'environment' } } })`, adaptive predict loop POSTing to `/predict/can`.
  - **Upload mode** — `<input type="file" accept="image/*"
    capture="environment">`. Single POST to `/predict/can`. Boxes drawn on a
    still `<img>` via the same canvas overlay code.

**Shared modules**
- Extract camera/loop/capture/draw logic out of `frontend/src/scripts/app.js`
  into `frontend/src/scripts/detector.js`. Parameters: endpoint URL, class
  → stroke-color map, optional "still image" mode.
- Smile page and Can page each instantiate the detector with their own
  config.

**Navbar + layout**
- New Astro layout `frontend/src/layouts/Base.astro` wraps every page. Both
  `/` and `/can` switch to it.
- New component `frontend/src/components/Navbar.astro`. Two links (`Smile`,
  `Can`). Active route highlighted via `aria-current="page"`.
- Mobile: hamburger toggle → slide-in panel.
- Astro 5's built-in `<ClientRouter />` (View Transitions) drives route
  swaps. GSAP (npm `gsap`) layers on motion:
  - **Mount:** nav links stagger in (y: -20 → 0, opacity 0 → 1, ~0.4s total).
  - **Page swap:** on `astro:after-swap`, GSAP `fromTo` on the new page's
    `<main>` element (opacity 0 → 1, y: 12 → 0, ~0.3s).
  - **Mobile menu:** GSAP timeline for open/close (height + opacity).

**Build / serve**
- `npm run build` still outputs to `../smileornot/static/`. FastAPI serves
  both routes from the static mount. No server-side change required for
  routing.

**Deferred to implementation:**
- Exact 4-class color palette (lean: green / lime / amber / red).
- Navbar position on mobile (lean: fixed top).

## Deployment

- Same `Dockerfile`, same `docker-compose.yml`, same HF Docker Space.
- The image bundles both `weights/best.pt` and `weights/can_best.pt` (when
  the latter exists locally at build time). If the can weights file is
  missing during the build, the image still ships and the can page returns
  503 until it's added.

## Testing

- `pytest -v` — backend smoke + endpoint tests (smile, can).
- `cd frontend && npm run build` — static build must succeed.
- Manual: open the deployed Space on a phone, verify rear camera works and
  upload mode accepts a captured photo.

## License

AGPL-3.0, unchanged.
