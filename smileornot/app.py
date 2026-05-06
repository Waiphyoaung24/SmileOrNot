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

CAN_CLASSES = ["can"]

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


app = FastAPI(lifespan=lifespan, title="Vision", version="0.2.0")


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
