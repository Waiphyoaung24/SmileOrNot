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
