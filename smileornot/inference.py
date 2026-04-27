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
