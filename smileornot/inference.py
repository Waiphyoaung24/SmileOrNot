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
