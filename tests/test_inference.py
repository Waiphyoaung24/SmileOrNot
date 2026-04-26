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


def test_predict_no_face(detector: SmileDetector) -> None:
    raw = (FIXTURES / "no_face.jpg").read_bytes()
    boxes, _ = detector.predict_bytes(raw, conf=0.5)
    assert boxes == []
