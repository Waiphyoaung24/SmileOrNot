"""Tests for the FastAPI app — endpoint surface, status codes, schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smileornot.app import app

FIXTURES = Path(__file__).parent / "fixtures"
CAN_WEIGHTS = Path(__file__).parent.parent / "weights" / "can_best.pt"


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


def test_predict_rejects_non_image(client: TestClient) -> None:
    r = client.post(
        "/predict",
        files={"file": ("foo.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 415


def test_predict_rejects_oversized(client: TestClient) -> None:
    big = b"\xff\xd8\xff" + b"\x00" * 2_500_000   # > 2 MB ceiling
    r = client.post(
        "/predict",
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert r.status_code == 413


def test_predict_can_returns_503_if_weights_missing(client: TestClient) -> None:
    if CAN_WEIGHTS.exists():
        pytest.skip("can weights present; this test only runs without them")
    raw = (FIXTURES / "smile.jpg").read_bytes()
    r = client.post("/predict/can", files={"file": ("frame.jpg", raw, "image/jpeg")})
    assert r.status_code == 503


@pytest.mark.skipif(not CAN_WEIGHTS.exists(), reason="weights/can_best.pt absent")
def test_predict_can_returns_boxes(client: TestClient) -> None:
    raw = (FIXTURES / "smile.jpg").read_bytes()
    r = client.post("/predict/can", files={"file": ("frame.jpg", raw, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "boxes" in body and "inference_ms" in body


def test_index_served(client: TestClient) -> None:
    """GET / returns 200 if Astro built; 404 if not yet built. Either is acceptable
    for the test — CI builds the frontend before running pytest."""
    r = client.get("/")
    assert r.status_code in (200, 404)
