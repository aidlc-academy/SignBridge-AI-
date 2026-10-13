"""
Backend tests for Phase 1: health, model status, and gestures endpoints.

Run:
    cd backend
    pytest tests/ -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── Health ─────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "timestamp" in body


# ── Model status ───────────────────────────────────────────────────────────

def test_model_status_endpoint_reachable():
    r = client.get("/api/model/status")
    assert r.status_code == 200


def test_model_status_has_expected_fields():
    r = client.get("/api/model/status")
    body = r.json()
    assert "loaded" in body
    assert "message" in body
    assert isinstance(body["class_labels"], list)


def test_model_status_not_loaded_returns_message():
    r = client.get("/api/model/status")
    body = r.json()
    # In Phase 1 no model is trained yet — must return a clear message
    if not body["loaded"]:
        assert "model" in body["message"].lower() or "train" in body["message"].lower()


# ── Gestures ───────────────────────────────────────────────────────────────

def test_gestures_returns_list():
    r = client.get("/api/gestures")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_gestures_returns_ten_items():
    r = client.get("/api/gestures")
    assert len(r.json()) == 10


def test_gestures_have_required_fields():
    r = client.get("/api/gestures")
    for item in r.json():
        assert "id"          in item
        assert "label"       in item
        assert "description" in item
        assert "language"    in item
        assert "category"    in item


def test_gestures_labels_correct():
    r = client.get("/api/gestures")
    labels = [item["label"] for item in r.json()]
    expected = {"A", "B", "C", "D", "I", "L", "O", "V", "Y", "1"}
    assert set(labels) == expected


# ── Recognize — guard rails ────────────────────────────────────────────────

def test_recognize_rejects_missing_file():
    r = client.post("/api/recognize")
    assert r.status_code == 422   # FastAPI validation error


def test_recognize_rejects_wrong_content_type():
    r = client.post(
        "/api/recognize",
        files={"frame": ("test.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400


def test_recognize_rejects_empty_file():
    r = client.post(
        "/api/recognize",
        files={"frame": ("frame.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 400


def test_recognize_valid_jpeg_returns_structured_response():
    """Send a 1×1 black JPEG — should get a structured response (no crash)."""
    import io
    from PIL import Image

    img = Image.new("RGB", (1, 1), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    r = client.post(
        "/api/recognize",
        files={"frame": ("frame.jpg", buf.read(), "image/jpeg")},
    )
    # Should be 200 with a valid response even if no hand detected
    assert r.status_code == 200
    body = r.json()
    assert "status"     in body
    assert "confidence" in body
    assert "timestamp"  in body



# ── Phase 3: Landmark detection endpoint ───────────────────────────────────

def test_landmarks_rejects_missing_file():
    r = client.post("/api/recognize/landmarks")
    assert r.status_code == 422


def test_landmarks_rejects_wrong_content_type():
    r = client.post(
        "/api/recognize/landmarks",
        files={"frame": ("test.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400


def test_landmarks_rejects_empty_file():
    r = client.post(
        "/api/recognize/landmarks",
        files={"frame": ("frame.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 400


def test_landmarks_black_frame_returns_no_hand():
    """A solid black image has no hand — expect no_hand status."""
    import io
    from PIL import Image

    img = Image.new("RGB", (640, 480), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    r = client.post(
        "/api/recognize/landmarks",
        files={"frame": ("frame.jpg", buf.read(), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["hands_detected"] == 0
    assert body["status"] == "no_hand"
    assert isinstance(body["hands"], list)
    assert len(body["hands"]) == 0
    assert "latency_ms" in body


def test_landmarks_response_structure():
    """Validate the full schema structure of the landmark response."""
    import io
    from PIL import Image

    img = Image.new("RGB", (640, 480), color=(50, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    r = client.post(
        "/api/recognize/landmarks",
        files={"frame": ("frame.jpg", buf.read(), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "hands_detected" in body
    assert "hands" in body
    assert "status" in body
    assert "message" in body
    assert "latency_ms" in body
    assert isinstance(body["hands_detected"], int)
    assert isinstance(body["hands"], list)
    assert isinstance(body["latency_ms"], int)



# ── Phase 3: Vision stack diagnostic endpoint ─────────────────────────────

def test_vision_status_reports_stack_health():
    """The vision diagnostic endpoint reports OpenCV + MediaPipe init state."""
    r = client.get("/api/vision/status")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "hands_init" in body
    assert "opencv" in body
    assert "mediapipe" in body
    # On a correctly provisioned environment the stack must initialize.
    assert body["status"] == "ok"
    assert body["hands_init"] is True
