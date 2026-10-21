"""
Phase 5 — tests for dataset collection (service + HTTP endpoints).

Honesty-focused: a sample is only recorded when a real hand is detected, labels
are validated, and the produced CSV is exactly what app.ml.train_model consumes.
A real hand image (backend/diag_hand.png) exercises the 'recorded' path; a solid
black frame exercises the 'no_hand' path. No gesture data is fabricated.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.dataset_service import (
    CSV_HEADER,
    SUPPORTED_LABELS,
    DatasetService,
)
import app.services.dataset_service as dataset_module


BACKEND_DIR = Path(__file__).resolve().parent.parent
HAND_IMAGE = BACKEND_DIR / "diag_hand.png"


def _black_jpeg() -> bytes:
    img = Image.new("RGB", (640, 480), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _hand_png() -> bytes:
    return HAND_IMAGE.read_bytes()


# ── Service-level tests ─────────────────────────────────────────────────────

def test_csv_header_matches_trainer_contract():
    # label + 63 feature columns (21 landmarks x 3 coords) — matches train_model.
    assert CSV_HEADER[0] == "label"
    assert len(CSV_HEADER) == 64
    assert CSV_HEADER[1:4] == ["x0", "y0", "z0"]
    assert CSV_HEADER[-3:] == ["x20", "y20", "z20"]


def test_invalid_label_is_rejected_and_writes_nothing(tmp_path):
    svc = DatasetService(csv_path=tmp_path / "landmarks.csv")
    res = svc.add_sample("ZZZ", _hand_png())
    assert res["status"] == "invalid_label"
    assert res["total"] == 0
    assert not (tmp_path / "landmarks.csv").exists()


def test_no_hand_frame_records_nothing(tmp_path):
    svc = DatasetService(csv_path=tmp_path / "landmarks.csv")
    res = svc.add_sample("A", _black_jpeg())
    assert res["status"] == "no_hand"
    assert res["total"] == 0
    assert res["total_for_label"] == 0


@pytest.mark.skipif(not HAND_IMAGE.exists(), reason="diag_hand.png not available")
def test_real_hand_is_recorded_as_valid_row(tmp_path):
    csv_path = tmp_path / "landmarks.csv"
    svc = DatasetService(csv_path=csv_path)

    res = svc.add_sample("A", _hand_png())
    assert res["status"] == "recorded"
    assert res["label"] == "A"
    assert res["total_for_label"] == 1
    assert res["total"] == 1

    # Row is well-formed: header + one row of 64 columns, 63 parseable floats.
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_HEADER
    assert len(rows) == 2
    label, *features = rows[1]
    assert label == "A"
    assert len(features) == 63
    [float(v) for v in features]  # must not raise


@pytest.mark.skipif(not HAND_IMAGE.exists(), reason="diag_hand.png not available")
def test_counts_accumulate_across_samples(tmp_path):
    svc = DatasetService(csv_path=tmp_path / "landmarks.csv")
    svc.add_sample("A", _hand_png())
    svc.add_sample("A", _hand_png())
    svc.add_sample("V", _hand_png())

    stats = svc.stats()
    assert stats["total"] == 3
    assert stats["counts"]["A"] == 2
    assert stats["counts"]["V"] == 1
    # every supported label is present in the counts map (zero-filled)
    assert set(stats["counts"].keys()) == set(SUPPORTED_LABELS)
    assert stats["ready_to_train"] is False  # not all labels collected


@pytest.mark.skipif(not HAND_IMAGE.exists(), reason="diag_hand.png not available")
def test_collected_csv_loads_via_train_model(tmp_path):
    """The produced CSV must be consumable by the real training loader."""
    from app.ml.train_model import load_dataset

    csv_path = tmp_path / "landmarks.csv"
    svc = DatasetService(csv_path=csv_path)
    for label in ("A", "V"):
        svc.add_sample(label, _hand_png())

    X, y = load_dataset(csv_path)
    assert X.shape == (2, 63)
    assert sorted(set(y)) == ["A", "V"]


def test_stats_on_empty_dataset(tmp_path):
    svc = DatasetService(csv_path=tmp_path / "landmarks.csv")
    stats = svc.stats()
    assert stats["total"] == 0
    assert all(v == 0 for v in stats["counts"].values())
    assert stats["ready_to_train"] is False


# ── HTTP-level tests ────────────────────────────────────────────────────────

@pytest.fixture()
def client_with_temp_dataset(tmp_path, monkeypatch):
    """Point the module singleton at a temp CSV so tests never touch real data."""
    svc = DatasetService(csv_path=tmp_path / "landmarks.csv")
    monkeypatch.setattr(dataset_module, "_dataset_service", svc, raising=False)
    with TestClient(app) as c:
        yield c


def test_sample_rejects_missing_file(client_with_temp_dataset):
    r = client_with_temp_dataset.post("/api/dataset/sample", data={"label": "A"})
    assert r.status_code == 422


def test_sample_rejects_missing_label(client_with_temp_dataset):
    r = client_with_temp_dataset.post(
        "/api/dataset/sample",
        files={"frame": ("f.jpg", _black_jpeg(), "image/jpeg")},
    )
    assert r.status_code == 422


def test_sample_rejects_wrong_content_type(client_with_temp_dataset):
    r = client_with_temp_dataset.post(
        "/api/dataset/sample",
        files={"frame": ("f.txt", b"hello", "text/plain")},
        data={"label": "A"},
    )
    assert r.status_code == 400


def test_sample_rejects_empty_frame(client_with_temp_dataset):
    r = client_with_temp_dataset.post(
        "/api/dataset/sample",
        files={"frame": ("f.jpg", b"", "image/jpeg")},
        data={"label": "A"},
    )
    assert r.status_code == 400


def test_sample_invalid_label_over_http(client_with_temp_dataset):
    r = client_with_temp_dataset.post(
        "/api/dataset/sample",
        files={"frame": ("f.jpg", _black_jpeg(), "image/jpeg")},
        data={"label": "ZZZ"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "invalid_label"


def test_sample_no_hand_over_http(client_with_temp_dataset):
    r = client_with_temp_dataset.post(
        "/api/dataset/sample",
        files={"frame": ("f.jpg", _black_jpeg(), "image/jpeg")},
        data={"label": "A"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_hand"
    assert body["total"] == 0


def test_stats_endpoint_structure(client_with_temp_dataset):
    r = client_with_temp_dataset.get("/api/dataset/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert set(body["counts"].keys()) == set(SUPPORTED_LABELS)
    assert body["ready_to_train"] is False
    assert "csv_path" in body


@pytest.mark.skipif(not HAND_IMAGE.exists(), reason="diag_hand.png not available")
def test_record_then_stats_over_http(client_with_temp_dataset):
    r = client_with_temp_dataset.post(
        "/api/dataset/sample",
        files={"frame": ("hand.png", _hand_png(), "image/png")},
        data={"label": "A"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "recorded"

    stats = client_with_temp_dataset.get("/api/dataset/stats").json()
    assert stats["total"] == 1
    assert stats["counts"]["A"] == 1
