"""
End-to-end tests for the LSTM feature over REAL HTTP with REAL data:

  real frames (from the model author's recorded demo video, as lossless PNG)
  -> uvicorn -> FastAPI -> Pillow decode -> real MediaPipe Holistic ->
  30-frame buffer -> real LSTM weights -> JSON response

Nothing in the data path is mocked. What this does and does NOT prove:
  * proves the whole pipeline runs, agrees with an independent in-process
    computation of the same frames, and that the hand-visibility gate stops the
    model's known "confident word on an empty scene" behaviour;
  * does NOT measure sign-recognition accuracy: the recording has the author's
    landmark drawings burned into the pixels, so Holistic sees little of the
    real face/hands. No labelled, overlay-free sign data is available.

Frames are sent as PNG on purpose: JPEG compression makes Holistic lose the
borderline pose detection on this close-up footage.

Skipped (not faked) if the weights or the video are not present.
"""
from __future__ import annotations

import io
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from app import config

BACKEND_DIR = Path(__file__).resolve().parent.parent
WEIGHTS = config.LSTM_WEIGHTS_PATH
VIDEO = WEIGHTS.parent / "OpenCV Feed 2023-07-03 16-32-41.mp4"

# 50 consecutive frames; Holistic detects a hand in about 20 of them (video frames 279-298)
HAND_RANGE = range(250, 300)
# Frames 90-149: the body appears at ~97 and no hand is ever detected. The final 30-frame window
# (120-149) has 30/30 pose frames, 0 hand frames, and the RAW model answers "thanks" at ~100%
# on it (verified by an independent continuous scan) -- the hazard the hand gate exists for.
# The run starts before the body appears so MediaPipe's tracking state matches a continuous run.
# NOTE: face is never detected in this footage (the author's mesh is burned over it), so this
# input is out-of-distribution; it demonstrates the gate works, not the in-distribution false-positive rate.
POSE_ONLY_RANGE = range(90, 150)

pytestmark = pytest.mark.skipif(
    not (WEIGHTS.is_file() and VIDEO.is_file()),
    reason="LSTM weights and/or the reference demo video are not present",
)


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start(port: int, extra_env: dict, tmp: Path):
    assert not _port_open(port), f"port {port} already in use"
    log = open(tmp / f"backend_{port}.log", "wb")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **extra_env}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=BACKEND_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 90
    while time.time() < deadline:
        assert proc.poll() is None, f"backend exited early; see {log.name}"
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=2).status_code == 200:
                return proc, log
        except httpx.TransportError:
            time.sleep(0.5)
    proc.kill()
    pytest.fail("backend did not become healthy")


def _stop(proc, log):
    proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def _read_png_frames(rng: range) -> list:
    """Real frames: the 640x480 video-content area of the recording, as lossless PNG bytes."""
    import cv2

    cap = cv2.VideoCapture(str(VIDEO))
    out, i = [], 0
    while True:
        ok, canvas = cap.read()
        if not ok or i >= rng.stop:
            break
        if i in rng:
            out.append(cv2.imencode(".png", canvas[0:480, 0:640])[1].tobytes())
        i += 1
    cap.release()
    assert len(out) == len(rng)
    return out


@pytest.fixture(scope="module")
def hand_frames():
    return _read_png_frames(HAND_RANGE)


@pytest.fixture(scope="module")
def pose_only_frames():
    return _read_png_frames(POSE_ONLY_RANGE)


@pytest.fixture(scope="module")
def default_server(tmp_path_factory):
    proc, log = _start(8010, {}, tmp_path_factory.mktemp("e2e_default"))
    yield "http://127.0.0.1:8010"
    _stop(proc, log)


@pytest.fixture(scope="module")
def open_gate_server(tmp_path_factory):
    """Same backend with the hand gate disabled -> shows what the raw model does on these frames."""
    proc, log = _start(8011, {"LSTM_MIN_HAND_FRAMES": "0"}, tmp_path_factory.mktemp("e2e_open"))
    yield "http://127.0.0.1:8011"
    _stop(proc, log)


def _post(c, base, png, sid):
    return c.post(
        f"{base}/api/recognize/sequence",
        files={"frame": ("f.png", png, "image/png")},
        data={"session_id": sid},
    )


def _stream(base, frames, sid):
    bodies, lat = [], []
    with httpx.Client(timeout=60) as c:
        for f in frames:
            t = time.perf_counter()
            r = _post(c, base, f, sid)
            lat.append((time.perf_counter() - t) * 1e3)
            assert r.status_code == 200, r.text
            bodies.append(r.json())
    return bodies, lat


# ───────────────────────── status + existing endpoints still fine ─────────────────────────

def test_lstm_status_over_http(default_server):
    s = httpx.get(f"{default_server}/api/lstm/status", timeout=10).json()
    assert s["loaded"] is True and s["labels"] == ["hello", "thanks", "iloveyou"]
    assert s["sequence_length"] == 30 and s["feature_size"] == 1662 and s["parameters"] == 596675


def test_existing_endpoints_unaffected(default_server):
    from PIL import Image

    assert httpx.get(f"{default_server}/api/health", timeout=10).json()["status"] == "ok"
    assert httpx.get(f"{default_server}/api/model/status", timeout=10).status_code == 200
    assert len(httpx.get(f"{default_server}/api/gestures", timeout=10).json()) == 10
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (0, 0, 0)).save(buf, format="JPEG")
    jpg = buf.getvalue()
    r = httpx.post(f"{default_server}/api/recognize/landmarks", files={"frame": ("f.jpg", jpg, "image/jpeg")}, timeout=30)
    assert r.status_code == 200 and r.json()["status"] == "no_hand"
    r = httpx.post(f"{default_server}/api/recognize", files={"frame": ("f.jpg", jpg, "image/jpeg")}, timeout=30)
    assert r.status_code == 200 and r.json()["status"] in ("no_hand", "model_unavailable")


# ───────────────────────── real frames, default (gated) configuration ─────────────────────────

def test_hand_frames_collect_then_predict_with_real_probabilities(default_server, hand_frames):
    bodies, lat = _stream(default_server, hand_frames, "e2e-hand")
    for i, b in enumerate(bodies[:29], start=1):
        assert b["status"] == "collecting" and b["frames_buffered"] == i and b["word"] is None
        assert b["method"] == "lstm" and b["session_id"] == "e2e-hand"
    after = bodies[29:]
    assert all(b["frames_buffered"] == 30 for b in after)
    # real hands were detected in this stretch -> the gate opens and the model really runs
    assert max(b["hands_in_window"] for b in after) >= config.LSTM_MIN_HAND_FRAMES
    ran = [b for b in after if b["probabilities"] is not None]
    assert ran, "model never ran although hands were visible"
    for b in ran:
        assert b["status"] in ("recognized", "low_confidence")
        assert set(b["probabilities"]) == {"hello", "thanks", "iloveyou"}
        assert sum(b["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
        assert b["confidence"] == pytest.approx(max(b["probabilities"].values()), abs=1e-6)
        assert (b["word"] is not None) == (b["status"] == "recognized")
        if b["status"] == "recognized":
            assert b["confidence"] >= config.LSTM_CONF_THRESHOLD
    print(f"\nlatency per frame (client round trip, Holistic+LSTM): mean {statistics.mean(lat):.0f} ms, "
          f"median {statistics.median(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms")
    print("window statuses after fill:", sorted({b['status'] for b in after}),
          "| hands_in_window range:", min(b['hands_in_window'] for b in after), "-", max(b['hands_in_window'] for b in after))


def test_gate_stops_confident_word_on_body_without_hand(default_server, open_gate_server, pose_only_frames):
    """Regression for the model's real hazard: on a body with NO hand the raw model says 'thanks' ~100%.
    With the default gate the API must return no word; with the gate disabled we can see the raw model do it."""
    gated, _ = _stream(default_server, pose_only_frames, "e2e-gate")
    raw, _ = _stream(open_gate_server, pose_only_frames, "e2e-raw")
    g, r = gated[-1], raw[-1]
    # raw model (gate off): a confident, wrong-by-construction word although no hand was ever seen
    assert r["hands_in_window"] == 0 and r["status"] == "recognized" and r["word"] == "thanks" and r["confidence"] > 0.95
    # default configuration: no word, no probabilities, no confirmation, anywhere in the stream
    assert g["status"] == "no_hand" and g["hands_in_window"] == 0
    assert all(b["word"] is None and b["confirmed_word"] is None for b in gated)
    print(f"\nraw model on hand-less body: word={r['word']} conf={r['confidence']:.3f}  |  gated API: status={g['status']}")


def test_reset_restarts_collection(default_server, hand_frames):
    sid = "e2e-reset"
    with httpx.Client(timeout=60) as c:
        for f in hand_frames[:5]:
            _post(c, default_server, f, sid)
        assert c.post(f"{default_server}/api/recognize/sequence/reset", data={"session_id": sid}).json() == {"session_id": sid, "reset": True}
        assert c.post(f"{default_server}/api/recognize/sequence/reset", data={"session_id": sid}).json()["reset"] is False
        assert _post(c, default_server, hand_frames[0], sid).json()["frames_buffered"] == 1


def test_input_validation_over_http(default_server, hand_frames):
    with httpx.Client(timeout=30) as c:
        base = default_server
        assert c.post(f"{base}/api/recognize/sequence").status_code == 422
        assert c.post(f"{base}/api/recognize/sequence", files={"frame": ("a.txt", b"hi", "text/plain")}).status_code == 400
        assert c.post(f"{base}/api/recognize/sequence", files={"frame": ("f.jpg", b"", "image/jpeg")}).status_code == 400
        assert _post(c, base, hand_frames[0], "bad id!").status_code == 400
        assert c.post(f"{base}/api/recognize/sequence/reset", data={"session_id": "../x"}).status_code == 400
        r = c.post(f"{base}/api/recognize/sequence", files={"frame": ("f.jpg", b"not a jpeg", "image/jpeg")}, data={"session_id": "e2e-bad"})
        assert r.status_code == 200 and r.json()["status"] == "error" and r.json()["word"] is None
        assert c.get(f"{base}/api/lstm/status").status_code == 200   # server still healthy


def test_sessions_are_independent_over_http(default_server, hand_frames):
    with httpx.Client(timeout=60) as c:
        for f in hand_frames[:3]:
            a = _post(c, default_server, f, "e2e-A").json()
        b = _post(c, default_server, hand_frames[0], "e2e-B").json()
    assert a["frames_buffered"] == 3 and b["frames_buffered"] == 1


# ───────────────────────── parity with an independent in-process computation ─────────────────────────

def test_service_output_equals_independent_pipeline_on_same_real_frames(open_gate_server, hand_frames):
    """Same PNG bytes -> (A) HTTP service, (B) a fresh Holistic + NumPy model in this process.
    Final-window probabilities and hand counts must agree, and the input must be non-trivial."""
    import cv2
    import mediapipe as mp
    from PIL import Image

    from app.ml.lstm_model import LstmSequenceClassifier
    from app.services.holistic_service import keypoints_from_results

    bodies, _ = _stream(open_gate_server, hand_frames, "e2e-parity")
    last = bodies[-1]
    assert last["probabilities"] is not None

    model = LstmSequenceClassifier.from_h5(WEIGHTS)
    vecs, hands = [], []
    with mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as h:
        for png in hand_frames:
            rgb = cv2.resize(np.array(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8), (640, 480))
            rgb.flags.writeable = False
            v, hv = keypoints_from_results(h.process(rgb))
            vecs.append(v)
            hands.append(hv)
    window = np.stack(vecs[-30:])
    assert np.count_nonzero(window) > 0, "test input is all zeros - parity would be meaningless"
    assert sum(hands[-30:]) > 0, "no hand in the compared window"

    expected = model.predict_proba(window)
    got = np.array([last["probabilities"][k] for k in ("hello", "thanks", "iloveyou")])
    assert last["hands_in_window"] == sum(hands[-30:])
    assert not np.allclose(expected, [0.3631, 0.2136, 0.4233], atol=1e-3), "output equals the all-zeros answer"
    assert np.allclose(got, expected, atol=2e-3), (got, expected)
    print(f"\nservice probs {np.round(got, 4)}  independent probs {np.round(expected, 4)}  "
          f"hands_in_window={last['hands_in_window']}  nonzero_frac={np.count_nonzero(window)/window.size:.3f}")
