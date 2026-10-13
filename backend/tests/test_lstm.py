"""
Tests for the LSTM word-sign feature.

* Golden values below were produced by TensorFlow 2.15 / Keras on the real
  weights (outside this project) -- they are an independent check of the
  NumPy runtime, which must not depend on TensorFlow itself.
* Service-logic tests use a fake Holistic extractor and a fake decoder so the
  buffer / gate / confirmation behaviour is deterministic. The model itself is
  always the REAL model.
* Tests that need the weights file are skipped (not faked) if it is absent.
"""
from __future__ import annotations

import io
import json
import time

import numpy as np
import pytest

from app import config
from app.ml.lstm_model import LstmSequenceClassifier, UnsupportedModelError
from app.services.holistic_service import KEYPOINT_SIZE, keypoints_from_results
from app.services.lstm_service import LstmSequenceService

WEIGHTS = config.LSTM_WEIGHTS_PATH
needs_weights = pytest.mark.skipif(not WEIGHTS.is_file(), reason=f"LSTM weights not present at {WEIGHTS}")

# label order: ['hello', 'thanks', 'iloveyou']
GOLDEN = [
    (11, "uniform", [0.0005306437378749251, 1.519166926300386e-05, 0.9994540810585022]),
    (22, "uniform", [0.004637842532247305, 2.6538265956332907e-05, 0.9953356385231018]),
    (33, "uniform", [0.9729782938957214, 0.00042508397018536925, 0.026596687734127045]),
    (44, "normal3", [0.3093937337398529, 0.6906062960624695, 3.738775689043905e-08]),
]
GOLDEN_ZEROS = [0.36309298872947693, 0.21359999477863312, 0.42330703139305115]


def golden_input(seed: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(0, 1, size=(30, 1662)).astype("float32")
    return rng.normal(0, 3, size=(30, 1662)).astype("float32")


@pytest.fixture(scope="module")
def model():
    if not WEIGHTS.is_file():
        pytest.skip(f"LSTM weights not present at {WEIGHTS}")
    return LstmSequenceClassifier.from_h5(WEIGHTS)


# ───────────────────────── model runtime ─────────────────────────

@needs_weights
def test_model_architecture_read_from_weights(model):
    assert (model.sequence_length, model.feature_size, model.n_classes) == (30, 1662, 3)
    assert model.param_count == 596675


@needs_weights
@pytest.mark.parametrize("seed,kind,expected", GOLDEN)
def test_matches_tensorflow_golden_outputs(model, seed, kind, expected):
    got = model.predict_proba(golden_input(seed, kind))
    assert np.allclose(got, expected, atol=2e-4)
    assert int(np.argmax(got)) == int(np.argmax(expected))


@needs_weights
def test_zero_input_matches_golden_and_is_not_confident(model):
    got = model.predict_proba(np.zeros((30, 1662), np.float32))
    assert np.allclose(got, GOLDEN_ZEROS, atol=2e-4)
    assert got.max() < config.LSTM_CONF_THRESHOLD   # an empty vector must never clear the threshold


@needs_weights
def test_probabilities_are_valid_and_deterministic(model):
    x = golden_input(5, "uniform")
    a, b = model.predict_proba(x), model.predict_proba(x)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all() and a.sum() == pytest.approx(1.0, abs=1e-5) and (a >= 0).all()


@needs_weights
def test_wrong_shape_and_nonfinite_rejected(model):
    with pytest.raises(ValueError):
        model.predict_proba(np.zeros((29, 1662), np.float32))
    with pytest.raises(ValueError):
        model.predict_proba(np.zeros((30, 63), np.float32))
    bad = np.zeros((30, 1662), np.float32)
    bad[3, 5] = np.nan
    with pytest.raises(ValueError):
        model.predict_proba(bad)


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        LstmSequenceClassifier.from_h5(tmp_path / "nope.h5")


def test_non_keras_file_rejected(tmp_path):
    import h5py

    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("x", data=[1, 2, 3])
    with pytest.raises(UnsupportedModelError):
        LstmSequenceClassifier.from_h5(p)


def test_unsupported_layer_rejected(tmp_path):
    import h5py

    p = tmp_path / "conv.h5"
    cfg = {"class_name": "Sequential", "config": {"layers": [
        {"class_name": "Conv1D", "config": {"name": "c1", "batch_input_shape": [None, 30, 4]}}]}}
    with h5py.File(p, "w") as f:
        f.attrs["model_config"] = json.dumps(cfg)
        f.create_group("model_weights").create_group("c1").attrs["weight_names"] = np.array([], dtype="S1")
    with pytest.raises(UnsupportedModelError, match="Unsupported layer"):
        LstmSequenceClassifier.from_h5(p)


# ───────────────────────── keypoint layout ─────────────────────────

class _LM:
    def __init__(self, n, vis=False):
        self.landmark = [type("P", (), {"x": i * 0.001, "y": 0.5, "z": -0.1, "visibility": 0.9})() for i in range(n)]


class _Res:
    def __init__(self, pose=True, face=True, lh=False, rh=False):
        self.pose_landmarks = _LM(33) if pose else None
        self.face_landmarks = _LM(468) if face else None
        self.left_hand_landmarks = _LM(21) if lh else None
        self.right_hand_landmarks = _LM(21) if rh else None


def test_keypoint_vector_layout_matches_training_order():
    vec, hand = keypoints_from_results(_Res(pose=True, face=True, lh=True, rh=False))
    assert vec.shape == (1662,) and vec.dtype == np.float32 and hand is True
    pose, face, lh, rh = vec[:132], vec[132:1536], vec[1536:1599], vec[1599:]
    assert pose[3] == pytest.approx(0.9)            # pose carries visibility as 4th value
    assert face[3] == pytest.approx(0.001)          # face is x,y,z triplets: landmark 1 x
    assert face[4] == pytest.approx(0.5) and face[5] == pytest.approx(-0.1)
    assert np.count_nonzero(lh) > 0 and not rh.any()  # right hand absent -> zeros, left present


def test_missing_parts_are_zero_filled():
    vec, hand = keypoints_from_results(_Res(pose=False, face=False, lh=False, rh=False))
    assert not vec.any() and hand is False and vec.shape == (1662,)
    vec, hand = keypoints_from_results(_Res(pose=False, face=False, lh=False, rh=True))
    assert hand is True and vec[:1599].sum() == 0 and vec[1599:].any()


# ───────────────────────── service logic (fake extractor, real model) ─────────────────────────

class FakeExtractor:
    """Returns a scripted (vector, hand_visible) per call."""
    def __init__(self, script):
        self.script, self.calls, self.closed = script, 0, False

    def extract(self, rgb):
        v = self.script(self.calls)
        self.calls += 1
        return v

    def close(self):
        self.closed = True


def _decode(_b):
    return np.zeros((480, 640, 3), np.uint8)


def make_service(script, clock=None, **kw):
    ex = {"made": []}

    def factory():
        e = FakeExtractor(script)
        ex["made"].append(e)
        return e

    svc = LstmSequenceService(extractor_factory=factory, decoder=_decode, clock=clock or time.monotonic, **kw)
    svc._ex = ex
    return svc


def vec_for(seed, kind="uniform", hand=True):
    return golden_input(seed, kind)[0], hand


@pytest.fixture(autouse=True)
def _fast_config(monkeypatch):
    monkeypatch.setattr(config, "LSTM_ENABLED", True)


@needs_weights
def test_collects_30_frames_before_any_prediction():
    svc = make_service(lambda i: vec_for(i))
    for n in range(1, 30):
        r = svc.process_frame("s", b"x")
        assert r["status"] == "collecting" and r["frames_buffered"] == n
        assert r["word"] is None and r["probabilities"] is None and r["confidence"] == 0.0
    r = svc.process_frame("s", b"x")
    assert r["status"] != "collecting" and r["frames_buffered"] == 30


@needs_weights
def test_prediction_equals_real_model_on_the_buffered_window():
    # Feed exactly the golden window (seed 11 -> iloveyou, seed 33 -> hello) one frame at a time.
    for seed, expected_word in ((11, "iloveyou"), (33, "hello")):
        window = golden_input(seed, "uniform")
        svc = make_service(lambda i, w=window: (w[i], True))
        r = None
        for _ in range(30):
            r = svc.process_frame("s", b"x")
        assert r["status"] == "recognized" and r["word"] == expected_word
        assert r["confidence"] > config.LSTM_CONF_THRESHOLD
        exp = dict(zip(config.LSTM_LABELS, GOLDEN[[g[0] for g in GOLDEN].index(seed)][2]))
        for k, v in exp.items():
            assert r["probabilities"][k] == pytest.approx(v, abs=3e-4)


@needs_weights
def test_no_hand_gate_blocks_words_even_if_model_is_confident():
    window = golden_input(11, "uniform")           # model is 99.9% 'iloveyou' on this window
    svc = make_service(lambda i: (window[i], False))   # ...but no hand was ever visible
    for _ in range(30):
        r = svc.process_frame("s", b"x")
    assert r["status"] == "no_hand" and r["word"] is None and r["confirmed_word"] is None
    assert r["hands_in_window"] == 0 and r["probabilities"] is None


@needs_weights
def test_hand_gate_threshold_is_exact(monkeypatch):
    monkeypatch.setattr(config, "LSTM_MIN_HAND_FRAMES", 8)
    window = golden_input(11, "uniform")
    for hands, expect in ((7, "no_hand"), (8, "recognized")):
        svc = make_service(lambda i, h=hands: (window[i], i < h))
        for _ in range(30):
            r = svc.process_frame("s", b"x")
        assert r["status"] == expect and r["hands_in_window"] == hands


@needs_weights
def test_low_confidence_returns_probabilities_but_no_word():
    window = golden_input(44, "normal3")           # ~0.69 thanks -> below 0.8
    svc = make_service(lambda i: (window[i], True))
    for _ in range(30):
        r = svc.process_frame("s", b"x")
    assert r["status"] == "low_confidence" and r["word"] is None and r["confirmed_word"] is None
    assert 0.6 < r["confidence"] < 0.8 and r["probabilities"]["thanks"] == pytest.approx(r["confidence"])


@needs_weights
def test_word_is_confirmed_once_then_suppressed_by_cooldown():
    """Isolated test of the confirmation logic only: the model is replaced by a stub with a fixed output."""
    class StubModel:
        n_classes, sequence_length, feature_size, param_count = 3, 30, 1662, 0

        def predict_proba(self, seq):
            return np.array([0.03, 0.02, 0.95])

    svc = make_service(lambda i: vec_for(i))
    svc._model = StubModel()
    confirmed = []
    for _ in range(60):
        confirmed.append(svc.process_frame("s", b"x")["confirmed_word"])
    assert [c for c in confirmed if c] == ["iloveyou"]       # exactly once within the cooldown
    first = next(i for i, c in enumerate(confirmed) if c)
    assert first >= 29 + config.LSTM_CONFIRM_WINDOW - 1       # needs a full vote window after the 30-frame fill


@needs_weights
def test_sessions_are_isolated():
    w1, w2 = golden_input(11, "uniform"), golden_input(33, "uniform")
    scripts = {"a": w1, "b": w2}
    made = []
    svc = LstmSequenceService(extractor_factory=lambda: made.append(1) or FakeExtractor(lambda i: None), decoder=_decode)

    def run(sid, w):
        svc._get_session(sid).extractor.script = lambda i: (w[i], True)
        r = None
        for _ in range(30):
            r = svc.process_frame(sid, b"x")
        return r

    ra, rb = run("a", w1), run("b", w2)
    assert (ra["word"], rb["word"]) == ("iloveyou", "hello")
    assert svc.status()["active_sessions"] == 2


@needs_weights
def test_time_gap_resets_the_window():
    t = {"now": 0.0}
    svc = make_service(lambda i: vec_for(i), clock=lambda: t["now"])
    for _ in range(20):
        t["now"] += 0.1
        svc.process_frame("s", b"x")
    t["now"] += config.LSTM_GAP_RESET_SECONDS + 1     # camera stalled
    r = svc.process_frame("s", b"x")
    assert r["status"] == "collecting" and r["frames_buffered"] == 1


@needs_weights
def test_window_seconds_reports_real_time_span():
    t = {"now": 100.0}
    svc = make_service(lambda i: vec_for(i), clock=lambda: t["now"])
    r = None
    for _ in range(30):
        t["now"] += 0.125     # 8 fps like the frontend loop
        r = svc.process_frame("s", b"x")
    assert r["window_seconds"] == pytest.approx(29 * 0.125, abs=0.01)


@needs_weights
def test_idle_sessions_expire_and_lru_limit_enforced(monkeypatch):
    monkeypatch.setattr(config, "LSTM_MAX_SESSIONS", 2)
    t = {"now": 0.0}
    svc = make_service(lambda i: vec_for(i), clock=lambda: t["now"])
    for sid in ("a", "b", "c"):
        t["now"] += 1
        svc.process_frame(sid, b"x")
    assert svc.status()["active_sessions"] == 2
    assert svc._ex["made"][0].closed is True                # evicted session's Holistic graph was released
    t["now"] += config.LSTM_SESSION_TTL_SECONDS + 1
    svc.process_frame("d", b"x")
    assert svc.status()["active_sessions"] == 1


@needs_weights
def test_reset_session_clears_state_and_closes_extractor():
    svc = make_service(lambda i: vec_for(i))
    for _ in range(5):
        svc.process_frame("s", b"x")
    assert svc.reset_session("s") is True and svc._ex["made"][0].closed
    assert svc.reset_session("s") is False
    assert svc.process_frame("s", b"x")["frames_buffered"] == 1


@needs_weights
def test_extractor_or_decoder_failure_reports_error_not_a_word():
    def boom(_b):
        raise ValueError("cannot decode")

    svc = LstmSequenceService(extractor_factory=lambda: FakeExtractor(lambda i: vec_for(i)), decoder=boom)
    r = svc.process_frame("s", b"junk")
    assert r["status"] == "error" and r["word"] is None and "cannot decode" in r["message"]


def test_missing_weights_report_model_unavailable_and_never_extract(tmp_path):
    ex = FakeExtractor(lambda i: vec_for(i))
    svc = LstmSequenceService(weights_path=tmp_path / "missing.h5", extractor_factory=lambda: ex, decoder=_decode)
    r = svc.process_frame("s", b"x")
    assert r["status"] == "model_unavailable" and r["word"] is None and r["confidence"] == 0.0
    assert "not found" in r["message"] and ex.calls == 0
    st = svc.status()
    assert st["loaded"] is False and "LSTM_WEIGHTS_PATH" in st["message"]


def test_disabled_flag_reports_model_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LSTM_ENABLED", False)
    svc = LstmSequenceService(extractor_factory=lambda: FakeExtractor(lambda i: vec_for(i)), decoder=_decode)
    assert svc.process_frame("s", b"x")["status"] == "model_unavailable"
    assert svc.status()["enabled"] is False


@needs_weights
def test_label_count_mismatch_is_rejected():
    svc = LstmSequenceService(labels=["only", "two"], extractor_factory=lambda: FakeExtractor(lambda i: vec_for(i)), decoder=_decode)
    assert svc.is_loaded is False and "labels" in svc.status()["message"]


@needs_weights
def test_weights_appearing_later_are_picked_up(monkeypatch, tmp_path):
    import shutil

    target = tmp_path / "action.h5"
    t = {"now": 0.0}
    svc = LstmSequenceService(weights_path=target, extractor_factory=lambda: FakeExtractor(lambda i: vec_for(i)),
                              decoder=_decode, clock=lambda: t["now"])
    assert svc.process_frame("s", b"x")["status"] == "model_unavailable"
    shutil.copy(WEIGHTS, target)                     # copy lives in tmp_path only
    t["now"] += config.LSTM_RELOAD_RETRY_SECONDS + 1
    assert svc.process_frame("s", b"x")["status"] == "collecting"
