"""
LSTM word-sign recognition service (additional recognition method).

Independent of the CNN service and of the landmark/RandomForest path: it owns
its own MediaPipe Holistic extractor and a per-session 30-frame buffer.

Honesty rules (no fabricated results):
  * model weights missing / unreadable      -> status "model_unavailable"
  * fewer than 30 consecutive frames buffered -> status "collecting" (no word)
  * too few frames with a visible hand       -> status "no_hand"     (no word)
  * best probability below the threshold     -> status "low_confidence" (no word)
  * a word is only returned when the real model probability clears the threshold
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Optional

import numpy as np

from app import config
from app.ml.lstm_model import LstmSequenceClassifier, UnsupportedModelError
from app.services.stabilization_service import StabilizationService

logger = logging.getLogger(__name__)


@dataclass
class _Session:
    session_id: str
    extractor: object
    frames: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    hands: Deque[bool] = field(default_factory=lambda: deque(maxlen=30))
    stamps: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    stabilizer: Optional[StabilizationService] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_seen: float = 0.0

    def clear_window(self) -> None:
        self.frames.clear()
        self.hands.clear()
        self.stamps.clear()
        if self.stabilizer:
            self.stabilizer.reset()


def _default_decoder(data: bytes) -> np.ndarray:
    from app.services.vision_service import get_vision_service

    return get_vision_service().decode_frame(data)


def _default_extractor_factory():
    from app.services.holistic_service import HolisticExtractor

    return HolisticExtractor()


class LstmSequenceService:
    def __init__(
        self,
        weights_path: Optional[Path] = None,
        labels: Optional[list[str]] = None,
        extractor_factory: Callable[[], object] = _default_extractor_factory,
        decoder: Callable[[bytes], np.ndarray] = _default_decoder,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.weights_path = Path(weights_path or config.LSTM_WEIGHTS_PATH)
        self.labels = list(labels or config.LSTM_LABELS)
        self._extractor_factory = extractor_factory
        self._decoder = decoder
        self._clock = clock
        self._model: Optional[LstmSequenceClassifier] = None
        self._load_error: Optional[str] = None
        self._last_load_attempt: float = float("-inf")
        self._model_lock = threading.Lock()
        self._sessions: "OrderedDict[str, _Session]" = OrderedDict()
        self._sessions_lock = threading.Lock()
        self._try_load(force=True)

    # --------------------------------------------------------------- model
    def _try_load(self, force: bool = False) -> None:
        if not config.LSTM_ENABLED:
            self._load_error = "LSTM recognition is disabled (LSTM_ENABLED=0)."
            return
        with self._model_lock:
            if self._model is not None:
                return
            now = self._clock()
            if not force and now - self._last_load_attempt < config.LSTM_RELOAD_RETRY_SECONDS:
                return
            self._last_load_attempt = now
            try:
                model = LstmSequenceClassifier.from_h5(self.weights_path)
                if model.n_classes != len(self.labels):
                    raise UnsupportedModelError(
                        f"Model has {model.n_classes} outputs but {len(self.labels)} labels are configured."
                    )
                if model.sequence_length != 30:
                    raise UnsupportedModelError(f"Unexpected sequence length {model.sequence_length}.")
                from app.services.holistic_service import KEYPOINT_SIZE

                if model.feature_size != KEYPOINT_SIZE:
                    raise UnsupportedModelError(
                        f"Model expects {model.feature_size} features, Holistic extractor produces {KEYPOINT_SIZE}."
                    )
                self._model = model
                self._load_error = None
                logger.info("Loaded LSTM word model from %s", self.weights_path)
            except FileNotFoundError:
                self._load_error = (
                    f"LSTM weights not found at {self.weights_path}. "
                    "Set LSTM_WEIGHTS_PATH to a local copy of the model file."
                )
                logger.warning(self._load_error)
            except Exception as exc:  # unreadable / unsupported file
                self._load_error = f"LSTM model could not be loaded: {exc}"
                logger.error(self._load_error)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        self._try_load()
        if self._model is not None:
            msg = "LSTM word model loaded and ready."
        else:
            msg = self._load_error or "LSTM word model is not loaded."
        return {
            "loaded": self._model is not None,
            "enabled": bool(config.LSTM_ENABLED),
            "labels": self.labels,
            "sequence_length": 30,
            "feature_size": self._model.feature_size if self._model else None,
            "parameters": self._model.param_count if self._model else None,
            "confidence_threshold": config.LSTM_CONF_THRESHOLD,
            "active_sessions": len(self._sessions),
            "message": msg,
        }

    # ------------------------------------------------------------ sessions
    def _new_session(self, session_id: str) -> _Session:
        return _Session(
            session_id=session_id,
            extractor=self._extractor_factory(),
            stabilizer=StabilizationService(
                window_size=config.LSTM_CONFIRM_WINDOW,
                required_votes=config.LSTM_CONFIRM_VOTES,
                cooldown_seconds=config.LSTM_CONFIRM_COOLDOWN_SECONDS,
            ),
        )

    def _close_session(self, s: _Session) -> None:
        close = getattr(s.extractor, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover
                logger.debug("extractor close failed", exc_info=True)

    def _get_session(self, session_id: str) -> _Session:
        now = self._clock()
        with self._sessions_lock:
            # expire idle sessions
            for sid in [k for k, v in self._sessions.items() if now - v.last_seen > config.LSTM_SESSION_TTL_SECONDS]:
                self._close_session(self._sessions.pop(sid))
            s = self._sessions.get(session_id)
            if s is None:
                while len(self._sessions) >= config.LSTM_MAX_SESSIONS:   # evict least recently used
                    _, old = self._sessions.popitem(last=False)
                    self._close_session(old)
                s = self._new_session(session_id)
                self._sessions[session_id] = s
            s.last_seen = now
            self._sessions.move_to_end(session_id)
            return s

    def reset_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            s = self._sessions.pop(session_id, None)
        if s is None:
            return False
        with s.lock:
            self._close_session(s)
        return True

    def close(self) -> None:
        with self._sessions_lock:
            sessions, self._sessions = list(self._sessions.values()), OrderedDict()
        for s in sessions:
            self._close_session(s)

    # ------------------------------------------------------------- process
    def _result(self, session_id: str, status: str, message: str, **kw) -> dict:
        base = {
            "status": status,
            "word": None,
            "confidence": 0.0,
            "confirmed_word": None,
            "probabilities": None,
            "frames_buffered": 0,
            "sequence_length": 30,
            "hands_in_window": 0,
            "window_seconds": None,
            "message": message,
            "session_id": session_id,
        }
        base.update(kw)
        return base

    def process_frame(self, session_id: str, frame_bytes: bytes) -> dict:
        self._try_load()
        if self._model is None:
            return self._result(session_id, "model_unavailable", self._load_error or "LSTM model is not loaded.")

        session = self._get_session(session_id)
        with session.lock:
            try:
                rgb = self._decoder(frame_bytes)
                vec, hand_visible = session.extractor.extract(rgb)
            except Exception as exc:
                logger.error("LSTM frame processing failed: %s", exc)
                return self._result(
                    session_id, "error", f"Frame processing failed: {exc}",
                    frames_buffered=len(session.frames),
                )

            now = self._clock()
            if session.stamps and now - session.stamps[-1] > config.LSTM_GAP_RESET_SECONDS:
                session.clear_window()   # frames are no longer consecutive

            session.frames.append(vec)
            session.hands.append(bool(hand_visible))
            session.stamps.append(now)

            n = len(session.frames)
            hands_n = int(sum(session.hands))
            span = float(session.stamps[-1] - session.stamps[0]) if n > 1 else 0.0
            common = dict(frames_buffered=n, hands_in_window=hands_n, window_seconds=round(span, 2))

            if n < 30:
                return self._result(session_id, "collecting", f"Collecting frames ({n}/30).", **common)

            if hands_n < config.LSTM_MIN_HAND_FRAMES:
                session.stabilizer.push(None)
                return self._result(
                    session_id, "no_hand",
                    f"Hand visible in only {hands_n} of the last 30 frames "
                    f"(need {config.LSTM_MIN_HAND_FRAMES}).",
                    **common,
                )

            try:
                probs = self._model.predict_proba(np.stack(session.frames))
            except Exception as exc:
                logger.error("LSTM inference failed: %s", exc)
                return self._result(session_id, "error", f"Inference failed: {exc}", **common)

            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            prob_map = {lab: round(float(p), 6) for lab, p in zip(self.labels, probs)}

            if conf < config.LSTM_CONF_THRESHOLD:
                session.stabilizer.push(None)
                return self._result(
                    session_id, "low_confidence",
                    f"Confidence {conf:.2f} below threshold {config.LSTM_CONF_THRESHOLD:.2f}.",
                    confidence=round(conf, 6), probabilities=prob_map, **common,
                )

            word = self.labels[idx]
            confirmed = session.stabilizer.push(word)
            return self._result(
                session_id, "recognized", f"Recognized '{word}' with confidence {conf:.2f}.",
                word=word, confidence=round(conf, 6), confirmed_word=confirmed,
                probabilities=prob_map, **common,
            )


_lstm_service: Optional[LstmSequenceService] = None
_lstm_lock = threading.Lock()


def get_lstm_service() -> LstmSequenceService:
    global _lstm_service
    with _lstm_lock:
        if _lstm_service is None:
            _lstm_service = LstmSequenceService()
        return _lstm_service


def shutdown_lstm_service() -> None:
    """Release Holistic graphs on app shutdown (no-op if the service was never used)."""
    global _lstm_service
    with _lstm_lock:
        svc, _lstm_service = _lstm_service, None
    if svc is not None:
        svc.close()
