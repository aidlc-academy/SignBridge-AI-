"""
Inference service: loads the trained gesture classifier and runs predictions.

If the model file is absent, all inference calls return a model_unavailable
status — no fake predictions are generated.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import TRAINED_MODELS_DIR, MODEL_FILENAME, LABELS_FILENAME, DEFAULT_CONF_THRESHOLD

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self) -> None:
        self._model = None
        self._labels: list[str] = []
        self._trained_at: Optional[datetime] = None
        self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        model_path  = TRAINED_MODELS_DIR / MODEL_FILENAME
        labels_path = TRAINED_MODELS_DIR / LABELS_FILENAME

        if not model_path.exists():
            logger.warning(
                "Gesture classifier not found at %s. "
                "Run backend/app/ml/train_model.py to train a model.",
                model_path,
            )
            return

        try:
            import joblib  # lazy import — only needed when model exists
            self._model = joblib.load(model_path)
            logger.info("Loaded gesture classifier from %s", model_path)

            if labels_path.exists():
                with labels_path.open() as f:
                    self._labels = json.load(f)
            else:
                # Attempt to read labels from sklearn classes_
                if hasattr(self._model, "classes_"):
                    self._labels = [str(c) for c in self._model.classes_]

            # Record modification time as proxy for training time
            mtime = model_path.stat().st_mtime
            self._trained_at = datetime.fromtimestamp(mtime, tz=timezone.utc)
            logger.info("Loaded %d class labels: %s", len(self._labels), self._labels)

        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            self._model = None

    # ── Status ─────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_labels(self) -> list[str]:
        return self._labels

    @property
    def trained_at(self) -> Optional[datetime]:
        return self._trained_at

    @property
    def model_type(self) -> Optional[str]:
        if self._model is None:
            return None
        return type(self._model).__name__

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(
        self,
        features: np.ndarray,
        threshold: float = DEFAULT_CONF_THRESHOLD,
    ) -> dict:
        """
        Predict the gesture class for a normalized landmark feature vector.

        Returns a dict with keys: gesture, confidence, status, timestamp.
        Never fabricates a result when the model is unavailable.
        """
        ts = datetime.now(timezone.utc)

        if self._model is None:
            return {
                "gesture":    None,
                "confidence": 0.0,
                "status":     "model_unavailable",
                "timestamp":  ts,
                "message":    (
                    "Recognition model not available. "
                    "Please train or load a model before running inference."
                ),
            }

        try:
            feature_row = features.reshape(1, -1)
            proba = self._model.predict_proba(feature_row)[0]
            best_idx  = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            label      = self._labels[best_idx] if self._labels else str(best_idx)

            if confidence < threshold:
                return {
                    "gesture":    None,
                    "confidence": confidence,
                    "status":     "low_confidence",
                    "timestamp":  ts,
                    "message":    f"Confidence {confidence:.2f} below threshold {threshold:.2f}.",
                }

            return {
                "gesture":    label,
                "confidence": confidence,
                "status":     "recognized",
                "timestamp":  ts,
                "message":    f"Recognized '{label}' with confidence {confidence:.2f}.",
            }

        except Exception as exc:
            logger.error("Inference error: %s", exc)
            return {
                "gesture":    None,
                "confidence": 0.0,
                "status":     "error",
                "timestamp":  ts,
                "message":    f"Inference failed: {exc}",
            }


# Module-level singleton — loaded once at import time
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
