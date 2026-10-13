"""
Feature preprocessing for the gesture classifier.

This module defines the feature extraction transform used by both
train_model.py (offline) and vision_service.py (online inference)
to ensure they are identical.
"""
from __future__ import annotations

import numpy as np

FEATURE_SIZE = 63   # 21 landmarks × 3 coordinates


def normalize_landmarks(raw_landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize a (21, 3) landmark array so that:
    - Landmark 0 (wrist) is at the origin
    - Scale is normalized by wrist-to-MCP (landmark 9) distance

    Returns a (63,) float32 feature vector.
    """
    if raw_landmarks.shape != (21, 3):
        raise ValueError(f"Expected (21, 3) landmark array, got {raw_landmarks.shape}")

    lm = raw_landmarks.astype(np.float32).copy()
    lm -= lm[0]   # translate to wrist origin

    scale = float(np.linalg.norm(lm[9]))
    if scale > 1e-6:
        lm /= scale

    return lm.flatten()


def landmarks_from_mediapipe(hand_landmarks) -> np.ndarray:
    """
    Convert a MediaPipe NormalizedLandmarkList to a (21, 3) numpy array.
    Accepts either the solutions.hands result or the tasks.vision result format.
    """
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
