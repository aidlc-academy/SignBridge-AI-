"""
Holistic keypoint extraction for the LSTM word-sign model.

The LSTM was trained on one 1662-value vector per frame, produced by MediaPipe
*Holistic* in this exact order (zeros for any body part that is not detected):

    pose  : 33 landmarks x (x, y, z, visibility) = 132
    face  : 468 landmarks x (x, y, z)            = 1404
    left hand  : 21 landmarks x (x, y, z)        = 63
    right hand : 21 landmarks x (x, y, z)        = 63

The existing Hands-based pipeline (``vision_service``) is NOT sufficient: it can
fill at most 126 of those 1662 values (7.6%). This module is therefore a
separate extractor; the existing MediaPipe Hands code is left untouched.

One ``HolisticExtractor`` owns one Holistic graph, which keeps temporal tracking
state, so use one instance per camera session and never share it between threads.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

POSE_SIZE = 33 * 4
FACE_SIZE = 468 * 3
HAND_SIZE = 21 * 3
KEYPOINT_SIZE = POSE_SIZE + FACE_SIZE + 2 * HAND_SIZE   # 1662


def keypoints_from_results(results) -> Tuple[np.ndarray, bool]:
    """Build the 1662-vector from a MediaPipe Holistic ``results`` object.

    Returns (vector float32 shape (1662,), hand_visible) where ``hand_visible``
    is True if either hand was detected in this frame.
    """
    def flat(landmarks, with_visibility: bool, size: int) -> np.ndarray:
        if landmarks is None:
            return np.zeros(size, dtype=np.float32)
        if with_visibility:
            arr = [[p.x, p.y, p.z, p.visibility] for p in landmarks.landmark]
        else:
            arr = [[p.x, p.y, p.z] for p in landmarks.landmark]
        out = np.asarray(arr, dtype=np.float32).reshape(-1)
        if out.size != size:   # defensive: never emit a wrongly-sized vector
            raise ValueError(f"Unexpected landmark count: got {out.size} values, expected {size}")
        return out

    vec = np.concatenate(
        [
            flat(results.pose_landmarks, True, POSE_SIZE),
            flat(results.face_landmarks, False, FACE_SIZE),
            flat(results.left_hand_landmarks, False, HAND_SIZE),
            flat(results.right_hand_landmarks, False, HAND_SIZE),
        ]
    )
    hand_visible = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
    return vec, hand_visible


class HolisticExtractor:
    """Wraps one MediaPipe Holistic graph (video/tracking mode)."""

    def __init__(self) -> None:
        import mediapipe as mp  # lazy: heavy import

        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, rgb_frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """rgb_frame: (H, W, 3) uint8 RGB -> (1662-vector, hand_visible)."""
        rgb_frame.flags.writeable = False
        results = self._holistic.process(rgb_frame)
        return keypoints_from_results(results)

    def close(self) -> None:
        try:
            self._holistic.close()
        except Exception:  # pragma: no cover - best effort
            logger.debug("Holistic close failed", exc_info=True)
