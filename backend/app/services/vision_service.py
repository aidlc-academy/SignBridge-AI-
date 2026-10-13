"""
Vision service: decodes JPEG frames from the browser, runs MediaPipe
hand detection, extracts + normalizes 21-landmark features.

The browser owns camera capture. This service only processes frames
it receives as bytes — it never opens cv2.VideoCapture.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy imports (avoid crash if packages absent at scaffold time) ──────────

def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:
        raise RuntimeError(
            "opencv-python-headless is not installed. "
            "Run: pip install -r requirements.txt"
        ) from e


def _import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except ImportError as e:
        raise RuntimeError(
            "mediapipe is not installed. "
            "Run: pip install -r requirements.txt"
        ) from e


def _import_pil():
    try:
        from PIL import Image
        return Image
    except ImportError as e:
        raise RuntimeError(
            "Pillow is not installed. "
            "Run: pip install -r requirements.txt"
        ) from e


# Number of landmarks × coordinates per landmark
LANDMARK_COUNT   = 21
COORDS_PER_POINT = 3   # x, y, z
FEATURE_SIZE     = LANDMARK_COUNT * COORDS_PER_POINT   # 63


class VisionService:
    """
    Stateless service that processes webcam frames for two purposes:
    1. ML inference: normalized 63-float feature vector (existing)
    2. Phase 3 visualization: raw landmark coordinates for all hands (new)
    """

    def decode_frame(self, frame_bytes: bytes) -> np.ndarray:
        """Decode JPEG bytes → RGB numpy array (H, W, 3)."""
        cv2 = _import_cv2()
        Image = _import_pil()

        # Use Pillow for robust JPEG decoding
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        # Resize to 640×480 for consistent processing
        arr = cv2.resize(arr, (640, 480))
        return arr

    def extract_landmarks(
        self, rgb_frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Run MediaPipe Hands on the RGB frame and extract landmarks for the
        first detected hand. Returns a (63,) float32 array or None if no
        hand is detected.
        """
        mp = _import_mediapipe()
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        ) as hands:
            results = hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )   # shape: (21, 3)

        return self._normalize(landmarks)

    def _normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmark coordinates relative to the wrist (landmark 0).
        """
        origin = landmarks[0].copy()
        landmarks = landmarks - origin
        scale = float(np.linalg.norm(landmarks[9]))
        if scale > 1e-6:
            landmarks = landmarks / scale
        return landmarks.flatten()   # (63,)

    def extract_raw_landmarks(self, rgb_frame: np.ndarray) -> dict:
        """
        Run MediaPipe Hands on the RGB frame and return raw (unnormalized)
        landmark coordinates for ALL detected hands (up to 2).

        Returns:
            {
                "hands_detected": int,
                "hands": [
                    {
                        "handedness": "Left" | "Right" | None,
                        "landmarks": [{"x": float, "y": float, "z": float}, ...] (21 items)
                    },
                    ...
                ]
            }
        """
        mp = _import_mediapipe()
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            results = hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return {"hands_detected": 0, "hands": []}

        hands_data = []
        num_hands = len(results.multi_hand_landmarks)

        for i in range(num_hands):
            hand_lms = results.multi_hand_landmarks[i]
            # Get handedness label if available
            handedness = None
            if results.multi_handedness and i < len(results.multi_handedness):
                handedness = results.multi_handedness[i].classification[0].label

            landmarks = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in hand_lms.landmark
            ]

            hands_data.append({
                "handedness": handedness,
                "landmarks": landmarks,
            })

        return {"hands_detected": num_hands, "hands": hands_data}

    def process_frame(self, frame_bytes: bytes) -> dict:
        """
        Full pipeline for ML inference: bytes → decoded frame → normalized feature vector.
        """
        try:
            rgb = self.decode_frame(frame_bytes)
            features = self.extract_landmarks(rgb)

            if features is None:
                return {
                    "features": None,
                    "status":   "no_hand",
                    "message":  "No hand detected in frame.",
                }

            return {
                "features": features,
                "status":   "ok",
                "message":  "Landmarks extracted successfully.",
            }

        except Exception as exc:
            logger.error("Vision pipeline error: %s", exc)
            return {
                "features": None,
                "status":   "error",
                "message":  f"Vision processing failed: {exc}",
            }

    def self_check(self) -> dict:
        """
        Diagnostic: verify the vision stack can initialize without a frame.
        Confirms cv2, Pillow and MediaPipe import and that a MediaPipe Hands
        graph can be constructed. Surfaced via /api/vision/status so a
        MediaPipe initialization failure is visible without sending a frame.
        """
        report: dict = {
            "opencv": None,
            "mediapipe": None,
            "pillow": None,
            "hands_init": False,
            "status": "ok",
            "message": "Vision stack initialized successfully.",
        }
        try:
            cv2 = _import_cv2()
            report["opencv"] = getattr(cv2, "__version__", "unknown")

            Image = _import_pil()
            report["pillow"] = getattr(Image, "__version__", "unknown")

            mp = _import_mediapipe()
            report["mediapipe"] = getattr(mp, "__version__", "unknown")

            # Constructing the Hands graph is where a bad MediaPipe install or
            # a missing model asset would raise.
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5,
            ):
                report["hands_init"] = True

        except Exception as exc:
            logger.error("Vision self-check failed: %s", exc)
            report["status"] = "error"
            report["message"] = f"Vision stack initialization failed: {exc}"

        return report

    def process_frame_landmarks(self, frame_bytes: bytes) -> dict:
        """
        Phase 3 pipeline: bytes → decoded frame → raw landmark coordinates.
        Used for frontend visualization (landmark drawing).
        """
        try:
            rgb = self.decode_frame(frame_bytes)
            result = self.extract_raw_landmarks(rgb)

            if result["hands_detected"] == 0:
                return {
                    **result,
                    "status": "no_hand",
                    "message": "No hand detected in frame.",
                }

            return {
                **result,
                "status": "hand_detected",
                "message": f"{result['hands_detected']} hand(s) detected.",
            }

        except Exception as exc:
            logger.error("Vision landmark pipeline error: %s", exc)
            return {
                "hands_detected": 0,
                "hands": [],
                "status": "error",
                "message": f"Vision processing failed: {exc}",
            }


_vision_service: Optional[VisionService] = None


def get_vision_service() -> VisionService:
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service
