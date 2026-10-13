"""
Verify MediaPipe returns real 21-point landmarks on actual hand photographs,
processed through the exact VisionService used by the FastAPI endpoint.
Also exercises decode_frame (OpenCV preprocessing) and both the raw-landmark
and normalized-feature pipelines.
"""
from __future__ import annotations

import glob
import sys

from app.services.vision_service import get_vision_service


def main() -> int:
    svc = get_vision_service()
    images = sorted(glob.glob("real_*.png") + glob.glob("real_*.jpg"))
    if not images:
        print("No real_*.png/jpg images present.")
        return 2

    any_detected = False
    for path in images:
        with open(path, "rb") as f:
            frame = f.read()

        # 1. Raw landmark pipeline (Phase 3 visualization endpoint)
        result = svc.process_frame_landmarks(frame)
        print(f"\n=== {path} ({len(frame)} bytes) ===")
        print("  status        :", result["status"])
        print("  hands_detected:", result["hands_detected"])

        if result["hands_detected"] > 0:
            any_detected = True
            hand = result["hands"][0]
            lms = hand["landmarks"]
            assert len(lms) == 21, f"expected 21 landmarks, got {len(lms)}"
            print("  handedness    :", hand["handedness"])
            print("  landmark count:", len(lms))
            print("  wrist   (lm0) :", {k: round(v, 4) for k, v in lms[0].items()})
            print("  index tip (8) :", {k: round(v, 4) for k, v in lms[8].items()})
            print("  pinky tip (20):", {k: round(v, 4) for k, v in lms[20].items()})
            # coords must be normalized image coords in [0,1] (x,y)
            assert 0.0 <= lms[0]["x"] <= 1.0 and 0.0 <= lms[0]["y"] <= 1.0

            # 2. Normalized feature pipeline (ML inference path)
            feat = svc.process_frame(frame)
            print("  feature status:", feat["status"])
            if feat["features"] is not None:
                print("  feature vector length:", len(feat["features"]))
                assert len(feat["features"]) == 63, "expected 63-float feature vector"

    if any_detected:
        print("\nREAL MEDIAPIPE LANDMARKS CONFIRMED (21 points, normalized coords).")
        return 0
    print("\nNo hands detected in any real image.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
