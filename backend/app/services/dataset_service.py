"""
Dataset-collection service for SignBridge AI (Phase 5 — data collection).

Turns captured webcam frames into labeled rows of normalized hand-landmark
features and appends them to ``datasets/landmarks.csv``. That CSV is the exact
input expected by ``app.ml.train_model`` (a ``label`` column followed by 63
feature columns ``x0,y0,z0 … x20,y20,z20``).

Honesty rules (no fabricated data):
  * a sample is only written when MediaPipe actually detects a hand in the
    frame; a frame with no hand returns status ``no_hand`` and writes nothing.
  * labels are validated against the supported vocabulary, so stray/empty
    labels never pollute the dataset.
  * the normalized feature vector written here is produced by the SAME
    ``VisionService`` pipeline used at inference time, so the training data and
    the runtime features are guaranteed to match.
"""
from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from app import config
from app.services.vision_service import FEATURE_SIZE, get_vision_service

logger = logging.getLogger(__name__)

# CSV column layout: label, x0, y0, z0, ... x20, y20, z20  (63 feature columns)
FEATURE_COLUMNS = [
    f"{axis}{i}" for i in range(FEATURE_SIZE // 3) for axis in ("x", "y", "z")
]
CSV_HEADER = ["label", *FEATURE_COLUMNS]

# Supported labels, derived from the single source of truth in config.
SUPPORTED_LABELS = [g["label"] for g in config.SUPPORTED_GESTURES]


class DatasetService:
    """Appends labeled normalized-landmark rows to a CSV and reports counts."""

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        vision_service_factory: Callable[[], object] = get_vision_service,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path else (config.DATASETS_DIR / "landmarks.csv")
        self._vision_service_factory = vision_service_factory
        self._lock = threading.Lock()

    # ── Internals ───────────────────────────────────────────────────────────

    def _ensure_file(self) -> None:
        """Create the parent dir and write the header row if the file is new."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADER)

    # ── Collection ──────────────────────────────────────────────────────────

    def add_sample(self, label: str, frame_bytes: bytes) -> dict:
        """
        Extract the normalized 63-float landmark vector from one frame and, if a
        hand is present and the label is valid, append it as a labeled row.

        Returns a dict with: status, label, message, total_for_label, total.
        Never writes a row unless a real hand was detected.
        """
        label = (label or "").strip()
        if label not in SUPPORTED_LABELS:
            return {
                "status": "invalid_label",
                "label": label,
                "message": (
                    f"Unsupported label '{label}'. "
                    f"Expected one of: {', '.join(SUPPORTED_LABELS)}."
                ),
                "total_for_label": self.count_for_label(label),
                "total": self.total(),
            }

        vision = self._vision_service_factory()
        result = vision.process_frame(frame_bytes)

        if result["status"] == "no_hand":
            return {
                "status": "no_hand",
                "label": label,
                "message": "No hand detected in frame; no sample recorded.",
                "total_for_label": self.count_for_label(label),
                "total": self.total(),
            }

        if result["status"] == "error":
            return {
                "status": "error",
                "label": label,
                "message": result["message"],
                "total_for_label": self.count_for_label(label),
                "total": self.total(),
            }

        features = np.asarray(result["features"], dtype=np.float32)
        if features.shape != (FEATURE_SIZE,) or not np.isfinite(features).all():
            return {
                "status": "error",
                "label": label,
                "message": (
                    f"Unexpected feature vector (shape={features.shape}, "
                    f"finite={bool(np.isfinite(features).all())})."
                ),
                "total_for_label": self.count_for_label(label),
                "total": self.total(),
            }

        with self._lock:
            self._ensure_file()
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([label, *(f"{v:.6f}" for v in features.tolist())])

        return {
            "status": "recorded",
            "label": label,
            "message": f"Recorded one '{label}' sample.",
            "total_for_label": self.count_for_label(label),
            "total": self.total(),
        }

    # ── Stats ─────────────────────────────────────────────────────────────

    def _read_labels(self) -> list[str]:
        if not self.csv_path.exists():
            return []
        with self._lock:
            with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return []
                return [row[0] for row in reader if row]

    def count_for_label(self, label: str) -> int:
        return sum(1 for lab in self._read_labels() if lab == label)

    def total(self) -> int:
        return len(self._read_labels())

    def stats(self) -> dict:
        """Per-label sample counts plus total, covering every supported label."""
        labels = self._read_labels()
        counts = {lab: 0 for lab in SUPPORTED_LABELS}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        return {
            "total": len(labels),
            "counts": counts,
            "labels": SUPPORTED_LABELS,
            "csv_path": str(self.csv_path),
            "ready_to_train": len(labels) > 0 and all(counts[l] > 0 for l in SUPPORTED_LABELS),
        }


# ── Module-level singleton ─────────────────────────────────────────────────

_dataset_service: Optional[DatasetService] = None
_dataset_lock = threading.Lock()


def get_dataset_service() -> DatasetService:
    global _dataset_service
    with _dataset_lock:
        if _dataset_service is None:
            _dataset_service = DatasetService()
        return _dataset_service
