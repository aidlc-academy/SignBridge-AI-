"""
Central configuration for SignBridge AI backend.
Values can be overridden via environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).resolve().parent.parent   # backend/
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
DATASETS_DIR    = BASE_DIR / "datasets"
DATABASE_PATH   = BASE_DIR / "database" / "signbridge.db"

# ── API ────────────────────────────────────────────────────────────────────

API_VERSION     = "1.0.0"
APP_TITLE       = "SignBridge AI API"
APP_DESCRIPTION = "Real-time ASL gesture recognition and communication backend."

# Allow the frontend dev server and production origin.
# Override via CORS_ORIGINS env var (comma-separated).
_cors_env = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
CORS_ORIGINS: list[str] = [o.strip() for o in _cors_env.split(",") if o.strip()]

# ── Recognition ────────────────────────────────────────────────────────────

MODEL_FILENAME         = "gesture_classifier.joblib"
LABELS_FILENAME        = "class_labels.json"
DEFAULT_CONF_THRESHOLD = 0.70    # reject predictions below this
MAX_FRAME_SIZE_BYTES   = 2 * 1024 * 1024   # 2 MB

# ── LSTM word-sign recognition (additional, independent method) ────────────
# Source model: a 3-word LSTM (hello / thanks / iloveyou) trained on MediaPipe
# *Holistic* keypoints over 30-frame windows. The reference repository has NO
# license, so its weights are NOT copied into this project: point
# LSTM_WEIGHTS_PATH at a local copy (default: ../../lstm-reference/action.h5,
# i.e. a sibling folder of this project). If the file is absent the feature
# reports "model_unavailable" and nothing else is affected.
LSTM_ENABLED = os.environ.get("LSTM_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
LSTM_WEIGHTS_PATH = Path(
    os.environ.get("LSTM_WEIGHTS_PATH", str(BASE_DIR.parent.parent / "lstm-reference" / "action.h5"))
)
# Class names are NOT stored in the weights file. Order comes from the model's
# training notebook: label_map = {'hello': 0, 'thanks': 1, 'iloveyou': 2}.
LSTM_LABELS: list[str] = ["hello", "thanks", "iloveyou"]
LSTM_CONF_THRESHOLD = float(os.environ.get("LSTM_CONF_THRESHOLD", "0.8"))   # the model author's demo threshold
# Require a visible hand in at least this many of the last 30 frames. The
# original model confidently outputs a word even for an empty scene (see the
# integration report), so without this gate the API would fabricate words.
LSTM_MIN_HAND_FRAMES = int(os.environ.get("LSTM_MIN_HAND_FRAMES", "8"))
LSTM_SESSION_TTL_SECONDS = float(os.environ.get("LSTM_SESSION_TTL_SECONDS", "60"))
LSTM_MAX_SESSIONS = int(os.environ.get("LSTM_MAX_SESSIONS", "4"))
LSTM_RELOAD_RETRY_SECONDS = 30.0   # re-check for missing weights at most this often
# If two frames of one session are further apart than this, the 30-frame
# window is no longer "consecutive frames" -> drop it and start collecting again.
LSTM_GAP_RESET_SECONDS = float(os.environ.get("LSTM_GAP_RESET_SECONDS", "2.0"))
# A word is "confirmed" (emitted once, for the UI to append) when it wins
# LSTM_CONFIRM_VOTES of the last LSTM_CONFIRM_WINDOW predictions, then is
# suppressed for LSTM_CONFIRM_COOLDOWN_SECONDS (reuses StabilizationService).
LSTM_CONFIRM_WINDOW = 8
LSTM_CONFIRM_VOTES = 6
LSTM_CONFIRM_COOLDOWN_SECONDS = 2.5

# ── Supported vocabulary ───────────────────────────────────────────────────

SUPPORTED_GESTURES = [
    {"id": "asl_a",  "label": "A",  "description": "Fist with thumb on side",                    "handShape": "Fist, thumb lateral",          "language": "ASL", "category": "alphabet"},
    {"id": "asl_b",  "label": "B",  "description": "Flat hand, fingers up, thumb across palm",   "handShape": "Flat, thumb folded",            "language": "ASL", "category": "alphabet"},
    {"id": "asl_c",  "label": "C",  "description": "Curved hand like holding a cup",             "handShape": "Curved, open C",               "language": "ASL", "category": "alphabet"},
    {"id": "asl_d",  "label": "D",  "description": "Index up, others curl to touch thumb",       "handShape": "Index up, O-ring",             "language": "ASL", "category": "alphabet"},
    {"id": "asl_i",  "label": "I",  "description": "Pinky extended, others curled",              "handShape": "Pinky up, fist",               "language": "ASL", "category": "alphabet"},
    {"id": "asl_l",  "label": "L",  "description": "Index up, thumb out — L shape",              "handShape": "L-shape",                      "language": "ASL", "category": "alphabet"},
    {"id": "asl_o",  "label": "O",  "description": "All fingers curve to touch thumb",           "handShape": "O-ring, all fingers",          "language": "ASL", "category": "alphabet"},
    {"id": "asl_v",  "label": "V",  "description": "Index + middle extended (victory/peace)",    "handShape": "Two-finger V",                 "language": "ASL", "category": "alphabet"},
    {"id": "asl_y",  "label": "Y",  "description": "Thumb + pinky extended, others curled",      "handShape": "Thumb + pinky",                "language": "ASL", "category": "alphabet"},
    {"id": "asl_1",  "label": "1",  "description": "Index finger pointing up",                   "handShape": "Single index up",              "language": "ASL", "category": "numeral"},
]
