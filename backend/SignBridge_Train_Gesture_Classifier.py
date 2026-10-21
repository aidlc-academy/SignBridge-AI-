# -*- coding: utf-8 -*-
"""SignBridge_Train_Gesture_Classifier.ipynb

# SignBridge AI — Gesture Classifier Training (Colab)

Trains a scikit-learn `RandomForestClassifier` on 63-float hand-landmark
features and exports three artifacts that the FastAPI backend loads WITHOUT any
code changes:

    trained_models/gesture_classifier.joblib   (joblib-dumped sklearn estimator)
    trained_models/class_labels.json           (labels in clf.classes_ order)
    trained_models/evaluation_report.json       (metrics)

Backend contract (do not deviate):
  * Feature vector = 21 MediaPipe Hand landmarks x (x,y,z) = 63 floats, flattened.
  * Normalization (identical to backend app/ml/preprocessing.py):
        lm -= lm[0]                 # wrist (idx 0) -> origin
        s = ||lm[9]||               # wrist-to-middle-MCP distance
        if s > 1e-6: lm /= s
  * CSV header: label, x0,y0,z0, ..., x20,y20,z20
  * class_labels.json MUST equal le.inverse_transform(clf.classes_) so that
    backend's  label = class_labels[argmax(predict_proba)]  is correct.
  * sklearn pinned to 1.5.2 to match the backend runtime (pickle compatibility).

HOW TO USE
  1. Run Cell 1 (install) once, then use Runtime > Restart runtime, then run
     Cell 2 onward. (We do NOT os.kill the runtime.)
  2. Pick a data source in Cell 3: 'csv' (Path A) or 'images' (Path B).
  3. Run through to the end; the final cell downloads a zip for the backend.
"""

# =============================================================================
# CELL 1 — Install pinned dependencies (RUN ONCE, THEN RESTART RUNTIME)
# =============================================================================
# After this cell finishes, use the menu: Runtime > Restart runtime.
# Then continue from Cell 2. Do NOT os.kill the process — it loses state and
# makes the notebook non-reproducible.
!pip -q install scikit-learn==1.5.2 joblib==1.4.2 numpy==1.26.4 \
    mediapipe==0.10.14 opencv-python==4.10.0.84 kagglehub


# =============================================================================
# CELL 2 — Imports and version check
# =============================================================================
import os
import csv
import json
from datetime import datetime, timezone

import numpy as np
import joblib
import sklearn

print("scikit-learn:", sklearn.__version__, "(must be 1.5.2 for backend pickle compatibility)")
print("numpy       :", np.__version__)
assert sklearn.__version__ == "1.5.2", (
    "sklearn must be 1.5.2 — run Cell 1 then Runtime > Restart runtime."
)


# =============================================================================
# CELL 3 — Configuration
# =============================================================================
# The 10 ASL static handshapes the backend supports.
CLASS_LABELS_10 = ["A", "B", "C", "D", "I", "L", "O", "V", "Y", "1"]

# "csv"    -> Path A: upload a landmarks.csv already produced by the backend's
#             POST /api/dataset/sample (rows are ALREADY normalized).
# "images" -> Path B: build landmarks.csv from raw images in data/<LABEL>/*.jpg
#             by running MediaPipe + the backend normalization here.
DATA_SOURCE = "images"       # "csv" or "images"

CSV_PATH = "landmarks.csv"
RAW_IMAGES_DIR = "data"
MAX_IMAGES_PER_CLASS = 300   # cap per class to keep extraction fast (Path B)

RANDOM_STATE = 42
N_ESTIMATORS = 200
TEST_SIZE = 0.2

OUTPUT_DIR = "trained_models"


# =============================================================================
# CELL 4 — Shared normalization (IDENTICAL to backend app/ml/preprocessing.py)
# =============================================================================
def normalize_landmarks(raw_landmarks: np.ndarray) -> np.ndarray:
    """(21,3) raw MediaPipe landmarks -> (63,) wrist-normalized float32 vector."""
    if raw_landmarks.shape != (21, 3):
        raise ValueError(f"Expected (21, 3), got {raw_landmarks.shape}")
    lm = raw_landmarks.astype(np.float32).copy()
    lm -= lm[0]                       # translate wrist (idx 0) to origin
    scale = float(np.linalg.norm(lm[9]))  # wrist-to-middle-MCP distance
    if scale > 1e-6:
        lm /= scale
    return lm.flatten()


CSV_HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


# =============================================================================
# CELL 5A — (Path A) Upload a pre-built landmarks.csv
# =============================================================================
if DATA_SOURCE == "csv":
    from google.colab import files
    print("Upload your 'landmarks.csv' (produced by POST /api/dataset/sample):")
    uploaded = files.upload()
    # Accept any uploaded name; normalize to CSV_PATH.
    name = next(iter(uploaded))
    if name != CSV_PATH:
        os.replace(name, CSV_PATH)
    print(f"Using dataset: {CSV_PATH}")
else:
    print("DATA_SOURCE != 'csv' — skip this cell, run the Path B cells instead.")


# =============================================================================
# CELL 5B — (Path B, optional) Download an ASL image dataset from Kaggle
# =============================================================================
# NOTE: the classic 'asl-alphabet' dataset has A-Z but NO digit "1".
# If you need "1", add your own images under data/1/ (e.g. collected via the
# backend's camera tool) OR a Kaggle ASL-digits dataset. Classes with zero
# detected samples are simply absent from the model and from class_labels.json.
if DATA_SOURCE == "images":
    import shutil
    import kagglehub

    print("Downloading ASL Alphabet dataset from Kaggle...")
    kaggle_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    print("Downloaded to:", kaggle_path)

    # Locate the training folder (nesting varies between dataset versions).
    train_src = None
    for root, dirs, _ in os.walk(kaggle_path):
        if os.path.basename(root).lower() == "asl_alphabet_train":
            # Prefer the deepest folder that directly contains class subdirs.
            if any(os.path.isdir(os.path.join(root, d)) for d in dirs):
                train_src = root
                break
    print("Train source:", train_src)

    os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
    for label in CLASS_LABELS_10:
        dest = os.path.join(RAW_IMAGES_DIR, label)
        os.makedirs(dest, exist_ok=True)
        src = os.path.join(train_src, label) if train_src else ""
        if src and os.path.isdir(src):
            names = sorted(os.listdir(src))[:MAX_IMAGES_PER_CLASS]
            for n in names:
                shutil.copy(os.path.join(src, n), os.path.join(dest, n))
            print(f"  {label}: copied {len(names)} images")
        else:
            print(f"  {label}: NOT in this Kaggle dataset "
                  f"(add images to {dest}/ manually if you need it)")
else:
    print("DATA_SOURCE != 'images' — skip this cell.")


# =============================================================================
# CELL 5C — (Path B) Extract + normalize landmarks -> landmarks.csv
# =============================================================================
if DATA_SOURCE == "images":
    import cv2
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    rows, processed, skipped = [], 0, 0
    for label in CLASS_LABELS_10:
        label_dir = os.path.join(RAW_IMAGES_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        for img_name in sorted(os.listdir(label_dir)):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            img = cv2.imread(os.path.join(label_dir, img_name))
            if img is None:
                skipped += 1
                continue
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks:
                skipped += 1          # no hand detected -> never fabricate a row
                continue
            raw = np.array(
                [[lm.x, lm.y, lm.z] for lm in res.multi_hand_landmarks[0].landmark],
                dtype=np.float32,
            )
            feat = normalize_landmarks(raw)
            rows.append([label] + [f"{v:.6f}" for v in feat.tolist()])
            processed += 1
    hands.close()

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)
    print(f"Extraction done. processed={processed} skipped={skipped} -> {CSV_PATH}")
else:
    print("DATA_SOURCE != 'images' — skip this cell.")


# =============================================================================
# CELL 6 — Load dataset (stdlib csv; no pandas, matching the backend trainer)
# =============================================================================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"{CSV_PATH} not found. For Path A upload the CSV; for Path B run the "
        f"extraction cell first."
    )

X_rows, y_rows = [], []
with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    if header is None or header[0] != "label":
        raise ValueError("CSV must start with a 'label' column header.")
    if len(header) - 1 != 63:
        raise ValueError(f"Expected 63 feature columns, found {len(header) - 1}.")
    for row in reader:
        if not row:
            continue
        y_rows.append(row[0])
        X_rows.append([float(v) for v in row[1:]])

if not X_rows:
    raise ValueError("CSV has a header but no data rows.")

X = np.asarray(X_rows, dtype=np.float32)
y = np.asarray(y_rows)

present = sorted(set(y))
print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, {len(present)} classes present: {present}")

missing = [c for c in CLASS_LABELS_10 if c not in present]
if missing:
    print(f"[WARNING] These target classes have NO samples and will NOT be "
          f"predictable: {missing}")

uniq, cnt = np.unique(y, return_counts=True)
for lbl, c in zip(uniq, cnt):
    if c < 20:
        print(f"[WARNING] Class '{lbl}' has only {c} samples (< 20).")
if len(y) < 200:
    print(f"[WARNING] Total samples {len(y)} < 200 — accuracy may be unreliable.")


# =============================================================================
# CELL 7 — Encode labels (ONLY on classes actually present)
# =============================================================================
from sklearn.preprocessing import LabelEncoder

# IMPORTANT: fit on the labels actually in the data, NOT on the fixed 10-label
# list. The backend maps argmax(predict_proba) -> class_labels[idx], so the
# exported labels must match the model's real class order exactly.
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Model classes (index order):", list(le.classes_))


# =============================================================================
# CELL 8 — Train + evaluate
# =============================================================================
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE
)
print(f"Train={len(X_train)}  Test={len(X_test)}")

clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy")
print(f"5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_acc:.4f}\n")

# target_names must follow le.classes_ order.
report = classification_report(
    y_test, y_pred, target_names=le.classes_, output_dict=True
)
print(classification_report(y_test, y_pred, target_names=le.classes_))
conf_mat = confusion_matrix(y_test, y_pred).tolist()


# =============================================================================
# CELL 9 — Export backend-compatible artifacts
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) model
joblib.dump(clf, os.path.join(OUTPUT_DIR, "gesture_classifier.joblib"))

# 2) labels — string labels in the model's OWN class index order.
#    clf.classes_ are the encoded ints [0..k-1]; invert them back to strings.
class_labels_list = le.inverse_transform(clf.classes_).tolist()
with open(os.path.join(OUTPUT_DIR, "class_labels.json"), "w", encoding="utf-8") as f:
    json.dump(class_labels_list, f)

# 3) evaluation report — schema matches backend app/ml/train_model.py
eval_report = {
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "dataset": os.path.abspath(CSV_PATH),
    "n_samples": int(X.shape[0]),
    "n_classes": int(len(class_labels_list)),
    "class_labels": class_labels_list,
    "test_accuracy": float(test_acc),
    "cv_accuracy_mean": float(cv_scores.mean()),
    "cv_accuracy_std": float(cv_scores.std()),
    "classification_report": report,
    "confusion_matrix": conf_mat,
}
with open(os.path.join(OUTPUT_DIR, "evaluation_report.json"), "w", encoding="utf-8") as f:
    json.dump(eval_report, f, indent=2)

print("Exported:")
print("  class order ->", class_labels_list)
for fn in ("gesture_classifier.joblib", "class_labels.json", "evaluation_report.json"):
    print("  ", os.path.join(OUTPUT_DIR, fn))


# =============================================================================
# CELL 10 — Verify artifacts reload exactly like the backend will
# =============================================================================
# Mirrors backend app/services/inference_service.py: argmax(predict_proba)
# indexes into class_labels.json.
reloaded = joblib.load(os.path.join(OUTPUT_DIR, "gesture_classifier.joblib"))
with open(os.path.join(OUTPUT_DIR, "class_labels.json"), encoding="utf-8") as f:
    reloaded_labels = json.load(f)

sample = X_test[0].reshape(1, -1)
proba = reloaded.predict_proba(sample)[0]
idx = int(np.argmax(proba))
print(f"Self-check: predicted '{reloaded_labels[idx]}' "
      f"(true '{le.inverse_transform([y_test[0]])[0]}'), conf={proba[idx]:.3f}")
assert len(reloaded_labels) == len(reloaded.classes_), "label/class count mismatch!"
print("Self-check passed: label count matches model classes.")


# =============================================================================
# CELL 11 — Zip and download
# =============================================================================
import shutil
from google.colab import files

required = ["gesture_classifier.joblib", "class_labels.json", "evaluation_report.json"]
for fn in required:
    p = os.path.join(OUTPUT_DIR, fn)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing artifact: {p}")

zip_base = "signbridge_trained_models"
shutil.make_archive(zip_base, "zip", OUTPUT_DIR)  # zips CONTENTS of trained_models/
print(f"Created {zip_base}.zip")

print("\nDEPLOYMENT — unzip and place the files here:")
print("  backend/trained_models/gesture_classifier.joblib")
print("  backend/trained_models/class_labels.json")
print("  backend/trained_models/evaluation_report.json")
print("\nThen GET /api/model/status -> loaded: true, and POST /api/recognize returns real predictions.")

files.download(f"{zip_base}.zip")
