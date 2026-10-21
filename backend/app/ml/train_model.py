"""
Gesture classifier training pipeline for SignBridge AI.

Usage:
    python -m app.ml.train_model \
        --dataset datasets/landmarks.csv \
        --output  trained_models/

The dataset CSV must have columns:
    label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20   (63 coordinate columns)

This is exactly the format produced by the dataset-collection endpoint
(POST /api/dataset/sample -> datasets/landmarks.csv).

Training produces:
    trained_models/gesture_classifier.joblib
    trained_models/class_labels.json
    trained_models/evaluation_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    p = argparse.ArgumentParser(description="Train SignBridge gesture classifier")
    p.add_argument("--dataset", required=True, help="Path to landmarks CSV")
    p.add_argument("--output",  required=True, help="Directory for trained model outputs")
    p.add_argument("--test-size",  type=float, default=0.2,  help="Test split fraction (default 0.2)")
    p.add_argument("--n-estimators", type=int, default=200,  help="RandomForest n_estimators (default 200)")
    p.add_argument("--random-state", type=int, default=42,   help="Random seed")
    return p.parse_args()


def load_dataset(csv_path: Path):
    """Load a CSV of labeled landmark features. Returns X (n, 63), y (n,).

    Uses the standard-library csv reader (no pandas dependency) so the training
    pipeline runs with only the declared backend requirements (numpy + sklearn).
    The CSV must have a 'label' column followed by exactly 63 feature columns.
    """
    import csv as _csv

    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        reader = _csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV is empty.")
        if "label" not in header:
            raise ValueError("CSV must contain a 'label' column.")

        label_idx = header.index("label")
        feature_idx = [i for i in range(len(header)) if i != label_idx]
        if len(feature_idx) != 63:
            raise ValueError(
                f"Expected 63 feature columns (21 landmarks × 3), found {len(feature_idx)}."
            )

        labels: list[str] = []
        rows: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            labels.append(row[label_idx])
            rows.append([float(row[i]) for i in feature_idx])

    if not rows:
        raise ValueError("CSV contains a header but no data rows.")

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels)
    return X, y


def train(args) -> None:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix
    )
    from sklearn.preprocessing import LabelEncoder

    dataset_path = Path(args.dataset)
    output_dir   = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Loading dataset from {dataset_path}")
    X, y_raw = load_dataset(dataset_path)
    print(f"[train] {X.shape[0]} samples, {len(set(y_raw))} classes: {sorted(set(y_raw))}")

    # Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    # Stratified split — keep participants separate if 'participant' column is
    # available; otherwise use a plain stratified split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size,
        stratify=y, random_state=args.random_state,
    )
    print(f"[train] Train: {len(X_train)}  Test: {len(X_test)}")

    # Train
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    print("[train] Training RandomForestClassifier …")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    cm     = confusion_matrix(y_test, y_pred).tolist()

    print(f"[train] Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Cross-val on training set
    cv_scores = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(5), scoring="accuracy")
    print(f"[train] 5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save model
    model_path = output_dir / "gesture_classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"[train] Model saved to {model_path}")

    # Save labels
    labels = le.classes_.tolist()
    labels_path = output_dir / "class_labels.json"
    with labels_path.open("w") as f:
        json.dump(labels, f)
    print(f"[train] Labels saved to {labels_path}")

    # Save evaluation report
    eval_report = {
        "trained_at":      datetime.now(timezone.utc).isoformat(),
        "dataset":         str(dataset_path),
        "n_samples":       int(X.shape[0]),
        "n_classes":       len(labels),
        "class_labels":    labels,
        "test_accuracy":   float(acc),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std":  float(cv_scores.std()),
        "classification_report": report,
        "confusion_matrix": cm,
    }
    eval_path = output_dir / "evaluation_report.json"
    with eval_path.open("w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"[train] Evaluation report saved to {eval_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
