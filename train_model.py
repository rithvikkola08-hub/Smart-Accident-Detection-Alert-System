"""Train and evaluate a session-split Random Forest on synthetic IMU data."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                             accuracy_score, balanced_accuracy_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "synthetic_raw_imu.csv"
FEATURE_PATH = ROOT / "outputs" / "synthetic_window_features.csv"
MODEL_PATH = ROOT / "outputs" / "synthetic_random_forest.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
CM_PATH = ROOT / "outputs" / "synthetic_confusion_matrix.png"
IMPORTANCE_PATH = ROOT / "outputs" / "synthetic_feature_importance.png"
FEATURES = [
    "accel_mean_norm_deviation", "accel_std_norm", "accel_peak_norm_deviation",
    "gyro_mean_norm", "gyro_rms_norm", "gyro_peak_norm", "peak_jerk",
]
LABELS = ["Normal", "Turn", "Bump", "Accident"]


def extract_window_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert each one-second session to a compact feature vector."""
    required = {"session_id", "label", "synthetic", "ax_mps2", "ay_mps2", "az_mps2",
                "gx_dps", "gy_dps", "gz_dps"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not raw["synthetic"].astype(str).str.lower().eq("true").all():
        raise ValueError("This demo trainer accepts only rows marked synthetic=true.")

    axes_a = ["ax_mps2", "ay_mps2", "az_mps2"]
    axes_g = ["gx_dps", "gy_dps", "gz_dps"]
    valid = raw.dropna(subset=axes_a + axes_g + ["label", "session_id"]).copy()
    numeric = axes_a + axes_g
    valid[numeric] = valid[numeric].apply(pd.to_numeric, errors="coerce")
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric)

    records = []
    for session_id, frame in valid.groupby("session_id", sort=False):
        frame = frame.sort_values("sample_index")
        if frame["label"].nunique() != 1:
            raise ValueError(f"Session {session_id} contains multiple class labels.")
        a = frame[axes_a].to_numpy(dtype=float)
        g = frame[axes_g].to_numpy(dtype=float)
        a_norm = np.linalg.norm(a, axis=1)
        g_norm = np.linalg.norm(g, axis=1)
        dt = np.diff(frame["timestamp_ms"].to_numpy(dtype=float)) / 1000.0
        dt = dt[dt > 0]
        sample_period = float(np.median(dt)) if len(dt) else 0.02
        jerk = np.linalg.norm(np.diff(a, axis=0), axis=1) / sample_period
        records.append({
            "session_id": session_id,
            "label": frame["label"].iloc[0],
            "accel_mean_norm_deviation": float(abs(a_norm.mean() - 9.81)),
            "accel_std_norm": float(a_norm.std()),
            "accel_peak_norm_deviation": float(abs(a_norm - 9.81).max()),
            "gyro_mean_norm": float(g_norm.mean()),
            "gyro_rms_norm": float(np.sqrt(np.mean(g_norm ** 2))),
            "gyro_peak_norm": float(g_norm.max()),
            "peak_jerk": float(jerk.max()) if len(jerk) else 0.0,
            "synthetic": True,
        })
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("No valid one-second sessions remain after preprocessing.")
    return result


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Run generate_synthetic_dataset.py first: {RAW_PATH}")
    (ROOT / "outputs").mkdir(exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    windows = extract_window_features(raw)
    windows.to_csv(FEATURE_PATH, index=False, float_format="%.6f")

    X = windows[FEATURES]
    y = windows["label"]
    groups = windows["session_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups, test_groups = set(groups.iloc[train_idx]), set(groups.iloc[test_idx])
    if train_groups & test_groups:
        raise RuntimeError("Session leakage detected between train and test splits.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=160, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, labels=LABELS,
                                   output_dict=True, zero_division=0)
    metrics = {
        "data_origin": "synthetic software-generated samples; not rover measurements",
        "model": "RandomForestClassifier in a median-imputation pipeline",
        "split_method": "GroupShuffleSplit by session_id; 75% train / 25% test",
        "train_sessions": len(train_groups),
        "test_sessions": len(test_groups),
        "train_windows": len(train_idx),
        "test_windows": len(test_idx),
        "accuracy_synthetic_test_only": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy_synthetic_test_only": float(balanced_accuracy_score(y_test, predictions)),
        "classification_report_synthetic_test_only": report,
        "warning": "These metrics only describe generated synthetic data and do not measure rover or real-world performance.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, MODEL_PATH)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=LABELS,
        display_labels=LABELS, cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title("Synthetic-only holdout evaluation\n(Not measured on rover hardware)")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=180)
    plt.close(fig)

    importances = model.named_steps["classifier"].feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(FEATURES)[order], importances[order], color="#4378bf")
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("Synthetic demo model — feature importance")
    fig.tight_layout()
    fig.savefig(IMPORTANCE_PATH, dpi=180)
    plt.close(fig)

    print(f"Synthetic-only model saved: {MODEL_PATH}")
    print(f"Synthetic-only metrics saved: {METRICS_PATH}")
    print(f"Synthetic holdout accuracy: {metrics['accuracy_synthetic_test_only']:.3f}")
    print("Do not report this score as rover or real-world performance.")


if __name__ == "__main__":
    main()
