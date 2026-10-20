"""Train and evaluate WESAD multimodal Random Forest models with LOSO CV.

Runs four experiments:
  1. Four-class WESAD condition classification using raw features
  2. Four-class classification using subject-baseline-normalized features
  3. Baseline-vs-Stress classification using raw features
  4. Baseline-vs-Stress classification using normalized features

Only rows marked training_label_usable are included. Each LOSO fold holds out
one entire subject. The script writes fold metrics, out-of-fold predictions,
confusion matrices, feature importances, and final all-data models.

Note: normalized-feature evaluation assumes a usable baseline calibration
window is available for the held-out subject, as in the dataset construction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

RAW_FEATURES = [
    "paper_mean_ibi_ms",
    "paper_mean_pulse_amplitude",
    "paper_phasic_response_rate_per_sec",
    "paper_tonic_std_uS",
    "paper_tonic_slope_uS_per_s",
    "rsp_mean_amplitude",
    "rsp_std_amplitude",
    "rsp_spectral_entropy",
    "rsp_mean_peak_valley_magnitude",
    "rsp_std_peak_valley_magnitude",
]
NORM_FEATURES = [f"norm_{name}" for name in RAW_FEATURES]
FOUR_CLASSES = ["Baseline", "Stress", "Amusement", "Meditation"]
BINARY_CLASSES = ["Baseline", "Stress"]
RANDOM_STATE = 42


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )


def safe_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def evaluate_loso(
    df: pd.DataFrame,
    features: list[str],
    classes: list[str],
    experiment_name: str,
    out_dir: Path,
) -> dict:
    """Run one fold per subject and aggregate out-of-fold predictions."""
    all_predictions: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    subjects = sorted(df["subject_id"].astype(str).unique(), key=lambda s: (int(s[1:]) if s[1:].isdigit() else 10**9, s))

    for held_out in subjects:
        train = df[df["subject_id"].astype(str) != held_out].copy()
        test = df[df["subject_id"].astype(str) == held_out].copy()
        if train.empty or test.empty:
            continue
        missing_train_classes = sorted(set(classes) - set(train["wesad_condition_label"].unique()))
        if missing_train_classes:
            raise ValueError(f"{experiment_name}, held-out {held_out}: training fold lacks classes {missing_train_classes}")

        X_train = train[features].to_numpy(dtype=float)
        X_test = test[features].to_numpy(dtype=float)
        if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
            raise ValueError(f"Non-finite feature value in {experiment_name}, held-out subject {held_out}.")

        model = make_model()
        model.fit(X_train, train["wesad_condition_label"].astype(str))
        pred = model.predict(X_test)
        y_true = test["wesad_condition_label"].astype(str).to_numpy()

        fold_rows.append({
            "experiment": experiment_name,
            "held_out_subject": held_out,
            "n_train_windows": len(train),
            "n_test_windows": len(test),
            "accuracy": accuracy_score(y_true, pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, labels=classes, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, pred, labels=classes, average="weighted", zero_division=0),
        })
        pred_df = test[["subject_id", "window_index", "window_start_seconds", "window_end_seconds", "wesad_condition_label", "label_purity_percent"]].copy()
        pred_df["experiment"] = experiment_name
        pred_df["predicted_label"] = pred
        pred_df["correct"] = pred_df["wesad_condition_label"].astype(str).to_numpy() == pred
        all_predictions.append(pred_df)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    y_true_all = predictions["wesad_condition_label"].astype(str).to_numpy()
    y_pred_all = predictions["predicted_label"].astype(str).to_numpy()
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    report = classification_report(y_true_all, y_pred_all, labels=classes, target_names=classes, output_dict=True, zero_division=0)

    summary = {
        "experiment": experiment_name,
        "features": features,
        "classes": classes,
        "subjects": len(subjects),
        "windows": len(predictions),
        "loso_accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "loso_balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
        "loso_macro_f1": float(f1_score(y_true_all, y_pred_all, labels=classes, average="macro", zero_division=0)),
        "loso_weighted_f1": float(f1_score(y_true_all, y_pred_all, labels=classes, average="weighted", zero_division=0)),
        "mean_subject_accuracy": float(folds["accuracy"].mean()),
        "std_subject_accuracy": float(folds["accuracy"].std(ddof=1)) if len(folds) > 1 else 0.0,
        "mean_subject_macro_f1": float(folds["macro_f1"].mean()),
        "std_subject_macro_f1": float(folds["macro_f1"].std(ddof=1)) if len(folds) > 1 else 0.0,
    }
    stem = safe_name(experiment_name)
    folds.to_csv(out_dir / f"{stem}_fold_metrics.csv", index=False)
    predictions.to_csv(out_dir / f"{stem}_oof_predictions.csv", index=False)
    pd.DataFrame(cm, index=[f"actual_{c}" for c in classes], columns=[f"predicted_{c}" for c in classes]).to_csv(out_dir / f"{stem}_confusion_matrix.csv")
    pd.DataFrame(report).transpose().to_csv(out_dir / f"{stem}_classification_report.csv")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train WESAD Random Forest models with Leave-One-Subject-Out evaluation.")
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent / "master_dataset" / "wesad_multimodal_master_180s.csv", help="Path to master dataset CSV")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: <CSV parent>/loso_results)")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    out_dir = (args.output.resolve() if args.output else csv_path.parent / "loso_results")
    if not csv_path.exists():
        print(f"ERROR: Dataset not found: {csv_path}", file=sys.stderr)
        print("Pass its path with --csv, e.g. --csv D:/PRJ/master_dataset/wesad_multimodal_master_180s.csv", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    df = pd.read_csv(csv_path)
    required = {"subject_id", "window_index", "wesad_condition_label", "training_label_usable", *RAW_FEATURES, *NORM_FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Parse booleans robustly in case a CSV reader represents them as strings.
    if df["training_label_usable"].dtype != bool:
        df["training_label_usable"] = df["training_label_usable"].astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if df["training_label_usable"].isna().any():
        raise ValueError("Could not parse some training_label_usable values as booleans.")

    # Only include eligible rows from the four target conditions.
    eligible = df[df["training_label_usable"] & df["wesad_condition_label"].isin(FOUR_CLASSES)].copy()
    if eligible.empty:
        raise ValueError("No eligible rows remain after filtering training_label_usable and WESAD classes.")
    if eligible.duplicated(["subject_id", "window_index"]).any():
        duplicates = eligible.loc[eligible.duplicated(["subject_id", "window_index"], keep=False), ["subject_id", "window_index"]]
        raise ValueError(f"Duplicate subject/window keys in eligible data:\n{duplicates.to_string(index=False)}")

    # Validate features without silently filling or replacing values.
    for col in RAW_FEATURES + NORM_FEATURES:
        eligible[col] = pd.to_numeric(eligible[col], errors="coerce")
        if not np.isfinite(eligible[col].to_numpy(dtype=float)).all():
            bad = int((~np.isfinite(eligible[col].to_numpy(dtype=float))).sum())
            raise ValueError(f"Feature {col} has {bad} NaN/inf values among eligible rows. Fix or explicitly handle them before training.")

    # Four-class experiments.
    experiments = [
        ("four_class_raw", eligible, RAW_FEATURES, FOUR_CLASSES),
        ("four_class_normalized", eligible, NORM_FEATURES, FOUR_CLASSES),
    ]
    # Binary stress task uses only Baseline and Stress windows.
    binary = eligible[eligible["wesad_condition_label"].isin(BINARY_CLASSES)].copy()
    experiments.extend([
        ("baseline_vs_stress_raw", binary, RAW_FEATURES, BINARY_CLASSES),
        ("baseline_vs_stress_normalized", binary, NORM_FEATURES, BINARY_CLASSES),
    ])

    print("=" * 72)
    print("WESAD MULTIMODAL RANDOM FOREST — LOSO CROSS-VALIDATION")
    print("=" * 72)
    print(f"Dataset: {csv_path}")
    print(f"All rows: {len(df)} | eligible four-class rows: {len(eligible)} | subjects: {eligible['subject_id'].nunique()}")
    print("Eligible class counts:")
    print(eligible["wesad_condition_label"].value_counts().reindex(FOUR_CLASSES, fill_value=0).to_string())
    print("\nRunning four experiments...\n")

    summaries = []
    for name, subset, features, classes in experiments:
        if subset.empty:
            print(f"SKIP {name}: no eligible rows")
            continue
        summary = evaluate_loso(subset, features, classes, name, out_dir)
        summaries.append(summary)
        print(f"{name}: accuracy={summary['loso_accuracy']:.3f}, balanced_accuracy={summary['loso_balanced_accuracy']:.3f}, macro-F1={summary['loso_macro_f1']:.3f}, mean-subject macro-F1={summary['mean_subject_macro_f1']:.3f}")

        # Save a deployable final model trained on all eligible rows for this task.
        X = subset[features].to_numpy(dtype=float)
        y = subset["wesad_condition_label"].astype(str)
        final_model = make_model().fit(X, y)
        model_bundle = {
            "model": final_model,
            "features": features,
            "classes": classes,
            "experiment": name,
            "training_subjects": sorted(subset["subject_id"].astype(str).unique()),
            "training_rows": len(subset),
            "random_state": RANDOM_STATE,
            "note": "LOSO metrics are the evaluation results; this saved model is refit on all eligible rows and is not a held-out fold model.",
        }
        joblib.dump(model_bundle, out_dir / f"{name}_final_model.joblib")
        importances = pd.DataFrame({"feature": features, "importance": final_model.feature_importances_}).sort_values("importance", ascending=False)
        importances.to_csv(out_dir / f"{name}_feature_importance.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "loso_experiment_summary.csv", index=False)
    (out_dir / "loso_experiment_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SUMMARY (out-of-fold predictions across held-out subjects)")
    print("=" * 72)
    if not summary_df.empty:
        print(summary_df[["experiment", "windows", "loso_accuracy", "loso_balanced_accuracy", "loso_macro_f1", "mean_subject_macro_f1"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved metrics, predictions, confusion matrices, reports, feature importances and final models to:\n{out_dir}")
    print(f"Runtime: {time.time() - started:.1f} seconds")
    print("\nInterpretation: compare experiments using the same LOSO folds. Do not use these results as evidence of generalization beyond WESAD without further validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
