"""Train and evaluate WESAD Respiration-Only (5-feature) Random Forest models with LOSO CV.

Isolated training pipeline for the Vernier GDX-RB live VR integration.
Reads the already-cleaned 15-subject master dataset (read-only) and extracts ONLY
the 5 respiration features defined in Bian et al. (2019 Table 3).

Runs four experiments:
  1. rsp_only_four_class_raw            (4-class WESAD states, 5 raw RSP features)
  2. rsp_only_four_class_normalized     (4-class WESAD states, 5 baseline-normalized RSP features)
  3. rsp_only_baseline_vs_stress_raw    (Binary Baseline vs Stress, 5 raw RSP features)
  4. rsp_only_baseline_vs_stress_normalized (Primary Live VR Model: Binary, 5 normalized RSP features)

Outputs are saved strictly inside D:/PRJ/RSP_Only_Model/results/ without touching
any existing code or multimodal models.
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
)

# Exact 5 Respiration features matching Bian et al. (2019 Table 3) & live_respiration_pipeline.py
RSP_RAW_FEATURES = [
    "rsp_mean_amplitude",
    "rsp_std_amplitude",
    "rsp_spectral_entropy",
    "rsp_mean_peak_valley_magnitude",
    "rsp_std_peak_valley_magnitude",
]
RSP_NORM_FEATURES = [f"norm_{name}" for name in RSP_RAW_FEATURES]

FOUR_CLASSES = ["Baseline", "Stress", "Amusement", "Meditation"]
BINARY_CLASSES = ["Baseline", "Stress"]
RANDOM_STATE = 42


def make_model() -> RandomForestClassifier:
    """Identical Random Forest hyperparameters to train_wesad_loso.py for fair comparison."""
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
    """Run strict Leave-One-Subject-Out (15 folds) and aggregate out-of-fold predictions."""
    all_predictions: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    subjects = sorted(
        df["subject_id"].astype(str).unique(),
        key=lambda s: (int(s[1:]) if s[1:].isdigit() else 10**9, s),
    )

    for held_out in subjects:
        train = df[df["subject_id"].astype(str) != held_out].copy()
        test = df[df["subject_id"].astype(str) == held_out].copy()
        if train.empty or test.empty:
            continue

        missing_train_classes = sorted(set(classes) - set(train["wesad_condition_label"].unique()))
        if missing_train_classes:
            raise ValueError(
                f"{experiment_name}, held-out {held_out}: training fold lacks classes {missing_train_classes}"
            )

        X_train = train[features].to_numpy(dtype=float)
        X_test = test[features].to_numpy(dtype=float)
        if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
            raise ValueError(f"Non-finite feature value in {experiment_name}, held-out subject {held_out}.")

        model = make_model()
        model.fit(X_train, train["wesad_condition_label"].astype(str))
        pred = model.predict(X_test)
        probs = model.predict_proba(X_test)
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

        pred_df = test[[
            "subject_id",
            "window_index",
            "window_start_seconds",
            "window_end_seconds",
            "wesad_condition_label",
            "label_purity_percent",
        ]].copy()
        pred_df["experiment"] = experiment_name
        pred_df["predicted_label"] = pred
        pred_df["confidence"] = np.max(probs, axis=1)
        pred_df["correct"] = pred_df["wesad_condition_label"].astype(str).to_numpy() == pred
        all_predictions.append(pred_df)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    y_true_all = predictions["wesad_condition_label"].astype(str).to_numpy()
    y_pred_all = predictions["predicted_label"].astype(str).to_numpy()
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    report = classification_report(
        y_true_all,
        y_pred_all,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "experiment": experiment_name,
        "n_features": len(features),
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
    pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in classes],
        columns=[f"predicted_{c}" for c in classes],
    ).to_csv(out_dir / f"{stem}_confusion_matrix.csv")
    pd.DataFrame(report).transpose().to_csv(out_dir / f"{stem}_classification_report.csv")
    return summary


def run_live_sanity_check(model_path: Path) -> None:
    """Verify the saved 5-feature binary normalized model responds properly to baseline & stress shifts."""
    bundle = joblib.load(model_path)
    model = bundle["model"]
    classes = list(model.classes_)

    print("\n" + "=" * 72)
    print("LIVE SANITY CHECK (5-Feature Normalized Respiration Model)")
    print("=" * 72)
    print(f"Loaded model: {model_path.name}")
    print(f"Features:     {bundle['features']}")

    scenarios = [
        ("At Personal Baseline (delta = 0)", [0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Calm Deep Breathing (+amp, +depth)", [0.8, 0.4, -0.05, 2.5, 0.6]),
        ("Acute Stress (shallow, rapid, irregular)", [-1.1, -0.8, 0.06, -3.5, -1.6]),
        ("Mild Tension (slightly shallow)", [-0.6, -0.4, 0.03, -2.0, -0.9]),
    ]

    for label, vec in scenarios:
        X = np.array([vec], dtype=float)
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        conf = float(np.max(probs))
        prob_map = {c: round(float(p), 3) for c, p in zip(classes, probs)}
        vr_state = "High" if pred == "Stress" else "Low"
        print(f"  {label:42s} -> {pred:8s} (VR={vr_state:4s}, conf={conf:.2f}) | {prob_map}")


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    default_csv = base_dir.parent / "master_dataset" / "wesad_multimodal_master_180s.csv"
    default_out = base_dir / "results"

    parser = argparse.ArgumentParser(
        description="Train isolated 5-feature Respiration-Only WESAD Random Forest models with LOSO CV."
    )
    parser.add_argument("--csv", type=Path, default=default_csv, help="Path to master dataset CSV (read-only)")
    parser.add_argument("--output", type=Path, default=default_out, help="Output directory")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    out_dir = args.output.resolve()

    if not csv_path.exists():
        print(f"ERROR: Master dataset not found at {csv_path}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    df = pd.read_csv(csv_path)

    required = {
        "subject_id",
        "window_index",
        "wesad_condition_label",
        "training_label_usable",
        *RSP_RAW_FEATURES,
        *RSP_NORM_FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df["training_label_usable"].dtype != bool:
        df["training_label_usable"] = (
            df["training_label_usable"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    eligible = df[df["training_label_usable"] & df["wesad_condition_label"].isin(FOUR_CLASSES)].copy()
    if eligible.empty:
        raise ValueError("No eligible rows remain after filtering.")

    for col in RSP_RAW_FEATURES + RSP_NORM_FEATURES:
        eligible[col] = pd.to_numeric(eligible[col], errors="coerce")
        if not np.isfinite(eligible[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Feature {col} contains NaN/inf values.")

    binary = eligible[eligible["wesad_condition_label"].isin(BINARY_CLASSES)].copy()

    experiments = [
        ("rsp_only_four_class_raw", eligible, RSP_RAW_FEATURES, FOUR_CLASSES),
        ("rsp_only_four_class_normalized", eligible, RSP_NORM_FEATURES, FOUR_CLASSES),
        ("rsp_only_baseline_vs_stress_raw", binary, RSP_RAW_FEATURES, BINARY_CLASSES),
        ("rsp_only_baseline_vs_stress_normalized", binary, RSP_NORM_FEATURES, BINARY_CLASSES),
    ]

    print("=" * 72)
    print("WESAD RESPIRATION-ONLY (5 FEATURES) RANDOM FOREST — LOSO CV")
    print("=" * 72)
    print(f"Source Dataset (read-only): {csv_path}")
    print(f"Output Directory:           {out_dir}")
    print(f"Subjects: {eligible['subject_id'].nunique()} | 4-Class Eligible: {len(eligible)} | Binary Eligible: {len(binary)}")
    print("\nEligible class counts:")
    print(eligible["wesad_condition_label"].value_counts().reindex(FOUR_CLASSES, fill_value=0).to_string())
    print("\nRunning 4 Respiration-Only LOSO experiments...\n")

    summaries = []
    primary_model_path = None

    for name, subset, features, classes in experiments:
        summary = evaluate_loso(subset, features, classes, name, out_dir)
        summaries.append(summary)
        print(
            f"{name:40s} | acc={summary['loso_accuracy']*100:5.2f}% | "
            f"bal_acc={summary['loso_balanced_accuracy']*100:5.2f}% | "
            f"macro_f1={summary['loso_macro_f1']:.3f}"
        )

        # Fit deployable final model on all 15 subjects
        X = subset[features].to_numpy(dtype=float)
        y = subset["wesad_condition_label"].astype(str)
        final_model = make_model().fit(X, y)

        model_bundle = {
            "model": final_model,
            "features": features,
            "classes": classes,
            "experiment": name,
            "modality": "RSP_ONLY",
            "training_subjects": sorted(subset["subject_id"].astype(str).unique()),
            "training_rows": len(subset),
            "random_state": RANDOM_STATE,
            "loso_accuracy": summary["loso_accuracy"],
            "loso_balanced_accuracy": summary["loso_balanced_accuracy"],
            "loso_macro_f1": summary["loso_macro_f1"],
            "note": "Respiration-only 5-feature Random Forest refit on all 15 WESAD subjects for live Vernier GDX-RB VR integration.",
        }
        out_model = out_dir / f"{name}_final_model.joblib"
        joblib.dump(model_bundle, out_model)

        if name == "rsp_only_baseline_vs_stress_normalized":
            primary_model_path = out_model

        importances = pd.DataFrame({
            "feature": features,
            "importance": final_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importances.to_csv(out_dir / f"{name}_feature_importance.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "rsp_only_loso_summary.csv", index=False)
    (out_dir / "rsp_only_loso_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("RESPIRATION-ONLY LOSO SUMMARY")
    print("=" * 72)
    print(
        summary_df[[
            "experiment",
            "windows",
            "loso_accuracy",
            "loso_balanced_accuracy",
            "loso_macro_f1",
            "mean_subject_macro_f1",
        ]].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    if primary_model_path and primary_model_path.exists():
        run_live_sanity_check(primary_model_path)

    print(f"\nAll models & reports saved cleanly to: {out_dir}")
    print(f"Runtime: {time.time() - started:.1f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
