"""Generate continuous timeline predictions and classify all unlabeled transient windows.

Loads the trained Random Forest models (Binary and 4-Class) and applies them
across the entire master dataset (476 windows), specifically extracting predictions
and probability distributions for:
  1. The 228 unlabeled/transient windows (unlabeled_transient_predictions.csv)
  2. The full continuous dataset (full_session_continuous_predictions.csv)
  3. A continuous timeline plot for representative subject S2
"""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
MASTER_CSV = ROOT_DIR / "master_dataset" / "wesad_multimodal_master_180s.csv"
RESULTS_DIR = ROOT_DIR / "master_dataset" / "loso_results"

def main():
    if not MASTER_CSV.is_file():
        raise FileNotFoundError(f"Master CSV not found at {MASTER_CSV}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load models
    binary_bundle = joblib.load(RESULTS_DIR / "baseline_vs_stress_normalized_final_model.joblib")
    binary_model = binary_bundle["model"]
    binary_features = binary_bundle["features"]
    binary_classes = list(binary_model.classes_)  # Order corresponding to predict_proba columns

    four_bundle = joblib.load(RESULTS_DIR / "four_class_normalized_final_model.joblib")
    four_model = four_bundle["model"]
    four_features = four_bundle["features"]
    four_classes = list(four_model.classes_)  # Order corresponding to predict_proba columns

    # 2. Load dataset
    df = pd.read_csv(MASTER_CSV)
    print(f"Loaded master dataset: {len(df)} windows across {df.subject_id.nunique()} subjects.")

    # 3. Predict Binary (Baseline vs. Stress)
    X_bin = df[binary_features].to_numpy(dtype=float)
    bin_preds = binary_model.predict(X_bin)
    bin_probs = binary_model.predict_proba(X_bin)

    df["pred_binary"] = bin_preds
    for idx, c in enumerate(binary_classes):
        df[f"prob_binary_{c.lower()}"] = bin_probs[:, idx]

    # 4. Predict 4-Class
    X_four = df[four_features].to_numpy(dtype=float)
    four_preds = four_model.predict(X_four)
    four_probs = four_model.predict_proba(X_four)

    df["pred_4class"] = four_preds
    for idx, c in enumerate(four_classes):
        df[f"prob_4class_{c.lower()}"] = four_probs[:, idx]

    # 5. Export full continuous predictions
    full_out = RESULTS_DIR / "full_session_continuous_predictions.csv"
    df.to_csv(full_out, index=False)
    print(f"Saved full continuous predictions ({len(df)} rows) to: {full_out}")

    # 6. Filter and export the 228 Unlabeled/Transient windows
    unlabeled = df[df["wesad_condition_label"] == "Transient"].copy()
    unlabeled_out = RESULTS_DIR / "unlabeled_transient_predictions.csv"
    
    export_cols = [
        "subject_id",
        "window_index",
        "window_start_seconds",
        "window_end_seconds",
        "wesad_condition_label",
        "pred_binary",
        "prob_binary_stress",
        "prob_binary_baseline",
        "pred_4class",
        "prob_4class_stress",
        "prob_4class_baseline",
        "prob_4class_amusement",
        "prob_4class_meditation",
        *binary_features,
    ]
    unlabeled[export_cols].to_csv(unlabeled_out, index=False)
    print(f"Saved unlabeled transient predictions ({len(unlabeled)} rows) to: {unlabeled_out}")

    print("\n--- Summary of Unlabeled Transient Predictions ---")
    print("Binary Predictions on 228 Unlabeled Windows:")
    print(unlabeled["pred_binary"].value_counts().to_string())
    print(f"Mean Stress Probability: {unlabeled['prob_binary_stress'].mean():.2%}")
    print("\n4-Class Predictions on 228 Unlabeled Windows:")
    print(unlabeled["pred_4class"].value_counts().to_string())

    # 7. Generate Timeline Visualization for Subject S2
    s2 = df[df["subject_id"] == "S2"].sort_values("window_start_seconds").reset_index(drop=True)
    time_min = (s2["window_start_seconds"] + s2["window_end_seconds"]) / 2.0 / 60.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    # Color mapping for ground-truth conditions
    condition_colors = {
        "Baseline": "#2ca02c",      # Green
        "Stress": "#d62728",        # Red
        "Amusement": "#1f77b4",     # Blue
        "Meditation": "#9467bd",    # Purple
        "Transient": "#e0e0e0",     # Light Gray
    }

    # Shade background for actual conditions
    for _, row in s2.iterrows():
        start_m = row["window_start_seconds"] / 60.0
        end_m = row["window_end_seconds"] / 60.0
        cond = row["wesad_condition_label"]
        col = condition_colors.get(cond, "#f0f0f0")
        ax1.axvspan(start_m, end_m, color=col, alpha=0.35, lw=0)
        ax2.axvspan(start_m, end_m, color=col, alpha=0.35, lw=0)

    # Plot continuous Stress Probability curve on ax1
    stress_prob = s2["prob_binary_stress"]
    ax1.plot(time_min, stress_prob, color="#b30000", lw=2.5, marker="o", markersize=6, label="Predicted Stress Probability")
    ax1.axhline(0.5, color="black", linestyle="--", alpha=0.6, label="Decision Threshold (0.5)")
    ax1.fill_between(time_min, 0.5, stress_prob, where=(stress_prob >= 0.5), color="#d62728", alpha=0.25, interpolate=True)
    ax1.fill_between(time_min, 0.5, stress_prob, where=(stress_prob < 0.5), color="#2ca02c", alpha=0.25, interpolate=True)

    ax1.set_ylabel("Stress Probability", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(["0% (Baseline)", "25%", "50%", "75%", "100% (Stress)"])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_title("Continuous Stress Monitoring Across Full Session (Subject S2)\nLeave-One-Subject-Out Random Forest (10 Multimodal Features)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)

    # Plot discrete predictions on ax2
    # Map predictions to numerical y for clean plotting
    pred_map = {"Baseline": 0, "Stress": 1}
    y_pred = [pred_map.get(p, 0) for p in s2["pred_binary"]]
    ax2.step(time_min, y_pred, where="mid", color="#111111", lw=2.0, label="Binary Decision")
    ax2.scatter(time_min, y_pred, c=["#d62728" if p == 1 else "#2ca02c" for p in y_pred], s=50, zorder=5)

    ax2.set_ylabel("Decision", fontsize=12, fontweight="bold")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Baseline", "Stress"])
    ax2.set_ylim(-0.3, 1.3)
    ax2.set_xlabel("Session Timeline (Minutes)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Custom legend for background shaded condition bands
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", alpha=0.4, label="True Baseline"),
        Patch(facecolor="#d62728", alpha=0.4, label="True Stress (TSST)"),
        Patch(facecolor="#1f77b4", alpha=0.4, label="True Amusement"),
        Patch(facecolor="#9467bd", alpha=0.4, label="True Meditation"),
        Patch(facecolor="#c0c0c0", alpha=0.4, label="Transient / Break"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", ncol=5, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plot_out = RESULTS_DIR / "continuous_stress_monitoring_timeline_S2.png"
    plt.savefig(plot_out, dpi=300)
    plt.close()
    print(f"\nGenerated continuous timeline plot: {plot_out}")

if __name__ == "__main__":
    main()
