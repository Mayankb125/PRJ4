"""
GSR / EDA FINAL PIPELINE — WESAD
================================
Final project-level pipeline for wrist EDA/GSR feature extraction on WESAD.

Methodology & Standards:
- Sampling rate: 4 Hz (Empatica E4 wrist EDA)
- Pre-filtering: 0.5 Hz lowpass filter (3rd-order Butterworth, sosfiltfilt)
- Tonic Decomposition: 0.05 Hz lowpass filter (isolates baseline drift / SCL)
- Phasic Decomposition: Cleaned signal minus Tonic component (isolates SCRs)
- Peak Detection: Candidate threshold >= 0.01 uS with minimum distance 1.0 s
- Windowing: 180s non-overlapping windows (720 samples per window)
- Feature Naming (Paper-Aligned):
    * paper_phasic_response_rate_per_sec
    * paper_tonic_std_uS
    * paper_tonic_slope_uS_per_s
    * mean_tonic_uS (additional inspection metric)
    * mean_phasic_amplitude_uS (additional inspection metric)
    * wesad_condition_label (Ground-truth condition)
- Output Files:
    * gsr_features_3min.csv
    * gsr_quality_audit.csv (peak-by-peak audit)
    * gsr_window_audit.csv (window-level quality screening)
    * eda_decomposition_first120s.png
    * eda_full_session_overview.png
"""

from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks

# =============================================================================
# DEFAULT CONFIGURATION & PATHS
# =============================================================================
WORKSPACE_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = WORKSPACE_DIR / "WESAD" / "WESAD" / "S2" / "S2.pkl"
DATA_FILE = _DEFAULT_DATA if _DEFAULT_DATA.exists() else Path(r"D:\PRJ\WESAD\WESAD\S2\S2.pkl")
OUTPUT_DIR = WORKSPACE_DIR / "gsr_output"

FS_EDA = 4.0                  # Hz (Empatica E4 wrist EDA)
FS_LABEL = 700.0              # Hz (WESAD synchronized chest labels)
WINDOW_SECONDS = 180.0        # 3 minutes
WINDOW_STEP_SECONDS = 180.0   # Non-overlapping
SAMPLES_PER_WINDOW = int(round(WINDOW_SECONDS * FS_EDA)) # 720 samples

# Filter Cutoffs (Bian et al. 2019 Section 2.2)
LP_CUTOFF_NOISE = 0.5         # Hz
LP_CUTOFF_TONIC = 0.05        # Hz

# Candidate Phasic Peak Detection Thresholds
MIN_PEAK_HEIGHT_US = 0.01     # uS
MIN_PEAK_DISTANCE_SEC = 1.0   # seconds
MIN_PEAK_DISTANCE_SAMPLES = int(round(MIN_PEAK_DISTANCE_SEC * FS_EDA))

LABEL_MAP = {
    0: "Transient",
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
    4: "Meditation"
}


# =============================================================================
# DATA LOADING & PRE-PROCESSING
# =============================================================================
def load_wesad_eda_and_labels(filepath: Path) -> tuple[np.ndarray, np.ndarray, str]:
    if not filepath.is_file():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    # Extract subject ID from file path or parent folder
    subject_id = filepath.stem if filepath.stem.startswith("S") else filepath.parent.name
    if not subject_id.startswith("S"):
        subject_id = "S2"

    with open(filepath, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    
    raw_eda = data["signal"]["wrist"]["EDA"].flatten().astype(np.float64)
    labels = data["label"].flatten().astype(np.int32)
    return raw_eda, labels, subject_id


def check_signal_integrity(raw_eda: np.ndarray) -> dict:
    nan_count = int(np.isnan(raw_eda).sum())
    zero_count = int((raw_eda == 0).sum())
    diffs = np.diff(raw_eda)
    max_jump = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
    flat_sections = int((diffs == 0).sum())

    stats = {
        "total_samples": len(raw_eda),
        "duration_minutes": len(raw_eda) / FS_EDA / 60.0,
        "nan_count": nan_count,
        "zero_count": zero_count,
        "flat_sample_count": flat_sections,
        "min_raw_uS": float(np.min(raw_eda)),
        "max_raw_uS": float(np.max(raw_eda)),
        "mean_raw_uS": float(np.mean(raw_eda)),
        "max_sample_to_sample_jump_uS": max_jump,
    }
    return stats


def decompose_eda(raw_eda: np.ndarray, fs: float = FS_EDA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    
    # 1. Lowpass at 0.5 Hz to smooth and remove high-frequency noise
    sos_lp = butter(3, LP_CUTOFF_NOISE / nyq, btype="lowpass", output="sos")
    smoothed = sosfiltfilt(sos_lp, raw_eda)
    
    # 2. Lowpass at 0.05 Hz to isolate the slow tonic baseline drift (SCL)
    sos_tonic = butter(3, LP_CUTOFF_TONIC / nyq, btype="lowpass", output="sos")
    tonic = sosfiltfilt(sos_tonic, smoothed)
    
    # 3. Phasic component = Smoothed signal minus Tonic component
    phasic = smoothed - tonic
    return smoothed, tonic, phasic


# =============================================================================
# PEAK DETECTION & AUDIT
# =============================================================================
def detect_and_audit_phasic_peaks(
    phasic: np.ndarray,
    tonic: np.ndarray,
    output_dir: Path
) -> tuple[np.ndarray, pd.DataFrame]:
    peak_indices, properties = find_peaks(
        phasic,
        height=MIN_PEAK_HEIGHT_US,
        distance=MIN_PEAK_DISTANCE_SAMPLES,
        prominence=0.005
    )
    
    peak_heights = properties.get("peak_heights", np.array([]))
    prominences = properties.get("prominences", np.array([]))

    audit_rows = []
    for idx, p_idx in enumerate(peak_indices):
        time_sec = p_idx / FS_EDA
        height_val = float(peak_heights[idx]) if idx < len(peak_heights) else float(phasic[p_idx])
        prom_val = float(prominences[idx]) if idx < len(prominences) else 0.0
        tonic_at_peak = float(tonic[p_idx])

        audit_rows.append({
            "peak_index": int(p_idx),
            "timestamp_seconds": round(time_sec, 3),
            "timestamp_minutes": round(time_sec / 60.0, 3),
            "phasic_amplitude_uS": round(height_val, 5),
            "prominence_uS": round(prom_val, 5),
            "underlying_tonic_level_uS": round(tonic_at_peak, 4),
            "quality_status": "CANDIDATE_PHASIC_RESPONSE"
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(output_dir / "gsr_quality_audit.csv", index=False)
    return peak_indices, audit_df


# =============================================================================
# FEATURE EXTRACTION & WINDOW AUDIT
# =============================================================================
def extract_gsr_features_and_window_audit(
    raw_eda: np.ndarray,
    tonic: np.ndarray,
    phasic: np.ndarray,
    peaks: np.ndarray,
    labels: np.ndarray,
    subject_id: str,
    output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows = []
    window_audit_rows = []
    n_windows = int(len(raw_eda) // SAMPLES_PER_WINDOW)

    for w in range(n_windows):
        start_sample = w * SAMPLES_PER_WINDOW
        end_sample = start_sample + SAMPLES_PER_WINDOW
        
        start_sec = start_sample / FS_EDA
        end_sec = end_sample / FS_EDA

        raw_win = raw_eda[start_sample:end_sample]
        t_win = tonic[start_sample:end_sample]
        p_win = phasic[start_sample:end_sample]
        
        # Peaks within window
        w_peaks = peaks[(peaks >= start_sample) & (peaks < end_sample)]
        w_peak_amps = p_win[w_peaks - start_sample] if len(w_peaks) else np.array([])

        # Paper-exact target features
        phasic_rate = len(w_peaks) / WINDOW_SECONDS
        tonic_std = float(np.std(t_win, ddof=1)) if len(t_win) > 1 else 0.0

        time_axis = np.arange(len(t_win)) / FS_EDA
        slope = float(np.polyfit(time_axis, t_win, 1)[0]) if len(t_win) > 1 else 0.0

        # Additional inspection metrics
        mean_tonic = float(np.mean(t_win))
        mean_phasic_amp = float(np.mean(w_peak_amps)) if len(w_peak_amps) else np.nan

        # Synchronized Ground-Truth Condition Label from WESAD (700 Hz)
        lbl_start = int(round(start_sec * FS_LABEL))
        lbl_end = int(round(end_sec * FS_LABEL))
        win_labels = labels[lbl_start:lbl_end]
        
        if len(win_labels):
            vals, counts = np.unique(win_labels, return_counts=True)
            dom_id = int(vals[np.argmax(counts)])
            condition_name = LABEL_MAP.get(dom_id, f"Condition_{dom_id}")
            purity = float(counts[np.argmax(counts)] / len(win_labels) * 100.0)
        else:
            condition_name = "UNKNOWN"
            purity = 0.0

        # Quality Control Flags for Window Audit
        qc_flags = []
        if (np.diff(raw_win) == 0).sum() > 0.5 * len(raw_win):
            qc_flags.append("HIGH_FLAT_SAMPLES")
        if purity < 70.0:
            qc_flags.append("MIXED_CONDITION_TRANSITION")
        if np.isnan(raw_win).any():
            qc_flags.append("HAS_NANS")
        qc_status = "VALID" if len(qc_flags) == 0 else ";".join(qc_flags)

        # Mark training-eligible windows: >= 90% purity and recognized condition
        training_usable = bool(purity >= 90.0 and condition_name in ["Baseline", "Stress", "Amusement", "Meditation"])

        feature_rows.append({
            "subject_id": subject_id,
            "window_index": w,
            "window_start_seconds": start_sec,
            "window_end_seconds": end_sec,
            "wesad_condition_label": condition_name,
            "label_purity_percent": round(purity, 1),
            "training_label_usable": training_usable,
            "paper_phasic_response_rate_per_sec": round(phasic_rate, 5),
            "paper_tonic_std_uS": round(tonic_std, 5),
            "paper_tonic_slope_uS_per_s": round(slope, 7),
            "mean_tonic_uS": round(mean_tonic, 4),
            "mean_phasic_amplitude_uS": round(mean_phasic_amp, 5) if np.isfinite(mean_phasic_amp) else np.nan,
            "samples_in_window": len(t_win),
            "qc_phasic_peak_count": int(len(w_peaks))
        })

        window_audit_rows.append({
            "window_index": w,
            "start_time_seconds": start_sec,
            "end_time_seconds": end_sec,
            "dominant_condition": condition_name,
            "label_purity_percent": round(purity, 1),
            "phasic_peaks_detected": int(len(w_peaks)),
            "tonic_drift_uS": round(float(t_win[-1] - t_win[0]), 5) if len(t_win) else 0.0,
            "qc_status": qc_status
        })

    features_df = pd.DataFrame(feature_rows)
    window_audit_df = pd.DataFrame(window_audit_rows)

    features_df.to_csv(output_dir / "gsr_features_3min.csv", index=False)
    window_audit_df.to_csv(output_dir / "gsr_window_audit.csv", index=False)
    return features_df, window_audit_df


# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================
def generate_diagnostic_plots(
    raw_eda: np.ndarray,
    smoothed: np.ndarray,
    tonic: np.ndarray,
    phasic: np.ndarray,
    peaks: np.ndarray,
    features_df: pd.DataFrame,
    output_dir: Path
) -> None:
    # 1. First 120s Zoom Plot
    zoom_sec = 120.0
    zoom_samples = int(zoom_sec * FS_EDA)
    t_zoom = np.arange(zoom_samples) / FS_EDA

    raw_z = raw_eda[:zoom_samples]
    smooth_z = smoothed[:zoom_samples]
    tonic_z = tonic[:zoom_samples]
    phasic_z = phasic[:zoom_samples]
    peaks_z = peaks[peaks < zoom_samples]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t_zoom, raw_z, color="#7f7f7f", lw=1.2, alpha=0.7, label="Raw Wrist EDA (4 Hz)")
    axes[0].plot(t_zoom, smooth_z, color="#1f77b4", lw=1.5, label="Smoothed (0.5 Hz Lowpass)")
    axes[0].set_ylabel("EDA (µS)", fontweight="bold")
    axes[0].set_title("WESAD S2 — EDA Decomposition & Phasic Peak Audit (First 120 Seconds)", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t_zoom, tonic_z, color="#2ca02c", lw=1.8, label="Tonic Baseline (0.05 Hz Lowpass)")
    axes[1].set_ylabel("Tonic (µS)", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t_zoom, phasic_z, color="#9467bd", lw=1.5, label="Phasic Component (Smoothed - Tonic)")
    if len(peaks_z):
        axes[2].scatter(
            t_zoom[peaks_z],
            phasic_z[peaks_z],
            color="#d62728",
            s=50,
            zorder=5,
            label=f"Detected Peaks (>= {MIN_PEAK_HEIGHT_US} µS)"
        )
    axes[2].axhline(MIN_PEAK_HEIGHT_US, color="#d62728", ls="--", lw=0.8, alpha=0.6, label="Threshold (0.01 µS)")
    axes[2].set_ylabel("Phasic (µS)", fontweight="bold")
    axes[2].set_xlabel("Time (seconds)", fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "eda_decomposition_first120s.png", dpi=150)
    plt.close(fig)

    # 2. Full Session Overview with Condition Shading
    t_full_min = np.arange(len(raw_eda)) / FS_EDA / 60.0
    fig, (ax_top, ax_mid) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

    ax_top.plot(t_full_min, raw_eda, color="#7f7f7f", lw=0.6, alpha=0.6, label="Raw Wrist EDA")
    ax_top.plot(t_full_min, tonic, color="#2ca02c", lw=1.5, label="Tonic Baseline (SCL)")
    ax_top.set_ylabel("Skin Conductance (µS)", fontweight="bold")
    ax_top.set_title("WESAD S2 — Full Session Electrodermal Activity & Experimental Conditions", fontsize=13, fontweight="bold")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right")

    ax_mid.plot(t_full_min, phasic, color="#9467bd", lw=0.8, label="Phasic Component")
    if len(peaks):
        ax_mid.scatter(t_full_min[peaks], phasic[peaks], color="#d62728", s=15, alpha=0.7, label="Phasic Responses")
    ax_mid.set_ylabel("Phasic (µS)", fontweight="bold")
    ax_mid.set_xlabel("Time (minutes)", fontweight="bold")
    ax_mid.grid(True, alpha=0.3)
    ax_mid.legend(loc="upper right")

    color_palette = {
        "Baseline": "#aec7e8",
        "Stress": "#ff9896",
        "Amusement": "#c5b0d5",
        "Meditation": "#98df8a"
    }

    for _, row in features_df.iterrows():
        cond = row["wesad_condition_label"]
        if cond in color_palette:
            t0 = row["window_start_seconds"] / 60.0
            t1 = row["window_end_seconds"] / 60.0
            for ax in (ax_top, ax_mid):
                ax.axvspan(t0, t1, color=color_palette[cond], alpha=0.25, zorder=0)

    fig.tight_layout()
    fig.savefig(output_dir / "eda_full_session_overview.png", dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================
def main() -> None:
    global DATA_FILE, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="WESAD wrist EDA/GSR feature extraction pipeline")
    parser.add_argument("--data", default=str(DATA_FILE), help="Path to WESAD subject pickle (e.g. S2.pkl)")
    parser.add_argument("--out", default=str(OUTPUT_DIR), help="Output directory path")
    args = parser.parse_args()

    DATA_FILE = Path(args.data).expanduser()
    OUTPUT_DIR = Path(args.out).expanduser()

    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}. Please check --data path.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(">>> RUNNING GSR / EDA FINAL PIPELINE FOR WESAD")
    print("=" * 75)

    print(f"\n[STEP 1] Loading EDA and labels from: {DATA_FILE}")
    raw_eda, labels, subject_id = load_wesad_eda_and_labels(DATA_FILE)
    stats = check_signal_integrity(raw_eda)
    print(f"Subject: {subject_id}")
    print(f"Loaded {stats['total_samples']:,} EDA samples ({stats['duration_minutes']:.2f} minutes) at {FS_EDA:g} Hz")
    print(f"Signal integrity: NaNs={stats['nan_count']}, Zeros={stats['zero_count']}, Flat samples={stats['flat_sample_count']}")
    print(f"Raw Range: {stats['min_raw_uS']:.4f} to {stats['max_raw_uS']:.4f} uS (Mean: {stats['mean_raw_uS']:.4f} uS)")

    print(f"\n[STEP 2] Decomposing EDA into Tonic (< {LP_CUTOFF_TONIC} Hz) and Phasic ({LP_CUTOFF_TONIC}-{LP_CUTOFF_NOISE} Hz)...")
    smoothed, tonic, phasic = decompose_eda(raw_eda, FS_EDA)
    print(f"Tonic range:  {np.min(tonic):.4f} to {np.max(tonic):.4f} uS")
    print(f"Phasic range: {np.min(phasic):.4f} to {np.max(phasic):.4f} uS (Std: {np.std(phasic):.4f} uS)")

    print(f"\n[STEP 3] Auditing phasic response peaks (candidate threshold >= {MIN_PEAK_HEIGHT_US} uS)...")
    peaks, audit_df = detect_and_audit_phasic_peaks(phasic, tonic, OUTPUT_DIR)
    print(f"Total candidate phasic peaks detected: {len(peaks):,}")
    print(f"Average response rate across entire recording: {len(peaks)/(len(raw_eda)/FS_EDA):.4f} peaks/second")
    print(f"Peak audit log saved to: {OUTPUT_DIR / 'gsr_quality_audit.csv'}")

    print(f"\n[STEP 4] Extracting 3-minute window features with WESAD ground truth & window audit...")
    features_df, window_audit_df = extract_gsr_features_and_window_audit(
        raw_eda, tonic, phasic, peaks, labels, subject_id, OUTPUT_DIR
    )
    print(f"Successfully extracted {len(features_df)} windows (180s each).")
    print(f"Features saved to: {OUTPUT_DIR / 'gsr_features_3min.csv'}")
    print(f"Window audit saved to: {OUTPUT_DIR / 'gsr_window_audit.csv'}")

    print(f"\n[STEP 5] Generating diagnostic inspection plots...")
    generate_diagnostic_plots(raw_eda, smoothed, tonic, phasic, peaks, features_df, OUTPUT_DIR)
    print(f"Saved plots to: {OUTPUT_DIR / 'eda_decomposition_first120s.png'}")
    print(f"Saved plots to: {OUTPUT_DIR / 'eda_full_session_overview.png'}")

    print("\n" + "=" * 75)
    print("GSR FEATURE EXTRACTION SUMMARY (By Condition)")
    print("=" * 75)

    condition_summary = features_df.groupby("wesad_condition_label").agg({
        "window_index": "count",
        "paper_phasic_response_rate_per_sec": "mean",
        "paper_tonic_std_uS": "mean",
        "paper_tonic_slope_uS_per_s": "mean",
        "mean_tonic_uS": "mean"
    }).rename(columns={"window_index": "window_count"})

    print(condition_summary.to_string())

    print("\n" + "=" * 75)
    print("TRAINING-ELIGIBLE WINDOWS SUMMARY (purity >= 90% and non-transient):")
    print("=" * 75)
    usable_counts = features_df.groupby(["wesad_condition_label", "training_label_usable"]).size().unstack(fill_value=0)
    print(usable_counts.to_string())

    print("\n" + "=" * 75)
    print("FIRST 5 WINDOWS:")
    cols_preview = [
        "window_index", "wesad_condition_label", "label_purity_percent",
        "training_label_usable", "paper_phasic_response_rate_per_sec",
        "paper_tonic_std_uS", "paper_tonic_slope_uS_per_s", "mean_tonic_uS"
    ]
    print(features_df[cols_preview].head().to_string(index=False))

    print(f"\nAll outputs saved in: {OUTPUT_DIR.resolve()}")
    print("=" * 75)
    print(">>> GSR FINAL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
