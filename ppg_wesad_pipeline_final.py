"""
PPG / BVP FINAL PIPELINE — WESAD
================================
Final project-level pipeline for wrist BVP/PPG feature extraction on WESAD.

Paper Alignment (Bian et al. 2019 Table 3):
- The paper specifies exactly 2 PPG features:
    1. paper_mean_ibi_ms (Paper: "Mean of heart rate variability")
    2. paper_mean_pulse_amplitude (Paper: "Mean of amplitude of the peak values")
- Extended metrics (ext_sd_ibi_ms, ext_sd_pulse_amplitude) are retained for signal audit.
- Synchronized with WESAD ground-truth condition labels.
- Sampling rate: 64 Hz (Empatica E4 wrist BVP)
- Filter: 1.1 - 3.6 Hz (3rd order Butterworth forward-backward bandpass)
- Peak Detection: Minimum peak distance = 0.35 s (~22 samples, max pulse 171.4 bpm)
- IBI Quality Check: 300-2000 ms physiological bounds + 9-beat local moving median
- Windowing: 180 s non-overlapping windows (per Bian et al. 2019 Section 4.3)
- Outputs:
    * ppg_output_final/ppg_features_3min.csv
    * ppg_output_final/ibi_quality_audit.csv
    * Diagnostic inspection plots
"""

from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
WORKSPACE_DIR = Path(__file__).resolve().parent
_DEFAULT_PATH = WORKSPACE_DIR / "WESAD" / "WESAD" / "S2" / "S2.pkl"
DATA_FILE = _DEFAULT_PATH if _DEFAULT_PATH.exists() else Path(r"D:\PRJ\WESAD\WESAD\S2\S2.pkl")

OUTPUT_DIR = WORKSPACE_DIR / "ppg_output_final"

FS_BVP = 64.0                  # Hz (Empatica E4 BVP)
FS_LABEL = 700.0              # Hz (WESAD synchronized labels)
WINDOW_SECONDS = 180.0         # 3 minutes
WINDOW_STEP_SECONDS = 180.0    # Non-overlapping
MIN_PEAK_DISTANCE_SEC = 0.35   # 0.35 seconds (~22 samples)
MIN_PEAK_DISTANCE_SAMPLES = int(round(MIN_PEAK_DISTANCE_SEC * FS_BVP))

# Physiological & Quality Check Parameters
MIN_IBI_MS = 300.0             # Up to 200 bpm
MAX_IBI_MS = 2000.0            # Down to 30 bpm
LOCAL_MEDIAN_RADIUS = 4        # 4 beats before and 4 beats after = 9-beat neighborhood
LOCAL_DEVIATION_FRACTION = 0.25 # 25% allowed deviation from local moving median

# Filter Settings (Paper exact)
FILTER_LOW = 1.1               # Hz
FILTER_HIGH = 3.6              # Hz

LABEL_MAP = {
    0: "Transient",
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
    4: "Meditation"
}


# =============================================================================
# DATA LOADING & FILTERING
# =============================================================================
def load_wesad_bvp_and_labels(filepath: Path) -> tuple[np.ndarray, np.ndarray, str]:
    if not filepath.is_file():
        raise FileNotFoundError(f"WESAD pickle file not found at: {filepath}")
    
    subject_id = filepath.stem if filepath.stem.startswith("S") else filepath.parent.name
    if not subject_id.startswith("S"):
        subject_id = "S2"

    with open(filepath, "rb") as f:
        data = pickle.load(f, encoding="latin1")
        
    raw_bvp = data["signal"]["wrist"]["BVP"].flatten().astype(np.float64)
    labels = data["label"].flatten().astype(np.int32)
    return raw_bvp, labels, subject_id


def apply_bandpass(signal: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def flag_extreme_samples(raw: np.ndarray, output_dir: Path) -> pd.DataFrame:
    q25, q75 = np.percentile(raw, [25, 75])
    iqr = q75 - q25
    lo_fence = q25 - 3.0 * iqr
    hi_fence = q75 + 3.0 * iqr
    outlier_mask = (raw < lo_fence) | (raw > hi_fence)
    df_outliers = pd.DataFrame({
        "sample_index": np.where(outlier_mask)[0],
        "time_seconds": np.where(outlier_mask)[0] / FS_BVP,
        "raw_value": raw[outlier_mask],
    })
    df_outliers.to_csv(output_dir / "flagged_extreme_samples.csv", index=False)
    return df_outliers


# =============================================================================
# PEAK DETECTION & AUDIT
# =============================================================================
def detect_peaks(filtered: np.ndarray) -> np.ndarray:
    peaks, _ = find_peaks(
        filtered,
        distance=MIN_PEAK_DISTANCE_SAMPLES,
        prominence=0.1 * np.std(filtered)
    )
    return peaks


def clean_ibi_intervals(peaks: np.ndarray, filtered: np.ndarray, output_dir: Path) -> pd.DataFrame:
    ibi_ms = np.diff(peaks) / FS_BVP * 1000.0
    rows = []

    for i in range(len(ibi_ms)):
        a = peaks[i]
        b = peaks[i + 1]
        ibi = float(ibi_ms[i])
        reasons = []

        if ibi < MIN_IBI_MS:
            reasons.append("BELOW_MIN_BOUND_300MS")
        if ibi > MAX_IBI_MS:
            reasons.append("ABOVE_MAX_BOUND_2000MS")

        # Local moving neighborhood check
        start_idx = max(0, i - LOCAL_MEDIAN_RADIUS)
        end_idx = min(len(ibi_ms), i + LOCAL_MEDIAN_RADIUS + 1)
        neighborhood = ibi_ms[start_idx:end_idx]

        # Robust local median: use bounded intervals in neighborhood if available
        bounded_neigh = neighborhood[(neighborhood >= MIN_IBI_MS) & (neighborhood <= MAX_IBI_MS)]
        local_med = float(np.median(bounded_neigh)) if len(bounded_neigh) else float(np.median(neighborhood))

        if len(reasons) == 0:
            deviation = abs(ibi - local_med) / local_med if local_med > 0 else np.nan
            if deviation > LOCAL_DEVIATION_FRACTION:
                reasons.append("LOCAL_DEVIATION_REVIEW")
        else:
            deviation = abs(ibi - local_med) / local_med if local_med > 0 else np.nan

        # Pulse amplitude: peak b minus preceding trough between a and b
        if b > a + 1:
            trough = a + int(np.argmin(filtered[a:b + 1]))
            amp = float(filtered[b] - filtered[trough])
            pulse_amp = amp if (np.isfinite(amp) and amp > 0) else np.nan
        else:
            pulse_amp = np.nan

        is_valid = (len(reasons) == 0)
        status = "KEPT" if is_valid else "FLAGGED_FOR_REVIEW:" + ";".join(reasons)

        start_time = float(a / FS_BVP)
        end_time = float(b / FS_BVP)
        mid_time = float((a + b) / (2.0 * FS_BVP))

        rows.append({
            "interval_index": i,
            "start_peak_index": int(a),
            "end_peak_index": int(b),
            "start_peak_time_seconds": round(start_time, 4),
            "end_peak_time_seconds": round(end_time, 4),
            "mid_time_seconds": round(mid_time, 4),
            "ibi_ms": round(ibi, 2),
            "implied_pulse_rate_bpm": round(60000.0 / ibi, 1) if ibi > 0 else np.nan,
            "pulse_amplitude": round(pulse_amp, 4) if np.isfinite(pulse_amp) else np.nan,
            "local_median_ibi_ms": round(local_med, 2),
            "relative_deviation_from_local_median": round(float(deviation), 4),
            "is_valid": bool(is_valid),
            "status": status,
        })

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / "ibi_quality_audit.csv", index=False)
    return audit_df


# =============================================================================
# FEATURE EXTRACTION (180s WINDOWS)
# =============================================================================
def extract_features(
    audit_df: pd.DataFrame,
    labels: np.ndarray,
    total_samples: int,
    subject_id: str,
    output_dir: Path
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_per_window = int(round(WINDOW_SECONDS * FS_BVP))
    step_samples = int(round(WINDOW_STEP_SECONDS * FS_BVP))
    rows = []
    start = 0
    window_idx = 0

    while start + samples_per_window <= total_samples:
        end = start + samples_per_window
        start_sec = start / FS_BVP
        end_sec = end / FS_BVP

        # Assign audited intervals to this window by midpoint time
        win_intervals = audit_df[
            (audit_df["mid_time_seconds"] >= start_sec) &
            (audit_df["mid_time_seconds"] < end_sec)
        ]

        kept_intervals = win_intervals[win_intervals["is_valid"]]
        flagged_intervals = win_intervals[~win_intervals["is_valid"]]

        # Mean and SD IBI on KEPT intervals only (direct alignment with audit)
        if len(kept_intervals):
            valid_ibis = kept_intervals["ibi_ms"].values
            mean_ibi = float(np.mean(valid_ibis))
            sd_ibi = float(np.std(valid_ibis, ddof=1)) if len(valid_ibis) > 1 else np.nan
            mean_hr = float(60000.0 / mean_ibi) if mean_ibi > 0 else np.nan

            valid_amps = kept_intervals["pulse_amplitude"].dropna().values
            mean_amp = float(np.mean(valid_amps)) if len(valid_amps) else np.nan
            sd_amp = float(np.std(valid_amps, ddof=1)) if len(valid_amps) > 1 else np.nan
        else:
            mean_ibi = np.nan
            sd_ibi = np.nan
            mean_hr = np.nan
            mean_amp = np.nan
            sd_amp = np.nan

        # Synchronize WESAD Ground-Truth Condition Label (700 Hz)
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

        # Mark training-eligible windows: >= 90% purity and defined condition
        training_usable = bool(purity >= 90.0 and condition_name in ["Baseline", "Stress", "Amusement", "Meditation"])

        rows.append({
            "subject_id": subject_id,
            "window_index": window_idx,
            "window_start_seconds": start_sec,
            "window_end_seconds": end_sec,
            "wesad_condition_label": condition_name,
            "label_purity_percent": round(purity, 1),
            "training_label_usable": training_usable,
            # --- THE 2 EXACT PAPER FEATURES (Bian et al. 2019 Table 3) ---
            "paper_mean_ibi_ms": round(mean_ibi, 4) if np.isfinite(mean_ibi) else np.nan,
            "paper_mean_pulse_amplitude": round(mean_amp, 4) if np.isfinite(mean_amp) else np.nan,
            # --- EXTENDED INSPECTION METRICS ---
            "ext_sd_ibi_ms": round(sd_ibi, 4) if np.isfinite(sd_ibi) else np.nan,
            "ext_sd_pulse_amplitude": round(sd_amp, 4) if np.isfinite(sd_amp) else np.nan,
            "ext_mean_heart_rate_bpm": round(mean_hr, 2) if np.isfinite(mean_hr) else np.nan,
            # --- QUALITY CONTROL COUNTS (100% matched to audit table) ---
            "qc_total_ibi_count": int(len(win_intervals)),
            "qc_kept_ibi_count": int(len(kept_intervals)),
            "qc_flagged_ibi_count": int(len(flagged_intervals)),
        })
        window_idx += 1
        start += step_samples

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "ppg_features_3min.csv", index=False)
    return df


# =============================================================================
# DIAGNOSTIC PLOTTING
# =============================================================================
def generate_plots(raw: np.ndarray, filtered: np.ndarray, peaks: np.ndarray, output_dir: Path) -> None:
    time_min = np.arange(len(raw)) / FS_BVP / 60.0
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time_min, raw, color="#2b5c8f", lw=0.6, label="Raw BVP")
    ax.set_title("WESAD S2 — Full Wrist BVP Recording (Raw)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Recorded Units")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "01_raw_bvp_full.png", dpi=150)
    plt.close(fig)

    snippet_start = int(300 * FS_BVP)
    snippet_len = int(10 * FS_BVP)
    snippet_end = snippet_start + snippet_len
    t_snippet = np.arange(snippet_len) / FS_BVP

    sig_snip = filtered[snippet_start:snippet_end]
    p_snip = peaks[(peaks >= snippet_start) & (peaks < snippet_end)] - snippet_start

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_snippet, sig_snip, color="#1f77b4", lw=1.5, label="Filtered BVP (1.1-3.6 Hz)")
    ax.scatter(t_snippet[p_snip], sig_snip[p_snip], color="#d62728", s=60, zorder=5, label=f"Detected Peaks (min dist {MIN_PEAK_DISTANCE_SEC}s)")
    ax.set_title(f"WESAD S2 — Filtered BVP & Peak Detections (300s - 310s)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Signal Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "02_peaks_zoom_300s.png", dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main() -> None:
    global DATA_FILE, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="WESAD wrist BVP/PPG feature extraction pipeline")
    parser.add_argument("--data", default=str(DATA_FILE), help="Path to WESAD subject pickle (e.g. S2.pkl)")
    parser.add_argument("--out", default=str(OUTPUT_DIR), help="Output directory path")
    args = parser.parse_args()

    DATA_FILE = Path(args.data).expanduser()
    OUTPUT_DIR = Path(args.out).expanduser()

    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}. Please check --data path.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(">>> RUNNING STANDARDIZED PPG PIPELINE FOR WESAD")
    print("=" * 75)

    print(f"\n[STEP 1] Loading BVP and labels from: {DATA_FILE}")
    raw, labels, subject_id = load_wesad_bvp_and_labels(DATA_FILE)
    print(f"Subject: {subject_id}")
    print(f"Loaded {len(raw):,} BVP samples ({len(raw)/FS_BVP/60:.2f} minutes) at {FS_BVP:g} Hz")

    print(f"\n[STEP 2] Applying paper bandpass filter: {FILTER_LOW} - {FILTER_HIGH} Hz...")
    selected = apply_bandpass(raw, FS_BVP, FILTER_LOW, FILTER_HIGH)

    print("\n[STEP 3] Flagging extreme samples for quality log...")
    outliers = flag_extreme_samples(raw, OUTPUT_DIR)
    print(f"Flagged {len(outliers):,} extreme samples")

    print(f"\n[STEP 4] Detecting pulse peaks (min distance = {MIN_PEAK_DISTANCE_SEC}s)...")
    peaks = detect_peaks(selected)
    print(f"Total candidate peaks found: {len(peaks):,}")

    print("\n[STEP 5] Auditing IBI quality with local moving neighborhood...")
    audit_df = clean_ibi_intervals(peaks, selected, OUTPUT_DIR)
    kept_count = int(audit_df["is_valid"].sum())
    flagged_count = len(audit_df) - kept_count
    print(f"IBIs audited: {len(audit_df):,} total")
    print(f"  - Kept: {kept_count:,} ({kept_count/len(audit_df)*100:.1f}%)")
    print(f"  - Flagged for review: {flagged_count:,} ({flagged_count/len(audit_df)*100:.1f}%)")

    print(f"\n[STEP 6] Extracting the 2 paper features + condition labels in 180s windows...")
    features = extract_features(audit_df, labels, len(selected), subject_id, OUTPUT_DIR)

    print("\n[STEP 7] Generating diagnostic inspection plots...")
    generate_plots(raw, selected, peaks, OUTPUT_DIR)

    print("\n" + "=" * 75)
    print("PPG FEATURE EXTRACTION SUMMARY (By Condition)")
    print("=" * 75)

    condition_summary = features.groupby("wesad_condition_label").agg({
        "window_index": "count",
        "paper_mean_ibi_ms": "mean",
        "paper_mean_pulse_amplitude": "mean",
        "ext_mean_heart_rate_bpm": "mean",
        "ext_sd_ibi_ms": "mean"
    }).rename(columns={"window_index": "window_count"})
    print(condition_summary.to_string())

    print("\n" + "=" * 75)
    print("TRAINING-ELIGIBLE WINDOWS SUMMARY (purity >= 90% and non-transient):")
    print("=" * 75)
    usable_counts = features.groupby(["wesad_condition_label", "training_label_usable"]).size().unstack(fill_value=0)
    print(usable_counts.to_string())

    print("\n" + "=" * 75)
    print("FIRST 5 WINDOWS:")
    cols_preview = [
        "window_index", "wesad_condition_label", "label_purity_percent",
        "training_label_usable", "paper_mean_ibi_ms", "paper_mean_pulse_amplitude", "ext_mean_heart_rate_bpm"
    ]
    print(features[cols_preview].head().to_string(index=False))

    print(f"\nSaved all results to: {OUTPUT_DIR.resolve()}")
    print("=" * 75)
    print(">>> PPG PIPELINE STANDARDIZED & COMPLETED!")
    print("=" * 75)


if __name__ == "__main__":
    main()
