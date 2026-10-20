"""
RESPIRATION (RSP) FINAL PIPELINE — WESAD
========================================
Final validated pipeline for chest RespiBAN feature extraction on WESAD.

Paper Alignment (Bian et al. 2019 Table 3):
- Extracts all 5 Respiration features:
    1. rsp_mean_amplitude: Mean magnitude of the filtered breathing wave
    2. rsp_std_amplitude: Standard deviation of breathing amplitude
    3. rsp_spectral_entropy: Subband spectral entropy (0.05-0.35 Hz) via Welch PSD
    4. rsp_mean_peak_valley_magnitude: Mean breath depth (inhalation peak minus preceding trough)
    5. rsp_std_peak_valley_magnitude: Standard deviation of breath depths
- Scientific Refinements Implemented:
    * Breath detection distance set to 2.85s (1 / 0.35 Hz cutoff = 2.857s) to eliminate sub-cycle double peaks
    * Cycle-aware peak-to-peak and inhalation interval tracking in breath audit
    * Anti-aliased polyphase resampling (resample_poly) for raw display plot
    * Baseline-corrected features (_bc) saved alongside raw values
    * Synchronized with WESAD ground-truth condition labels
- Sampling rate: 700 Hz raw downsampled to 100 Hz after 0.05-0.35 Hz bandpass
- Windowing: 180s non-overlapping windows (exact 1-to-1 match with PPG & GSR)
- Outputs:
    * rsp_output/rsp_features_3min.csv
    * rsp_output/rsp_breaths_audit.csv
    * rsp_output/rsp_window_audit.csv
    * Diagnostic inspection plots & run summary
"""

from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# =============================================================================
# CONFIGURATION & PARAMETERS
# =============================================================================
WORKSPACE_DIR = Path(__file__).resolve().parent
_DEFAULT_PATH = WORKSPACE_DIR / "WESAD" / "WESAD" / "S2" / "S2.pkl"
DATA_FILE = _DEFAULT_PATH if _DEFAULT_PATH.exists() else Path(r"D:\PRJ\WESAD\WESAD\S2\S2.pkl")
OUTPUT_DIR = WORKSPACE_DIR / "rsp_output"

RAW_FS = 700.0                # Hz (Chest RespiBAN)
TARGET_FS = 100.0             # Hz (Paper target sampling rate)
DOWNSAMPLE_FACTOR = int(RAW_FS // TARGET_FS) # 7

WINDOW_S = 180.0              # 3 minutes
STEP_S = 180.0                # Non-overlapping
SAMPLES_PER_WINDOW = int(round(WINDOW_S * TARGET_FS)) # 18,000 samples at 100 Hz

# Respiration Bandpass Filter (Bian et al. 2019 Section 2.2)
LOW_HZ = 0.05                 # Hz (~3 breaths/min, slow deep breathing)
HIGH_HZ = 0.35                # Hz (~21 breaths/min, rapid breathing)

# Breath Detection Parameters (Strictly bounded by 0.35 Hz cutoff: 1 / 0.35 = 2.857s)
MIN_BREATH_DISTANCE_S = 2.85  # seconds
PROMINENCE_SD_FRACTION = 0.1  # fraction of window SD for peak/valley prominence

LABEL_MAP = {
    0: "Transient",
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
    4: "Meditation"
}


# =============================================================================
# DATA LOADING & SIGNAL PREPROCESSING
# =============================================================================
def load_wesad_resp_and_labels(filepath: Path) -> tuple[np.ndarray, np.ndarray, str]:
    if not filepath.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    subject_id = filepath.stem if filepath.stem.startswith("S") else filepath.parent.name
    if not subject_id.startswith("S"):
        subject_id = "S2"

    with open(filepath, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    raw_resp = np.asarray(data["signal"]["chest"]["Resp"]).ravel().astype(np.float64)
    labels = np.asarray(data["label"]).ravel().astype(np.int32)
    return raw_resp, labels, subject_id


def clean_raw_signal(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bad_mask = ~np.isfinite(raw)
    clean = raw.copy()
    if bad_mask.any():
        idx = np.arange(len(raw))
        clean[bad_mask] = np.interp(idx[bad_mask], idx[~bad_mask], raw[~bad_mask])
    return clean, bad_mask


def filter_and_downsample_rsp(clean: np.ndarray, raw_fs: float = RAW_FS, down_factor: int = DOWNSAMPLE_FACTOR) -> np.ndarray:
    # 2nd-order Butterworth bandpass forward-backward (0.05 - 0.35 Hz)
    sos = signal.butter(2, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=raw_fs, output="sos")
    filtered_700 = signal.sosfiltfilt(sos, clean)
    
    # Since bandpass removes all energy above 0.35 Hz (far below Nyquist 50 Hz at 100 Hz),
    # decimation by factor 7 is mathematically anti-aliased and instantaneous.
    filtered_100 = filtered_700[::down_factor]
    return filtered_100


# =============================================================================
# FEATURE EXTRACTION MATHEMATICS
# =============================================================================
def spectral_entropy(x: np.ndarray, fs: float = TARGET_FS, band: tuple = (LOW_HZ, HIGH_HZ), nperseg_s: float = 60.0) -> float:
    nperseg = min(len(x), int(nperseg_s * fs))
    f, p = signal.welch(x, fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    
    mask = (f >= band[0]) & (f <= band[1])
    p_sub = p[mask]
    total_power = p_sub.sum()
    
    if total_power <= 0 or len(p_sub) <= 1:
        return np.nan
    
    p_norm = p_sub / total_power
    entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12)) / np.log2(len(p_norm))
    return float(np.clip(entropy, 0.0, 1.0))


def detect_breaths(x: np.ndarray, fs: float = TARGET_FS, window_start_s: float = 0.0) -> tuple[list, np.ndarray, np.ndarray]:
    dist_samples = int(MIN_BREATH_DISTANCE_S * fs)
    sd = float(np.std(x))
    prom = PROMINENCE_SD_FRACTION * sd if sd > 0 else 0.001

    peaks, _ = signal.find_peaks(x, distance=dist_samples, prominence=prom)
    valleys, _ = signal.find_peaks(-x, distance=dist_samples, prominence=prom)

    breaths = []
    prev_p_sec = None

    for p in peaks:
        prev_valleys = valleys[valleys < p]
        if len(prev_valleys):
            v = prev_valleys[-1]
            magnitude = float(x[p] - x[v])
            p_time = window_start_s + p / fs
            v_time = window_start_s + v / fs
            cycle_duration = round(p_time - prev_p_sec, 3) if prev_p_sec is not None else np.nan

            if magnitude > 0:
                breaths.append({
                    "peak_index_in_window": int(p),
                    "valley_index_in_window": int(v),
                    "peak_time_seconds": round(p_time, 3),
                    "valley_time_seconds": round(v_time, 3),
                    "inhalation_duration_seconds": round((p - v) / fs, 3),
                    "cycle_duration_seconds": cycle_duration,
                    "peak_amplitude": round(float(x[p]), 4),
                    "valley_amplitude": round(float(x[v]), 4),
                    "peak_valley_magnitude": round(magnitude, 4)
                })
            prev_p_sec = p_time

    return breaths, peaks, valleys


# =============================================================================
# WINDOW PROCESSING & AUDIT
# =============================================================================
def process_respiration_windows(
    raw: np.ndarray,
    clean: np.ndarray,
    bad_mask: np.ndarray,
    filtered100: np.ndarray,
    labels: np.ndarray,
    subject: str,
    output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    n_windows = int(len(filtered100) // SAMPLES_PER_WINDOW)
    fs = TARGET_FS
    
    features = []
    breath_rows = []
    audit_rows = []

    for w in range(n_windows):
        start = w * SAMPLES_PER_WINDOW
        end = (w + 1) * SAMPLES_PER_WINDOW
        start_s = start / fs
        end_s = end / fs

        x = filtered100[start:end]
        abs_x = np.abs(x)

        # 1 & 2: Mean & SD Amplitude of filtered waveform (documented candidate definition)
        mean_amp = float(np.mean(abs_x))
        std_amp = float(np.std(abs_x, ddof=1)) if len(abs_x) > 1 else 0.0

        # 3: Subband Spectral Entropy (0.05 - 0.35 Hz)
        entropy = spectral_entropy(x, fs)

        # 4 & 5: Peak-Valley Breath Magnitudes
        breaths, peaks, valleys = detect_breaths(x, fs, start_s)
        mags = np.asarray([b["peak_valley_magnitude"] for b in breaths], dtype=float)
        mean_pv = float(np.mean(mags)) if len(mags) else np.nan
        std_pv = float(np.std(mags, ddof=1)) if len(mags) > 1 else np.nan

        # Synchronize WESAD Ground-Truth Label (700 Hz)
        raw_a = int(round(start_s * RAW_FS))
        raw_b = int(round(end_s * RAW_FS))
        win_labels = labels[raw_a:raw_b]

        if len(win_labels):
            vals, counts = np.unique(win_labels, return_counts=True)
            dom_id = int(vals[np.argmax(counts)])
            condition = LABEL_MAP.get(dom_id, f"Condition_{dom_id}")
            purity = float(counts[np.argmax(counts)] / len(win_labels) * 100.0)
        else:
            condition = "UNKNOWN"
            purity = 0.0

        # Quality Control & Signal Integrity Checks
        raw_win = raw[raw_a:raw_b]
        raw_finite = raw_win[np.isfinite(raw_win)]
        diffs = np.diff(raw_win)
        finite_diffs = diffs[np.isfinite(diffs)]
        
        jump_thr = 5.0 * np.std(finite_diffs) if len(finite_diffs) else np.inf
        jump_count = int(np.sum(np.abs(finite_diffs) > jump_thr)) if len(finite_diffs) else 0
        flat_fraction = float(np.mean(np.isclose(finite_diffs, 0.0))) if len(finite_diffs) else np.nan

        qc = []
        if bad_mask[raw_a:raw_b].any():
            qc.append("NONFINITE_RAW_INTERPOLATED")
        if len(breaths) < 3:
            qc.append("LOW_BREATH_COUNT_REVIEW")
        if purity < 70.0:
            qc.append("MIXED_CONDITION_TRANSITION")
        status = ";".join(qc) if qc else "OK"

        features.append({
            "subject_id": subject,
            "window_index": w,
            "window_start_seconds": start_s,
            "window_end_seconds": end_s,
            "wesad_condition_label": condition,
            "label_purity_percent": round(purity, 1),
            "training_label_usable": bool(purity >= 90.0 and condition in ["Baseline", "Stress", "Amusement", "Meditation"]),
            # --- THE 5 EXACT PAPER FEATURES (Table 3) ---
            "rsp_mean_amplitude": round(mean_amp, 5),
            "rsp_std_amplitude": round(std_amp, 5),
            "rsp_spectral_entropy": round(entropy, 5),
            "rsp_mean_peak_valley_magnitude": round(mean_pv, 5) if np.isfinite(mean_pv) else np.nan,
            "rsp_std_peak_valley_magnitude": round(std_pv, 5) if np.isfinite(std_pv) else np.nan,
            # --- EXTRA DIAGNOSTIC METRICS ---
            "rsp_breath_rate_bpm_qc": round(len(breaths) / (WINDOW_S / 60.0), 2),
            "n_breaths": int(len(breaths)),
            "is_baseline_reference": bool(condition == "Baseline" and purity >= 90.0),
            "qc_status": status
        })

        for bi, breath in enumerate(breaths):
            breath_rows.append({
                "subject_id": subject,
                "window_index": w,
                "condition": condition,
                "breath_index_in_window": bi,
                **breath
            })

        audit_rows.append({
            "subject_id": subject,
            "window_index": w,
            "window_start_seconds": start_s,
            "window_end_seconds": end_s,
            "condition": condition,
            "label_purity_percent": round(purity, 1),
            "n_breaths_detected": len(breaths),
            "raw_min": round(float(np.min(raw_finite)), 3) if len(raw_finite) else np.nan,
            "raw_max": round(float(np.max(raw_finite)), 3) if len(raw_finite) else np.nan,
            "raw_std": round(float(np.std(raw_finite)), 3) if len(raw_finite) else np.nan,
            "large_jump_count": jump_count,
            "flat_difference_fraction": round(flat_fraction, 4) if np.isfinite(flat_fraction) else np.nan,
            "quality_status": status
        })

    feat_df = pd.DataFrame(features)
    breath_df = pd.DataFrame(breath_rows)
    audit_df = pd.DataFrame(audit_rows)

    # Baseline Correction (subtract reference baseline window, per Bian et al. Section 2.3.1)
    baseline_indices = feat_df.index[feat_df["is_baseline_reference"]].tolist()
    raw_feature_cols = [
        "rsp_mean_amplitude", "rsp_std_amplitude", "rsp_spectral_entropy",
        "rsp_mean_peak_valley_magnitude", "rsp_std_peak_valley_magnitude"
    ]
    
    if baseline_indices:
        ref_idx = baseline_indices[0]
        base_vals = feat_df.loc[ref_idx, raw_feature_cols]
        for col in raw_feature_cols:
            feat_df[col + "_bc"] = round(feat_df[col] - base_vals[col], 5)
        baseline_win_idx = int(feat_df.loc[ref_idx, "window_index"])
    else:
        for col in raw_feature_cols:
            feat_df[col + "_bc"] = np.nan
        baseline_win_idx = None

    feat_df.to_csv(output_dir / "rsp_features_3min.csv", index=False)
    breath_df.to_csv(output_dir / "rsp_breaths_audit.csv", index=False)
    audit_df.to_csv(output_dir / "rsp_window_audit.csv", index=False)

    return feat_df, breath_df, audit_df, baseline_win_idx


# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================
def generate_diagnostic_plots(
    clean: np.ndarray,
    filtered100: np.ndarray,
    features_df: pd.DataFrame,
    breaths_df: pd.DataFrame,
    output_dir: Path
) -> None:
    fs = TARGET_FS

    # 1. First 120s Zoom with anti-aliased raw display and candidate breaths
    zoom = min(len(filtered100), int(120 * fs))
    t_zoom = np.arange(zoom) / fs
    
    # Proper polyphase anti-aliased resampling of raw signal for visual display
    raw_snippet = clean[:int(120 * RAW_FS)]
    raw_display = signal.resample_poly(raw_snippet, up=1, down=DOWNSAMPLE_FACTOR)[:zoom]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t_zoom, raw_display - np.mean(raw_display), color="#7f7f7f", lw=0.7, alpha=0.6, label="Raw Respiration (resampled for display)")
    ax.plot(t_zoom, filtered100[:zoom], color="#1f77b4", lw=1.2, label=f"Filtered ({LOW_HZ} - {HIGH_HZ} Hz)")

    # Mark peaks and valleys in first 120s
    first_breaths = breaths_df[breaths_df["window_index"] == 0] if len(breaths_df) else pd.DataFrame()
    if len(first_breaths):
        p_times = first_breaths["peak_time_seconds"].values
        v_times = first_breaths["valley_time_seconds"].values
        p_in_zoom = [pt for pt in p_times if pt < 120.0]
        v_in_zoom = [vt for vt in v_times if vt < 120.0]
        
        p_samples = [int(round(pt * fs)) for pt in p_in_zoom]
        v_samples = [int(round(vt * fs)) for vt in v_in_zoom]
        
        if p_samples:
            ax.scatter(np.array(p_samples) / fs, filtered100[p_samples], marker="^", color="#2ca02c", s=60, zorder=5, label=f"Inhalation Peaks (min {MIN_BREATH_DISTANCE_S}s)")
        if v_samples:
            ax.scatter(np.array(v_samples) / fs, filtered100[v_samples], marker="v", color="#d62728", s=60, zorder=5, label="Exhalation Valleys")

    ax.set_title("WESAD S2 Chest Respiration — First 120 Seconds (Validated Breath Detection)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_ylabel("Respiration (% stretch)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "01_rsp_raw_vs_filtered_120s.png", dpi=150)
    plt.close(fig)

    # 2. Welch Power Spectrum
    first_window = filtered100[:min(len(filtered100), int(WINDOW_S * fs))]
    ff, pxx = signal.welch(first_window, fs=fs, nperseg=int(fs * 60), detrend="constant")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ff, pxx, color="#1f77b4", lw=1.2, label="Welch PSD")
    ax.axvspan(LOW_HZ, HIGH_HZ, color="#2ca02c", alpha=0.2, label=f"Breathing Band ({LOW_HZ} - {HIGH_HZ} Hz)")
    ax.set_xlim(0, 1.0)
    ax.set_title("WESAD S2 Respiration — Welch Power Spectral Density", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax.set_ylabel("Power Spectral Density", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "02_rsp_power_spectrum.png", dpi=150)
    plt.close(fig)

    # 3. Full Session Overview with Condition Shading
    t_min = np.arange(len(filtered100)) / fs / 60.0
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(t_min, filtered100, color="#2b5c8f", lw=0.5, label="Filtered Respiration")
    
    colors = {"Baseline": "green", "Stress": "red", "Amusement": "orange", "Meditation": "purple"}
    for _, r in features_df.iterrows():
        cond = r["wesad_condition_label"]
        if cond in colors:
            ax.axvspan(r["window_start_seconds"] / 60.0, r["window_end_seconds"] / 60.0, color=colors[cond], alpha=0.15)

    ax.set_title("WESAD S2 Chest Respiration — Full Session (Protocol Condition Shading)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (minutes)", fontweight="bold")
    ax.set_ylabel("Respiration (% stretch)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "03_rsp_session_overview.png", dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================
def main() -> None:
    global DATA_FILE, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="WESAD chest respiration feature extraction pipeline")
    parser.add_argument("--data", default=str(DATA_FILE), help="Path to WESAD subject pickle (e.g. S2.pkl)")
    parser.add_argument("--out", default=str(OUTPUT_DIR), help="Output directory path")
    parser.add_argument("--save_filtered_signal", action="store_true", help="Save filtered signal CSV")
    args = parser.parse_args()

    DATA_FILE = Path(args.data).expanduser()
    OUTPUT_DIR = Path(args.out).expanduser()

    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}. Please check --data path.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(">>> RUNNING REFINED RESPIRATION (RSP) PIPELINE FOR WESAD")
    print("=" * 75)

    print(f"\n[STEP 1] Loading Respiration and labels from: {DATA_FILE}")
    raw, labels, subject_id = load_wesad_resp_and_labels(DATA_FILE)
    clean, bad_mask = clean_raw_signal(raw)
    duration_min = len(raw) / RAW_FS / 60.0
    print(f"Subject: {subject_id}")
    print(f"Loaded {len(raw):,} raw samples ({duration_min:.2f} minutes) at {RAW_FS:g} Hz")
    print(f"Interpolated non-finite samples: {int(bad_mask.sum())}")

    print(f"\n[STEP 2] Applying {LOW_HZ}-{HIGH_HZ} Hz bandpass & downsampling to {TARGET_FS:g} Hz...")
    filtered100 = filter_and_downsample_rsp(clean, RAW_FS, DOWNSAMPLE_FACTOR)
    print(f"Filtered {len(filtered100):,} samples at {TARGET_FS:g} Hz")

    print(f"\n[STEP 3] Extracting the 5 paper features (Min breath distance = {MIN_BREATH_DISTANCE_S}s)...")
    features_df, breaths_df, audit_df, base_win = process_respiration_windows(
        raw, clean, bad_mask, filtered100, labels, subject_id, OUTPUT_DIR
    )
    print(f"Extracted {len(features_df)} windows (180s each).")
    print(f"Total validated breaths detected: {len(breaths_df):,}")
    print(f"Baseline reference window used: Window {base_win} (start={base_win*WINDOW_S:.0f}s)")

    print(f"\n[STEP 4] Generating diagnostic inspection plots (anti-aliased display)...")
    generate_diagnostic_plots(clean, filtered100, features_df, breaths_df, OUTPUT_DIR)
    print(f"Saved plots to: {OUTPUT_DIR.resolve()}")

    if args.save_filtered_signal:
        sig_df = pd.DataFrame({
            "time_seconds": np.arange(len(filtered100)) / TARGET_FS,
            "filtered_respiration": filtered100
        })
        sig_df.to_csv(OUTPUT_DIR / "rsp_filtered_signal.csv", index=False)
        print("Saved filtered respiration signal CSV")

    # Generate Run Summary
    summary_lines = [
        "WESAD RESPIRATION PIPELINE SUMMARY (REFINED)",
        f"Input: {DATA_FILE}",
        f"Subject: {subject_id}",
        f"Raw samples: {len(raw):,} at {RAW_FS:g} Hz ({duration_min:.2f} mins)",
        f"Target sampling rate: {TARGET_FS:g} Hz (downsampled by factor {DOWNSAMPLE_FACTOR})",
        f"Complete 180s non-overlapping windows: {len(features_df)}",
        f"Total validated breaths detected: {len(breaths_df):,}",
        f"Bandpass: {LOW_HZ} - {HIGH_HZ} Hz",
        f"Min candidate breath distance: {MIN_BREATH_DISTANCE_S} s (strictly bounded by 0.35 Hz cutoff)",
        f"Baseline reference window: Window {base_win}",
        "Paper features: rsp_mean_amplitude, rsp_std_amplitude, rsp_spectral_entropy, rsp_mean_peak_valley_magnitude, rsp_std_peak_valley_magnitude",
        "Baseline-corrected features saved with suffix _bc",
        "Plot 1 uses polyphase anti-aliased resampling for raw display."
    ]
    (OUTPUT_DIR / "rsp_run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "=" * 75)
    print("RESPIRATION FEATURE EXTRACTION SUMMARY (By Condition)")
    print("=" * 75)

    condition_summary = features_df.groupby("wesad_condition_label").agg({
        "window_index": "count",
        "rsp_mean_amplitude": "mean",
        "rsp_spectral_entropy": "mean",
        "rsp_mean_peak_valley_magnitude": "mean",
        "rsp_breath_rate_bpm_qc": "mean"
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
        "training_label_usable", "rsp_mean_amplitude", "rsp_spectral_entropy", "rsp_mean_peak_valley_magnitude", "rsp_breath_rate_bpm_qc"
    ]
    print(features_df[cols_preview].head().to_string(index=False))

    print(f"\nAll outputs saved in: {OUTPUT_DIR.resolve()}")
    print("=" * 75)
    print(">>> REFINED RESPIRATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
