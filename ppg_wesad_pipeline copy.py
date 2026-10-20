"""
PPG / BVP SIGNAL PROCESSING PIPELINE — WESAD
============================================

This script is organized into commented sections:
1. Load local WESAD S2.pkl
2. Inspect raw BVP
3. Compare candidate filters
4. Flag possible outliers for review (does not delete them)
5. Detect candidate pulse peaks
6. Extract two exploratory features in 3-minute windows
7. Save plots, signals, and CSV

Install in the VS Code terminal:
    python -m pip install numpy pandas matplotlib scipy

Before running, edit DATA_FILE below to match your local S2.pkl path.

Scientific cautions:
- WESAD wrist BVP is sampled at 64 Hz; the paper used toe PPG and a different setup.
- WESAD BVP amplitude is not assumed to be microvolts.
- Mean IBI is a basic HRV-related measure, not necessarily the paper's exact "mean HRV".
- WESAD has no original paper High/Low engagement labels. No labels are invented.
- Filter and peak settings are initial candidates; inspect the plots before trusting results.
- Pickle files can execute code when loaded. Only load trusted dataset files.
"""

# =============================================================================
# SECTION 1 — IMPORT LIBRARIES
# =============================================================================
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks


# =============================================================================
# SECTION 2 — USER SETTINGS
# =============================================================================
# Path to local WESAD S2.pkl file (checks project directory first, with fallback)
_DEFAULT_PATH = Path(__file__).resolve().parent / "WESAD" / "WESAD" / "S2" / "S2.pkl"
DATA_FILE = _DEFAULT_PATH if _DEFAULT_PATH.exists() else Path(r"D:\PRJ\WESAD\WESAD\S2\S2.pkl")

FS_BVP = 64.0                 # WESAD wrist BVP sampling rate (Hz)
WINDOW_SECONDS = 180          # 3-minute windows, as specified in Bian et al. (2019) Section 4.3
WINDOW_STEP_SECONDS = 180     # Paper-aligned: 180s non-overlapping windows (set to 30s for exploratory overlapping)
PLOT_START_SECONDS = 300      # Starting point for zoomed plots
PLOT_DURATION_SECONDS = 10    # Duration of zoomed plots

# Candidate bands to compare:
# - '1.1-3.6 Hz' is the exact PPG band from Bian et al. (2019)
# - '0.5-4 Hz' is a candidate wearable band that preserves low resting/meditation heart rates (down to 30 bpm)
FILTER_SETTINGS = {
    "1.1-3.6 Hz (paper exact)": (1.1, 3.6),
    "0.5-4 Hz (wearable candidate)": (0.5, 4.0),
    "0.7-4 Hz": (0.7, 4.0),
    "0.5-8 Hz": (0.5, 8.0),
}
FILTER_ORDER = 3              # Order 3 forward-backward Butterworth (effective 6th-order zero-phase)
SELECTED_FILTER = "1.1-3.6 Hz (paper exact)"  # Default to exact paper filter for reproduction benchmark

# Results will be saved in a folder next to this .py file.
OUTPUT_DIR = Path(__file__).resolve().parent / "ppg_output"

# Initial peak-detection heuristics; inspect detected peaks before using features.
MIN_PEAK_DISTANCE_SECONDS = 0.30          # Max detectable heart rate = 200 bpm
PROMINENCE_ROBUST_SPREAD_FACTOR = 0.20    # Responsive factor to capture vasoconstricted stress beats
IBI_RELATIVE_TOLERANCE = 0.25            # Rejection threshold for ectopic/spurious beat jumps (>25% from local median)

# Diagnostic only: unusually extreme samples are flagged, not automatically removed.
ROBUST_OUTLIER_THRESHOLD = 8.0


# =============================================================================
# SECTION 3 — LOAD BVP FROM A LOCAL WESAD PICKLE FILE
# =============================================================================
def load_wesad_bvp(file_path: Path) -> np.ndarray:
    """
    Read the wrist BVP channel from a standard WESAD .pkl file.
    Expected path inside the pickle: data["signal"]["wrist"]["BVP"].
    If your file uses another structure, the printed keys help you inspect it.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}\n"
            "Edit DATA_FILE in Section 2 so it points to your local S2.pkl."
        )

    print(f"Loading dataset: {file_path}")
    # Pickle can execute code while loading: only load files you trust.
    with file_path.open("rb") as file:
        data = pickle.load(file, encoding="latin1")

    if not isinstance(data, dict):
        raise TypeError("Expected the WESAD pickle to contain a dictionary.")

    try:
        bvp = data["signal"]["wrist"]["BVP"]
    except (KeyError, TypeError) as exc:
        print("Top-level keys:", list(data.keys()))
        if isinstance(data.get("signal"), dict):
            print("Signal keys:", list(data["signal"].keys()))
            wrist = data["signal"].get("wrist")
            if isinstance(wrist, dict):
                print("Wrist channel keys:", list(wrist.keys()))
        raise KeyError(
            "Could not find data['signal']['wrist']['BVP']; inspect the keys above."
        ) from exc

    bvp = np.asarray(bvp, dtype=np.float64).squeeze()
    if bvp.ndim != 1 or bvp.size == 0:
        raise ValueError(f"Expected a non-empty 1D BVP signal; got shape {bvp.shape}.")
    return bvp


# =============================================================================
# SECTION 4 — INSPECT RAW DATA BEFORE FILTERING
# =============================================================================
def inspect_raw_signal(raw: np.ndarray, fs: float) -> None:
    """Print basic statistics and save plots of the original, unchanged signal."""
    duration = len(raw) / fs
    finite = np.isfinite(raw)

    print("\\n" + "=" * 70)
    print("RAW BVP SUMMARY")
    print("=" * 70)
    print(f"Samples:            {len(raw):,}")
    print(f"Sampling rate:      {fs:g} Hz")
    print(f"Duration:           {duration:.2f} s ({duration / 60:.2f} min)")
    print(f"NaN values:         {np.isnan(raw).sum():,}")
    print(f"+Infinity values:   {np.isposinf(raw).sum():,}")
    print(f"-Infinity values:   {np.isneginf(raw).sum():,}")

    if not finite.all():
        raise ValueError(
            "Raw BVP contains NaN/Inf. Investigate and explicitly handle these "
            "before filtering; this script will not silently fill them."
        )

    print(f"Minimum:            {np.min(raw):.4f}")
    print(f"Maximum:            {np.max(raw):.4f}")
    print(f"Mean:               {np.mean(raw):.4f}")
    print(f"Median:             {np.median(raw):.4f}")
    print(f"Standard deviation: {np.std(raw):.4f}")
    print(f"1st percentile:     {np.percentile(raw, 1):.4f}")
    print(f"99th percentile:    {np.percentile(raw, 99):.4f}")

    time = np.arange(len(raw)) / fs
    plt.figure(figsize=(15, 4))
    plt.plot(time, raw, linewidth=0.5)
    plt.title("Raw wrist BVP — complete recording")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Recorded BVP value (units not assumed)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_raw_bvp_full.png", dpi=150)
    plt.close()

    start = max(0, int(PLOT_START_SECONDS * fs))
    end = min(len(raw), int((PLOT_START_SECONDS + PLOT_DURATION_SECONDS) * fs))
    if end > start:
        plt.figure(figsize=(15, 4))
        plt.plot(time[start:end], raw[start:end], linewidth=0.8)
        plt.title(f"Raw BVP — {start / fs:.1f} to {end / fs:.1f} seconds")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Recorded BVP value")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "02_raw_bvp_zoom.png", dpi=150)
        plt.close()


# =============================================================================
# SECTION 5 — APPLY AND COMPARE CANDIDATE FILTERS
# =============================================================================
def apply_bandpass(raw: np.ndarray, fs: float, low_hz: float,
                   high_hz: float, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth bandpass using forward-backward filtering.
    This is zero-phase offline filtering; it is not a causal real-time method.
    """
    nyquist = fs / 2
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Invalid band {low_hz}-{high_hz} Hz for {fs} Hz sampling. "
            f"Cutoffs must be below Nyquist ({nyquist} Hz)."
        )
    sos = butter(
        N=order, Wn=[low_hz, high_hz], btype="bandpass",
        fs=fs, output="sos"
    )
    return sosfiltfilt(sos, raw)


def compare_filters(raw: np.ndarray, fs: float) -> dict:
    """Filter the same raw signal four ways and plot the same time interval."""
    filtered = {}
    for name, (low, high) in FILTER_SETTINGS.items():
        print(f"Applying candidate filter: {name}")
        filtered[name] = apply_bandpass(raw, fs, low, high, FILTER_ORDER)

    start = max(0, int(PLOT_START_SECONDS * fs))
    end = min(len(raw), int((PLOT_START_SECONDS + PLOT_DURATION_SECONDS) * fs))
    if end <= start:
        start, end = 0, min(len(raw), int(PLOT_DURATION_SECONDS * fs))
    time = np.arange(start, end) / fs

    fig, axes = plt.subplots(len(filtered) + 1, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(time, raw[start:end], linewidth=0.8)
    axes[0].set_title("Raw BVP — no filtering")
    axes[0].set_ylabel("Recorded value")
    axes[0].grid(True, alpha=0.3)

    for ax, (name, signal) in zip(axes[1:], filtered.items()):
        ax.plot(time, signal[start:end], linewidth=0.8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_filter_comparison.png", dpi=150)
    plt.close()

    summary = pd.DataFrame([
        {
            "filter": name,
            "minimum": float(np.min(signal)),
            "maximum": float(np.max(signal)),
            "mean": float(np.mean(signal)),
            "standard_deviation": float(np.std(signal)),
        }
        for name, signal in filtered.items()
    ])
    print("\nFilter summary (descriptive only; not enough to choose a winner):")
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_DIR / "filter_summary.csv", index=False)
    return filtered


# =============================================================================
# SECTION 6 — FLAG POSSIBLE EXTREME VALUES FOR MANUAL REVIEW
# =============================================================================
def report_possible_outliers(raw: np.ndarray, fs: float) -> None:
    """
    Use a robust spread check to flag extreme samples for inspection.
    Flagging does NOT prove a sample is an artifact, so nothing is deleted.
    """
    median = np.median(raw)
    mad = np.median(np.abs(raw - median))
    print("\n" + "=" * 70)
    print("POSSIBLE OUTLIERS — REVIEW ONLY")
    print("=" * 70)

    if mad == 0:
        print("Median absolute deviation is zero; check skipped.")
        return

    robust_z = np.abs(raw - median) / (1.4826 * mad)
    flagged = np.flatnonzero(robust_z > ROBUST_OUTLIER_THRESHOLD)
    print(f"Samples flagged for review: {len(flagged):,}")
    print("No samples have been removed or changed.")

    if len(flagged):
        shown = flagged[:20]
        table = pd.DataFrame({
            "sample_index": shown,
            "time_seconds": shown / fs,
            "raw_value": raw[shown],
            "robust_z_score": robust_z[shown],
        })
        print("\nFirst flagged samples:")
        print(table.to_string(index=False))

    time = np.arange(len(raw)) / fs
    plt.figure(figsize=(15, 4))
    plt.plot(time, raw, linewidth=0.45, label="Raw BVP")
    if len(flagged):
        plt.scatter(time[flagged], raw[flagged], s=8, label="Flagged for review")
    plt.title("Raw BVP — possible extreme samples highlighted")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Recorded BVP value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_possible_outliers.png", dpi=150)
    plt.close()


# =============================================================================
# SECTION 7 — DETECT CANDIDATE PULSE PEAKS
# =============================================================================
def detect_pulse_peaks(filtered: np.ndarray, fs: float) -> np.ndarray:
    """
    Find candidate peaks with heuristic distance/prominence settings.
    These detections must be visually checked before trusting any feature.
    """
    median = np.median(filtered)
    mad = np.median(np.abs(filtered - median))
    robust_spread = 1.4826 * mad
    if robust_spread <= 0:
        raise ValueError("Filtered signal has no usable robust spread.")

    minimum_distance = max(1, int(MIN_PEAK_DISTANCE_SECONDS * fs))
    prominence = PROMINENCE_ROBUST_SPREAD_FACTOR * robust_spread
    peaks, _ = find_peaks(
        filtered, distance=minimum_distance, prominence=prominence
    )

    print("\n" + "=" * 70)
    print("EXPLORATORY PEAK DETECTION")
    print("=" * 70)
    print(f"Candidate peaks detected: {len(peaks):,}")
    if len(peaks) > 1:
        ibi_seconds = np.diff(peaks) / fs
        print(f"Median IBI: {np.median(ibi_seconds) * 1000:.1f} ms")
        print(f"Median implied heart rate: {60 / np.median(ibi_seconds):.1f} bpm")
        print("Check the peak plot for missed or extra detections.")

    start = max(0, int(PLOT_START_SECONDS * fs))
    end = min(len(filtered), int((PLOT_START_SECONDS + PLOT_DURATION_SECONDS) * fs))
    if end <= start:
        start, end = 0, min(len(filtered), int(PLOT_DURATION_SECONDS * fs))
    local_peaks = peaks[(peaks >= start) & (peaks < end)]

    plt.figure(figsize=(15, 4))
    plt.plot(np.arange(start, end) / fs, filtered[start:end],
             linewidth=0.8, label="Filtered BVP")
    plt.scatter(local_peaks / fs, filtered[local_peaks], s=28, marker="x",
                label="Candidate peaks")
    plt.title(f"Candidate pulse peaks — {SELECTED_FILTER}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Filtered BVP value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_candidate_peaks.png", dpi=150)
    plt.close()
    return peaks


# =============================================================================
# SECTION 8 — EXTRACT PPG FEATURES IN 3-MINUTE WINDOWS
# =============================================================================
def extract_window_features(filtered: np.ndarray, peaks: np.ndarray,
                            fs: float, window_seconds: int = WINDOW_SECONDS,
                            step_seconds: int = WINDOW_STEP_SECONDS) -> pd.DataFrame:
    """
    Extract PPG features in 3-minute windows.
    Default: 180s non-overlapping windows (step=180s) matching Bian et al. (2019) Section 4.3.

    1. Paper-Exact Features (Bian et al. 2019 Table 1):
       - paper_mean_hrv_ms: Mean variation in pulse interval (approximated via mean normal-to-normal IBI).
       - paper_mean_peak_amplitude: Mean pulse peak-to-trough rise (recorded units).

    2. Exploratory Extensions:
       - ext_std_ibi_ms: Standard deviation of IBIs (SDRR/SDNN proxy).
       - ext_mean_heart_rate_bpm: Heart rate in beats per minute (60,000 / mean IBI).
       - ext_std_pulse_amplitude: Std of pulse heights (captures respiratory-induced amplitude modulation).

    3. Quality & Cleaning Audit (Ensures IBI cleaning does not discard genuine HRV):
       - qc_raw_peak_count: Total detected candidate peaks in window.
       - qc_valid_ibi_count: IBIs passing range check and relative tolerance.
       - qc_rejected_ibi_count: Spurious/ectopic intervals flagged by >25% jump rule.
    """
    samples_per_window = int(round(window_seconds * fs))
    samples_per_step = int(round(step_seconds * fs))
    total_samples = len(filtered)

    if total_samples < samples_per_window:
        raise ValueError("Signal is shorter than one complete window.")

    num_windows = (total_samples - samples_per_window) // samples_per_step + 1

    rows = []
    for window_index in range(num_windows):
        start = window_index * samples_per_step
        end = start + samples_per_window
        win_peaks = peaks[(peaks >= start) & (peaks < end)]

        # --- Inter-Beat Interval (IBI) Processing ---
        raw_ibi_count = 0
        rejected_count = 0
        valid_ibi = np.array([])

        if len(win_peaks) > 1:
            ibi_ms = np.diff(win_peaks) / fs * 1000.0
            raw_ibi_count = len(ibi_ms)

            # Physiological range check: 300 ms (200 bpm) to 2000 ms (30 bpm)
            in_range = ibi_ms[(ibi_ms >= 300.0) & (ibi_ms <= 2000.0)]

            # Relative jump check: flag intervals jumping >25% from local median
            if len(in_range) > 3:
                med_ibi = float(np.median(in_range))
                if med_ibi > 0:
                    rel_err = np.abs(in_range - med_ibi) / med_ibi
                    is_valid = rel_err <= IBI_RELATIVE_TOLERANCE
                    valid_ibi = in_range[is_valid]
                    rejected_count = int(np.sum(~is_valid)) + (raw_ibi_count - len(in_range))
                else:
                    valid_ibi = in_range
            else:
                valid_ibi = in_range

        # Feature calculations
        mean_ibi = float(np.mean(valid_ibi)) if len(valid_ibi) else np.nan
        std_ibi = float(np.std(valid_ibi)) if len(valid_ibi) else np.nan
        mean_hr = float(60000.0 / mean_ibi) if (np.isfinite(mean_ibi) and mean_ibi > 0) else np.nan

        # --- Pulse Amplitude Processing (Peak to preceding trough) ---
        amplitudes = []
        for previous, current in zip(win_peaks[:-1], win_peaks[1:]):
            if current <= previous + 1:
                continue
            trough_index = previous + int(np.argmin(filtered[previous:current + 1]))
            amplitude = filtered[current] - filtered[trough_index]
            if np.isfinite(amplitude) and amplitude > 0:
                amplitudes.append(float(amplitude))

        mean_amp = float(np.mean(amplitudes)) if len(amplitudes) else np.nan
        std_amp = float(np.std(amplitudes)) if len(amplitudes) else np.nan

        rows.append({
            "subject_id": "S2",
            "window_index": window_index,
            "window_start_seconds": start / fs,
            "window_end_seconds": end / fs,
            # 1. Paper-aligned features (Table 1)
            "paper_mean_hrv_ms": mean_ibi,
            "paper_mean_peak_amplitude": mean_amp,
            # 2. Exploratory extensions
            "ext_mean_heart_rate_bpm": mean_hr,
            "ext_std_ibi_ms": std_ibi,
            "ext_std_pulse_amplitude": std_amp,
            # 3. Quality & validation metrics
            "qc_raw_peak_count": int(len(win_peaks)),
            "qc_valid_ibi_count": int(len(valid_ibi)),
            "qc_rejected_ibi_count": rejected_count,
            "qc_amplitude_count": int(len(amplitudes)),
            # Placeholder for ground-truth label integration in next stage
            "label": np.nan,
        })

    features = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("WINDOW FEATURE SUMMARY (Paper-Aligned: 3-minute windows)")
    print("=" * 70)
    print(f"Window duration:      {window_seconds} seconds")
    print(f"Window step size:     {step_seconds} seconds (overlap = {max(0, window_seconds - step_seconds)}s)")
    print(f"Total windows:        {len(features)}")
    print(f"Missing Paper Mean HRV:       {features['paper_mean_hrv_ms'].isna().sum()}")
    print(f"Missing Paper Peak Amplitude: {features['paper_mean_peak_amplitude'].isna().sum()}")
    print(f"Total rejected IBI samples:   {features['qc_rejected_ibi_count'].sum()}")
    print("\nFirst 10 windows:")
    print(features[["window_index", "paper_mean_hrv_ms", "paper_mean_peak_amplitude",
                    "ext_mean_heart_rate_bpm", "qc_valid_ibi_count", "qc_rejected_ibi_count"]].head(10).to_string(index=False))
    return features


# =============================================================================
# SECTION 9 — SAVE INTERMEDIATE SIGNALS AND FEATURES
# =============================================================================
def save_outputs(raw: np.ndarray, filtered: np.ndarray, peaks: np.ndarray,
                 features: pd.DataFrame) -> None:
    """Save outputs so results can be reviewed and reused later."""
    np.save(OUTPUT_DIR / "bvp_raw.npy", raw)
    np.save(OUTPUT_DIR / "bvp_filtered_selected.npy", filtered)
    np.save(OUTPUT_DIR / "bvp_candidate_peak_indices.npy", peaks)
    features.to_csv(OUTPUT_DIR / "ppg_features_3min.csv", index=False)

    print("\nSaved outputs in:", OUTPUT_DIR.resolve())
    print("  bvp_raw.npy")
    print("  bvp_filtered_selected.npy")
    print("  bvp_candidate_peak_indices.npy")
    print("  ppg_features_3min.csv")
    print("  filter_summary.csv and diagnostic PNG plots")


# =============================================================================
# SECTION 10 — RUN THE PIPELINE
# =============================================================================
def main() -> None:
    """Run the workflow in order, from raw signal to exploratory feature CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: load the original BVP channel from the local pickle file.
    raw = load_wesad_bvp(DATA_FILE)

    # Step 2: inspect the unmodified signal before making processing decisions.
    inspect_raw_signal(raw, FS_BVP)

    # Step 3: compare candidate filter bands on the same signal segment.
    filtered_signals = compare_filters(raw, FS_BVP)

    # Step 4: flag possible extreme values for review; do not delete them.
    report_possible_outliers(raw, FS_BVP)

    # Step 5: choose a candidate filter (review plots before treating it as final).
    if SELECTED_FILTER not in filtered_signals:
        raise ValueError(
            f"Unknown SELECTED_FILTER. Choose one of: {list(filtered_signals)}"
        )
    selected = filtered_signals[SELECTED_FILTER]

    # Step 6: detect peaks and inspect the candidate peak plot.
    peaks = detect_pulse_peaks(selected, FS_BVP)

    # Step 7: extract features in 3-minute windows (paper-aligned 180s non-overlapping step by default).
    features = extract_window_features(
        selected, peaks, FS_BVP, WINDOW_SECONDS, WINDOW_STEP_SECONDS
    )

    # Step 8: save all key outputs for the next stage of analysis.
    save_outputs(raw, selected, peaks, features)

    print("\nPipeline finished.")
    print(
        "Important: inspect the plots and validate feature definitions before "
        "using the CSV for Random Forest. Valid target labels are still required."
    )


if __name__ == "__main__":
    main()
