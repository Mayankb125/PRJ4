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
def detect_pulse_peaks(filtered: np.ndarray, fs: float,
                       min_distance_seconds: float = MIN_PEAK_DISTANCE_SECONDS,
                       save_plot_name: str = "05_candidate_peaks.png") -> np.ndarray:
    """
    Find candidate peaks with configurable distance/prominence settings.
    These detections must be visually checked before trusting any feature.
    """
    median = np.median(filtered)
    mad = np.median(np.abs(filtered - median))
    robust_spread = 1.4826 * mad
    if robust_spread <= 0:
        raise ValueError("Filtered signal has no usable robust spread.")

    minimum_distance = max(1, int(min_distance_seconds * fs))
    prominence = PROMINENCE_ROBUST_SPREAD_FACTOR * robust_spread
    peaks, _ = find_peaks(
        filtered, distance=minimum_distance, prominence=prominence
    )

    max_bpm = 60.0 / min_distance_seconds
    print("\n" + "=" * 70)
    print(f"PEAK DETECTION (Min distance = {min_distance_seconds:.2f}s, Max pulse rate = {max_bpm:.1f} bpm)")
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
    plt.title(f"Candidate pulse peaks (Min Dist: {min_distance_seconds:.2f}s) — {SELECTED_FILTER}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Filtered BVP value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_plot_name, dpi=150)
    plt.close()
    return peaks


# =============================================================================
# SECTION 8 — EXTRACT PPG FEATURES IN 3-MINUTE WINDOWS
# =============================================================================
def extract_window_features(filtered: np.ndarray, peaks: np.ndarray,
                            fs: float, window_seconds: int = WINDOW_SECONDS,
                            step_seconds: int = WINDOW_STEP_SECONDS,
                            cleaning_method: str = "static_25",
                            tolerance: float = IBI_RELATIVE_TOLERANCE) -> pd.DataFrame:
    """
    Extract PPG features in 3-minute windows.
    Default: 180s non-overlapping windows (step=180s) matching Bian et al. (2019) Section 4.3.

    Cleaning methods supported:
    - 'static_25'   : Rejects intervals deviating > tolerance (25%) from static 3-min window median (Baseline v1).
    - 'local_moving': Rejects intervals deviating > tolerance from local moving median (neighborhood of 9 beats).
    - 'uncleaned'   : Keeps all physiologically plausible intervals (300-2000 ms) with no jump filtering.
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
            rejected_range = raw_ibi_count - len(in_range)

            if len(in_range) > 3:
                if cleaning_method == "static_25":
                    # Baseline v1: static window median comparison
                    med_ibi = float(np.median(in_range))
                    if med_ibi > 0:
                        rel_err = np.abs(in_range - med_ibi) / med_ibi
                        is_valid = rel_err <= tolerance
                        valid_ibi = in_range[is_valid]
                        rejected_count = int(np.sum(~is_valid)) + rejected_range
                    else:
                        valid_ibi = in_range
                elif cleaning_method == "local_moving":
                    # Candidate v2: local moving neighborhood median (9 beats: 4 before, 4 after)
                    is_valid = np.ones(len(in_range), dtype=bool)
                    for i in range(len(in_range)):
                        nb_start = max(0, i - 4)
                        nb_end = min(len(in_range), i + 5)
                        local_med = float(np.median(in_range[nb_start:nb_end]))
                        if local_med > 0:
                            rel_err = abs(in_range[i] - local_med) / local_med
                            if rel_err > tolerance:
                                is_valid[i] = False
                    valid_ibi = in_range[is_valid]
                    rejected_count = int(np.sum(~is_valid)) + rejected_range
                elif cleaning_method == "uncleaned":
                    # Full physiological transparency without relative jump deletion
                    valid_ibi = in_range
                    rejected_count = rejected_range
                else:
                    raise ValueError(f"Unknown cleaning_method: {cleaning_method}")
            else:
                valid_ibi = in_range
                rejected_count = rejected_range

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
            "mean_ibi_ms": mean_ibi,
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
    print(f"Missing Mean IBI:             {features['mean_ibi_ms'].isna().sum()}")
    print(f"Missing Paper Peak Amplitude: {features['paper_mean_peak_amplitude'].isna().sum()}")
    print(f"Total rejected IBI samples:   {features['qc_rejected_ibi_count'].sum()}")
    print("\nFirst 10 windows:")
    print(features[["window_index", "mean_ibi_ms", "paper_mean_peak_amplitude",
                    "ext_mean_heart_rate_bpm", "qc_valid_ibi_count", "qc_rejected_ibi_count"]].head(10).to_string(index=False))
    return features


# =============================================================================
# SECTION 8B — PEAK DISAGREEMENT INSPECTOR (0.30s vs 0.35s)
# =============================================================================
def inspect_peak_disagreements(filtered: np.ndarray, fs: float,
                               peaks_a: np.ndarray, peaks_b: np.ndarray) -> pd.DataFrame:
    """
    Directly identify and plot locations where 0.30s (Config A) and 0.35s (Config B) disagree.
    Finds peaks detected by Config A that were suppressed/removed by Config B.
    """
    removed_mask = ~np.isin(peaks_a, peaks_b)
    removed_indices = peaks_a[removed_mask]

    print("\n" + "=" * 75)
    print(">>> STEP 7A: PEAK DISAGREEMENT AUDIT (0.30s vs 0.35s)")
    print("=" * 75)
    print(f"Total peaks in Config A (0.30s): {len(peaks_a):,}")
    print(f"Total peaks in Config B (0.35s): {len(peaks_b):,}")
    print(f"Peaks removed in Config B:       {len(removed_indices):,}")

    details = []
    for idx in removed_indices:
        t_sec = idx / fs
        prev_peaks = peaks_a[peaks_a < idx]
        next_peaks = peaks_a[peaks_a > idx]
        dist_prev_ms = (idx - prev_peaks[-1]) / fs * 1000.0 if len(prev_peaks) else np.nan
        dist_next_ms = (next_peaks[0] - idx) / fs * 1000.0 if len(next_peaks) else np.nan
        details.append({
            "peak_sample_index": idx,
            "time_seconds": round(t_sec, 3),
            "amplitude": round(float(filtered[idx]), 3),
            "distance_to_prev_peak_ms": round(dist_prev_ms, 1) if np.isfinite(dist_prev_ms) else np.nan,
            "distance_to_next_peak_ms": round(dist_next_ms, 1) if np.isfinite(dist_next_ms) else np.nan,
        })
    df_removed = pd.DataFrame(details)
    df_removed.to_csv(OUTPUT_DIR / "removed_peaks_details.csv", index=False)
    print(f"[OUTPUT] Saved details of removed peaks to: {OUTPUT_DIR / 'removed_peaks_details.csv'}")

    # Plot 3 representative segments where disagreements happen
    if len(removed_indices) > 0:
        times = removed_indices / fs
        # Pick 3 target times spaced out across the recording (e.g. 20%, 50%, 80%)
        q1 = times[len(times) // 5]
        q2 = times[len(times) // 2]
        q3 = times[4 * len(times) // 5]
        target_centers = [q1, q2, q3]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        for i, (ax, center_t) in enumerate(zip(axes, target_centers)):
            start_t = max(0, center_t - 5.0)
            end_t = min(len(filtered) / fs, center_t + 5.0)
            s_idx = int(start_t * fs)
            e_idx = int(end_t * fs)
            t_axis = np.arange(s_idx, e_idx) / fs

            in_b = peaks_b[(peaks_b >= s_idx) & (peaks_b < e_idx)]
            in_rem = removed_indices[(removed_indices >= s_idx) & (removed_indices < e_idx)]

            ax.plot(t_axis, filtered[s_idx:e_idx], color="gray", linewidth=1.0, label="Filtered BVP")
            if len(in_b):
                ax.scatter(in_b / fs, filtered[in_b], color="blue", s=50, marker="o", label="Kept in Config B (0.35s)")
            if len(in_rem):
                ax.scatter(in_rem / fs, filtered[in_rem], color="red", s=70, marker="x", linewidths=2.0, label="Removed in Config B (only in 0.30s)")

            ax.set_title(f"Disagreement Segment #{i+1}: {start_t:.1f}s to {end_t:.1f}s (Red X = Disappeared Peak)")
            ax.set_ylabel("Filtered BVP")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time (seconds)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "06_peak_disagreements_030_vs_035.png", dpi=150)
        plt.close()
        print(f"[OUTPUT] Saved disagreement plot to: {OUTPUT_DIR / '06_peak_disagreements_030_vs_035.png'}")

    return df_removed


# =============================================================================
# SECTION 8C — BEAT-BY-BEAT IBI CLEANING AUDIT
# =============================================================================
def audit_ibi_cleaning(peaks: np.ndarray, fs: float, window_seconds: int = WINDOW_SECONDS,
                       step_seconds: int = WINDOW_STEP_SECONDS,
                       tolerance: float = IBI_RELATIVE_TOLERANCE) -> pd.DataFrame:
    """
    Audit every single inter-beat interval (IBI), recording exact value, implied BPM,
    and the specific reason if flagged/rejected.
    """
    samples_per_window = int(round(window_seconds * fs))
    samples_per_step = int(round(step_seconds * fs))

    records = []
    ibi_index = 0
    for prev_p, curr_p in zip(peaks[:-1], peaks[1:]):
        ibi_index += 1
        ibi_ms = (curr_p - prev_p) / fs * 1000.0
        peak_time_sec = curr_p / fs
        implied_bpm = 60000.0 / ibi_ms if ibi_ms > 0 else np.nan

        # Determine which 3-minute window this beat falls into
        win_idx = int(curr_p // samples_per_step)
        win_start = win_idx * samples_per_step
        win_end = win_start + samples_per_window

        # Get window peaks to compute window median
        w_peaks = peaks[(peaks >= win_start) & (peaks < win_end)]
        if len(w_peaks) > 1:
            w_ibis = np.diff(w_peaks) / fs * 1000.0
            w_valid = w_ibis[(w_ibis >= 300.0) & (w_ibis <= 2000.0)]
            w_med = float(np.median(w_valid)) if len(w_valid) else np.nan
        else:
            w_med = np.nan

        status = "VALID"
        reason = "Passed all physiological and relative variance checks"

        if ibi_ms < 300.0:
            status = "FLAGGED_TOO_FAST"
            reason = f"Interval < 300 ms (implied {implied_bpm:.1f} bpm > 200 bpm limit)"
        elif ibi_ms > 2000.0:
            status = "FLAGGED_TOO_SLOW"
            reason = f"Interval > 2000 ms (implied {implied_bpm:.1f} bpm < 30 bpm limit)"
        elif np.isfinite(w_med) and w_med > 0:
            rel_err = abs(ibi_ms - w_med) / w_med
            if rel_err > tolerance:
                status = "FLAGGED_RELATIVE_JUMP"
                pct_dev = rel_err * 100.0
                reason = f"Jump of {pct_dev:.1f}% from window median ({w_med:.1f} ms) exceeds {tolerance*100:.0f}% tolerance"

        records.append({
            "interval_index": ibi_index,
            "peak_time_seconds": round(peak_time_sec, 3),
            "window_index": win_idx,
            "ibi_ms": round(ibi_ms, 2),
            "implied_bpm": round(implied_bpm, 1),
            "window_median_ibi_ms": round(w_med, 2) if np.isfinite(w_med) else np.nan,
            "status": status,
            "reason": reason
        })

    df_audit = pd.DataFrame(records)
    df_audit.to_csv(OUTPUT_DIR / "ibi_cleaning_audit_035s.csv", index=False)
    print("\n" + "=" * 75)
    print(">>> STEP 7B: BEAT-BY-BEAT IBI AUDIT BREAKDOWN (Config B: 0.35s)")
    print("=" * 75)
    counts = df_audit["status"].value_counts()
    for st, cnt in counts.items():
        pct = cnt / len(df_audit) * 100.0
        print(f"  {st:<28} : {cnt:,} intervals ({pct:.1f}%)")
    print(f"[OUTPUT] Saved beat-by-beat IBI audit to: {OUTPUT_DIR / 'ibi_cleaning_audit_035s.csv'}")

# =============================================================================
# SECTION 8D — WAVEFORM PATTERN INSPECTOR (Pattern A vs Pattern B)
# =============================================================================
def plot_patterns_inspection(filtered: np.ndarray, fs: float, peaks: np.ndarray) -> None:
    """
    Directly inspect the pulse waveform for:
    - Pattern A (4.5s to 11.5s): Gradual physiological change (breathing wave)
    - Pattern B (11.5s to 17.0s): Abrupt disruption (motion / artifact)
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    segments = [
        ("Pattern A: Gradual Physiological Shift (Seconds 4.5 - 11.5)", 4.5, 11.5, "blue"),
        ("Pattern B: Abrupt Disruption / Motion Disturbance (Seconds 11.5 - 17.0)", 11.5, 17.0, "red"),
    ]

    for ax, (title, t_start, t_end, color) in zip(axes, segments):
        s_idx = int(t_start * fs)
        e_idx = int(t_end * fs)
        t_axis = np.arange(s_idx, e_idx) / fs

        ax.plot(t_axis, filtered[s_idx:e_idx], color="gray", linewidth=1.2, label="Filtered BVP")

        p_in = peaks[(peaks >= s_idx) & (peaks < e_idx)]
        if len(p_in):
            ax.scatter(p_in / fs, filtered[p_in], color=color, s=60, zorder=5, label="Detected Peaks")

            # Annotate interval lengths between peaks
            for p1, p2 in zip(p_in[:-1], p_in[1:]):
                gap_ms = (p2 - p1) / fs * 1000.0
                mid_t = (p1 + p2) / (2.0 * fs)
                y_pos = (filtered[p1] + filtered[p2]) / 2.0 + 15.0
                ax.annotate(f"{gap_ms:.0f}ms", xy=(mid_t, y_pos),
                            ha="center", fontsize=9, fontweight="bold", color="black",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.6))

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Filtered BVP")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_waveform_inspection_patternA_vs_patternB.png", dpi=150)
    plt.close()
    print(f"[OUTPUT] Saved Pattern A vs Pattern B inspection plot to: {OUTPUT_DIR / '07_waveform_inspection_patternA_vs_patternB.png'}")


# =============================================================================
# SECTION 9 — SAVE INTERMEDIATE SIGNALS AND FEATURES
# =============================================================================
def save_outputs(raw: np.ndarray, filtered: np.ndarray, peaks: np.ndarray,
                 features: pd.DataFrame, csv_filename: str = "ppg_features_3min.csv") -> None:
    """Save outputs so results can be reviewed and reused later."""
    np.save(OUTPUT_DIR / "bvp_raw.npy", raw)
    np.save(OUTPUT_DIR / "bvp_filtered_selected.npy", filtered)
    np.save(OUTPUT_DIR / "bvp_candidate_peak_indices.npy", peaks)
    features.to_csv(OUTPUT_DIR / csv_filename, index=False)
    if csv_filename != "ppg_features_3min.csv":
        features.to_csv(OUTPUT_DIR / "ppg_features_3min.csv", index=False)

    print(f"[OUTPUT] Saved feature CSV: {OUTPUT_DIR / csv_filename}")


# =============================================================================
# SECTION 10 — RUN THE PIPELINE WITH CONTROLLED TEST
# =============================================================================
def main() -> None:
    """Run the workflow in order, comparing baseline (0.30s) vs test (0.35s) and cleaning methods."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(">>> STEP 1/7: Loading raw BVP from WESAD S2...")
    print("=" * 75)
    raw = load_wesad_bvp(DATA_FILE)

    print("\n" + "=" * 75)
    print(">>> STEP 2/7: Inspecting unmodified BVP signal...")
    print("=" * 75)
    inspect_raw_signal(raw, FS_BVP)

    print("\n" + "=" * 75)
    print(">>> STEP 3/7: Comparing candidate bandpass filters...")
    print("=" * 75)
    filtered_signals = compare_filters(raw, FS_BVP)

    print("\n" + "=" * 75)
    print(">>> STEP 4/7: Checking for extreme outlier spikes...")
    print("=" * 75)
    report_possible_outliers(raw, FS_BVP)

    print("\n" + "=" * 75)
    print(f">>> STEP 5/7: Selecting baseline filter: '{SELECTED_FILTER}'")
    print("=" * 75)
    if SELECTED_FILTER not in filtered_signals:
        raise ValueError(
            f"Unknown SELECTED_FILTER. Choose one of: {list(filtered_signals)}"
        )
    selected = filtered_signals[SELECTED_FILTER]

    # --- CONFIGURATION A: Baseline v1 (Min Distance = 0.30s, max 200 bpm) ---
    print("\n" + "=" * 75)
    print(">>> STEP 6A: Running Configuration A (Baseline v1: Min Peak Distance = 0.30s)...")
    print("=" * 75)
    peaks_a = detect_pulse_peaks(selected, FS_BVP, min_distance_seconds=0.30,
                                 save_plot_name="05_peaks_baseline_030s.png")
    features_a = extract_window_features(
        selected, peaks_a, FS_BVP, WINDOW_SECONDS, WINDOW_STEP_SECONDS, cleaning_method="static_25"
    )
    save_outputs(raw, selected, peaks_a, features_a, csv_filename="ppg_features_baseline_v1.csv")

    # --- CONFIGURATION B: Test 1 (Min Distance = 0.35s, max 171 bpm) ---
    print("\n" + "=" * 75)
    print(">>> STEP 6B: Running Configuration B (Controlled Test: Min Peak Distance = 0.35s)...")
    print("=" * 75)
    peaks_b = detect_pulse_peaks(selected, FS_BVP, min_distance_seconds=0.35,
                                 save_plot_name="05_peaks_test_035s.png")

    # Method 1: Static 25% (Baseline cleaning)
    features_b_static = extract_window_features(
        selected, peaks_b, FS_BVP, WINDOW_SECONDS, WINDOW_STEP_SECONDS, cleaning_method="static_25"
    )
    save_outputs(raw, selected, peaks_b, features_b_static, csv_filename="ppg_features_test_dist035.csv")

    # Method 2: Candidate v2: Local Moving Neighborhood (9 beats)
    print("\n>>> Extracting features with Candidate v2 (Local Moving Neighborhood quality check)...")
    features_b_local = extract_window_features(
        selected, peaks_b, FS_BVP, WINDOW_SECONDS, WINDOW_STEP_SECONDS, cleaning_method="local_moving"
    )
    save_outputs(raw, selected, peaks_b, features_b_local, csv_filename="ppg_features_candidate_v2_local.csv")

    # Method 3: Uncleaned (Full physiological range 300-2000ms, no jump filter)
    print("\n>>> Extracting features with Uncleaned baseline (physiological range only)...")
    features_b_uncleaned = extract_window_features(
        selected, peaks_b, FS_BVP, WINDOW_SECONDS, WINDOW_STEP_SECONDS, cleaning_method="uncleaned"
    )
    save_outputs(raw, selected, peaks_b, features_b_uncleaned, csv_filename="ppg_features_uncleaned.csv")

    # --- STEP 7A: Inspect Disagreements Between 0.30s and 0.35s ---
    inspect_peak_disagreements(selected, FS_BVP, peaks_a, peaks_b)

    # --- STEP 7B: Beat-by-Beat IBI Cleaning Audit ---
    audit_ibi_cleaning(peaks_b, FS_BVP, WINDOW_SECONDS, WINDOW_STEP_SECONDS, IBI_RELATIVE_TOLERANCE)

    # --- STEP 7C: Waveform Pattern Inspection Plot (Pattern A vs Pattern B) ---
    print("\n" + "=" * 75)
    print(">>> STEP 7C: GENERATING PATTERN A vs PATTERN B INSPECTION PLOT...")
    print("=" * 75)
    plot_patterns_inspection(selected, FS_BVP, peaks_b)

    # --- STEP 7D: Peak Distance Comparison Table ---
    print("\n" + "=" * 75)
    print(">>> STEP 7D: PEAK DISTANCE COMPARISON (0.30s vs 0.35s)")
    print("=" * 75)
    tot_peaks_a = len(peaks_a)
    tot_peaks_b = len(peaks_b)
    ibi_a = np.diff(peaks_a) / FS_BVP * 1000.0 if len(peaks_a) > 1 else np.array([])
    ibi_b = np.diff(peaks_b) / FS_BVP * 1000.0 if len(peaks_b) > 1 else np.array([])

    valid_beats_a = int(features_a["qc_valid_ibi_count"].sum())
    valid_beats_b = int(features_b_static["qc_valid_ibi_count"].sum())
    rejected_a = int(features_a["qc_rejected_ibi_count"].sum())
    rejected_b = int(features_b_static["qc_rejected_ibi_count"].sum())

    pct_rejected_a = (rejected_a / (valid_beats_a + rejected_a) * 100.0) if (valid_beats_a + rejected_a) else 0.0
    pct_rejected_b = (rejected_b / (valid_beats_b + rejected_b) * 100.0) if (valid_beats_b + rejected_b) else 0.0

    peak_comp_df = pd.DataFrame([
        {"Metric": "Minimum Peak Distance", "Config A (0.30s)": "0.30 seconds", "Config B (0.35s)": "0.35 seconds", "Difference": "+0.05 seconds"},
        {"Metric": "Max Detectable Heart Rate", "Config A (0.30s)": "200.0 bpm", "Config B (0.35s)": "171.4 bpm", "Difference": "Safe for seated tasks"},
        {"Metric": "Candidate Peaks Found", "Config A (0.30s)": f"{tot_peaks_a:,}", "Config B (0.35s)": f"{tot_peaks_b:,}", "Difference": f"{tot_peaks_b - tot_peaks_a:+,} peaks"},
        {"Metric": "Overall Median IBI", "Config A (0.30s)": f"{np.median(ibi_a):.1f} ms", "Config B (0.35s)": f"{np.median(ibi_b):.1f} ms", "Difference": f"{np.median(ibi_b) - np.median(ibi_a):+.1f} ms"},
        {"Metric": "Implied Median Heart Rate", "Config A (0.30s)": f"{60000.0 / np.median(ibi_a):.1f} bpm", "Config B (0.35s)": f"{60000.0 / np.median(ibi_b):.1f} bpm", "Difference": f"{(60000.0 / np.median(ibi_b)) - (60000.0 / np.median(ibi_a)):+.1f} bpm"},
        {"Metric": "Valid Intervals Kept", "Config A (0.30s)": f"{valid_beats_a:,}", "Config B (0.35s)": f"{valid_beats_b:,}", "Difference": f"{valid_beats_b - valid_beats_a:+,} beats"},
        {"Metric": "Rejected Intervals (25% rule)", "Config A (0.30s)": f"{rejected_a:,} ({pct_rejected_a:.1f}%)", "Config B (0.35s)": f"{rejected_b:,} ({pct_rejected_b:.1f}%)", "Difference": f"{rejected_b - rejected_a:+,} ({pct_rejected_b - pct_rejected_a:+.1f}%)"},
    ])
    print(peak_comp_df.to_string(index=False))
    peak_comp_df.to_csv(OUTPUT_DIR / "peak_distance_controlled_comparison.csv", index=False)

    # --- STEP 7E: 3-Way Cleaning Methods Controlled Comparison Table ---
    print("\n" + "=" * 75)
    print(">>> STEP 7E: CLEANING METHODS CONTROLLED COMPARISON (On Config B 0.35s Peaks)")
    print("=" * 75)

    v_unc = int(features_b_uncleaned["qc_valid_ibi_count"].sum())
    r_unc = int(features_b_uncleaned["qc_rejected_ibi_count"].sum())
    pct_r_unc = (r_unc / (v_unc + r_unc) * 100.0) if (v_unc + r_unc) else 0.0

    v_stat = int(features_b_static["qc_valid_ibi_count"].sum())
    r_stat = int(features_b_static["qc_rejected_ibi_count"].sum())
    pct_r_stat = (r_stat / (v_stat + r_stat) * 100.0) if (v_stat + r_stat) else 0.0

    v_loc = int(features_b_local["qc_valid_ibi_count"].sum())
    r_loc = int(features_b_local["qc_rejected_ibi_count"].sum())
    pct_r_loc = (r_loc / (v_loc + r_loc) * 100.0) if (v_loc + r_loc) else 0.0

    clean_comp_df = pd.DataFrame([
        {
            "Metric": "Cleaning Philosophy",
            "Method 1: Uncleaned": "No relative jump deletion",
            "Method 2: Static 25% (Baseline)": "Compare to 3-min static median",
            "Method 3: Local Moving (Candidate v2)": "Compare to 9-beat moving median"
        },
        {
            "Metric": "Valid Beats Kept",
            "Method 1: Uncleaned": f"{v_unc:,}",
            "Method 2: Static 25% (Baseline)": f"{v_stat:,}",
            "Method 3: Local Moving (Candidate v2)": f"{v_loc:,}"
        },
        {
            "Metric": "Rejected Beats Count",
            "Method 1: Uncleaned": f"{r_unc:,} ({pct_r_unc:.1f}%)",
            "Method 2: Static 25% (Baseline)": f"{r_stat:,} ({pct_r_stat:.1f}%)",
            "Method 3: Local Moving (Candidate v2)": f"{r_loc:,} ({pct_r_loc:.1f}%)"
        },
        {
            "Metric": "Mean Window IBI",
            "Method 1: Uncleaned": f"{features_b_uncleaned['mean_ibi_ms'].mean():.1f} ms",
            "Method 2: Static 25% (Baseline)": f"{features_b_static['mean_ibi_ms'].mean():.1f} ms",
            "Method 3: Local Moving (Candidate v2)": f"{features_b_local['mean_ibi_ms'].mean():.1f} ms"
        },
        {
            "Metric": "Mean Window PRV (Std IBI)",
            "Method 1: Uncleaned": f"{features_b_uncleaned['ext_std_ibi_ms'].mean():.1f} ms",
            "Method 2: Static 25% (Baseline)": f"{features_b_static['ext_std_ibi_ms'].mean():.1f} ms",
            "Method 3: Local Moving (Candidate v2)": f"{features_b_local['ext_std_ibi_ms'].mean():.1f} ms"
        },
        {
            "Metric": "Mean Peak Amplitude",
            "Method 1: Uncleaned": f"{features_b_uncleaned['paper_mean_peak_amplitude'].mean():.2f}",
            "Method 2: Static 25% (Baseline)": f"{features_b_static['paper_mean_peak_amplitude'].mean():.2f}",
            "Method 3: Local Moving (Candidate v2)": f"{features_b_local['paper_mean_peak_amplitude'].mean():.2f}"
        },
    ])

    print(clean_comp_df.to_string(index=False))
    clean_comp_df.to_csv(OUTPUT_DIR / "cleaning_methods_controlled_comparison.csv", index=False)
    print(f"\n[OUTPUT] Saved cleaning comparison table to: {OUTPUT_DIR / 'cleaning_methods_controlled_comparison.csv'}")

    print("\n" + "=" * 75)
    print(">>> PIPELINE EXECUTION FINISHED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
