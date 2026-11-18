"""Live Respiration Feature Extraction Pipeline.

Real-time streaming adaptation of rsp_wesad_pipeline_final.py for Vernier GDX-RB.
Extracts all 5 target respiration features from Bian et al. (2019) Table 3:
  1. rsp_mean_amplitude: Mean magnitude of the filtered breathing waveform
  2. rsp_std_amplitude: Standard deviation of breathing wave magnitude
  3. rsp_spectral_entropy: Subband spectral entropy (0.05 - 0.35 Hz) via Welch PSD
  4. rsp_mean_peak_valley_magnitude: Mean breath depth (inhalation peak minus preceding trough)
  5. rsp_std_peak_valley_magnitude: Standard deviation of breath depths
Plus accurate breathing rate (BPM) and breath count across both calm (3-20 bpm)
and rapid/stressed breathing (21-51 bpm).
"""
from __future__ import annotations
import numpy as np

# Physiological Cutoffs
LOW_HZ = 0.05                 # ~3 breaths/min (slow deep breathing)
HIGH_HZ = 0.85                # ~51 breaths/min (captures rapid/stressed breathing without clipping)
ENTROPY_HIGH_HZ = 0.35        # Bian et al. Table 3 spectral entropy subband upper bound
MIN_BREATH_DISTANCE_S = 0.90  # Allows accurate counting up to ~60 bpm while blocking micro-jitter
PROMINENCE_SD_FRACTION = 0.25 # Fraction of window SD for robust peak/valley prominence


def extract_live_respiration_features(
    times: np.ndarray | list[float],
    forces: np.ndarray | list[float],
    fs: float = 10.0,
    min_window_s: float = 20.0
) -> dict:
    """Extract all 5 Bian et al. respiration features + accurate breath count & BPM.

    Args:
        times: Array of elapsed seconds.
        forces: Array of raw force values (N).
        fs: Effective sampling rate in Hz (e.g. 10.0, 20.0, 50.0).
        min_window_s: Minimum duration required for feature calculation.

    Returns:
        dict containing the 5 features, diagnostics, and peak/valley markers.
    """
    times_arr = np.asarray(times, dtype=np.float64)
    forces_arr = np.asarray(forces, dtype=np.float64)

    mask_finite = np.isfinite(forces_arr) & np.isfinite(times_arr)
    times_arr = times_arr[mask_finite]
    forces_arr = forces_arr[mask_finite]

    if len(forces_arr) < 10:
        return _empty_result("INSUFFICIENT_SAMPLES")

    duration_s = float(times_arr[-1] - times_arr[0]) if len(times_arr) > 1 else 0.0
    if duration_s < min_window_s:
        return _empty_result("WARMUP", duration_s=duration_s, n_samples=len(forces_arr))

    calc_fs = float(len(forces_arr) / duration_s) if duration_s > 0 else fs
    effective_fs = calc_fs if (0.5 * fs <= calc_fs <= 2.0 * fs) else fs

    # 1. Wide Respiration Bandpass Filtering (0.05 - 0.85 Hz)
    filtered = _apply_bandpass(forces_arr, effective_fs, LOW_HZ, HIGH_HZ)

    # 2. Features 1 & 2: Mean & Std Amplitude of filtered breathing wave
    abs_filt = np.abs(filtered)
    mean_amp = float(np.mean(abs_filt))
    std_amp = float(np.std(abs_filt, ddof=1)) if len(abs_filt) > 1 else 0.0

    # 3. Feature 3: Subband Spectral Entropy (0.05 - 0.35 Hz per Bian et al.)
    entropy = _compute_spectral_entropy(filtered, effective_fs, LOW_HZ, ENTROPY_HIGH_HZ)

    # 4. Features 4 & 5: Peak-Valley Inhalation/Exhalation Detection
    breaths, peak_indices, valley_indices = _detect_breaths(
        filtered, times_arr, effective_fs
    )

    if breaths:
        mags = np.asarray([b["magnitude"] for b in breaths], dtype=np.float64)
        mean_pv = float(np.mean(mags))
        std_pv = float(np.std(mags, ddof=1)) if len(mags) > 1 else 0.0
    else:
        mean_pv = 0.0
        std_pv = 0.0

    # Accurate Breath Count & Breathing Rate (BPM)
    n_breaths = int(len(peak_indices))
    if duration_s > 0:
        calculated_bpm = round(n_breaths / (duration_s / 60.0), 1)
    else:
        calculated_bpm = 0.0

    return {
        "status": "OK",
        "duration_seconds": round(duration_s, 2),
        "n_samples": int(len(forces_arr)),
        "effective_fs": round(effective_fs, 2),
        # --- THE 5 BIAN ET AL. (2019) PAPER FEATURES ---
        "rsp_mean_amplitude": round(mean_amp, 4),
        "rsp_std_amplitude": round(std_amp, 4),
        "rsp_spectral_entropy": round(entropy, 4) if np.isfinite(entropy) else 0.0,
        "rsp_mean_peak_valley_magnitude": round(mean_pv, 4),
        "rsp_std_peak_valley_magnitude": round(std_pv, 4),
        # --- CLINICAL DIAGNOSTICS ---
        "calculated_bpm": calculated_bpm,
        "n_breaths": n_breaths,
        "peak_indices": peak_indices.tolist() if isinstance(peak_indices, np.ndarray) else peak_indices,
        "valley_indices": valley_indices.tolist() if isinstance(valley_indices, np.ndarray) else valley_indices,
        "peak_times": [round(float(times_arr[p]), 2) for p in peak_indices],
        "filtered_signal": filtered,
    }


def _apply_bandpass(
    x: np.ndarray, fs: float, low_hz: float = LOW_HZ, high_hz: float = HIGH_HZ
) -> np.ndarray:
    """2nd-order Butterworth bandpass forward-backward filter with robust fallback."""
    try:
        from scipy import signal
        nyquist = 0.5 * fs
        low = max(0.001, low_hz / nyquist)
        high = min(0.999, high_hz / nyquist)
        if low < high:
            sos = signal.butter(2, [low, high], btype="bandpass", output="sos")
            return signal.sosfiltfilt(sos, x - np.mean(x))
    except Exception:
        pass

    centered = x - np.mean(x)
    smooth_len = max(3, int(round(fs * 0.25)))
    kernel = np.ones(smooth_len) / smooth_len
    return np.convolve(centered, kernel, mode="same")


def _compute_spectral_entropy(
    x: np.ndarray, fs: float, low_hz: float = LOW_HZ, high_hz: float = ENTROPY_HIGH_HZ
) -> float:
    """Welch PSD Shannon entropy normalized inside the 0.05-0.35 Hz subband."""
    try:
        from scipy import signal
        nperseg = min(len(x), int(60.0 * fs))
        if nperseg >= 8:
            f, p = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
            mask = (f >= low_hz) & (f <= high_hz)
            p_sub = p[mask]
            total_power = np.sum(p_sub)
            if total_power > 0 and len(p_sub) > 1:
                p_norm = p_sub / total_power
                entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12)) / np.log2(len(p_norm))
                return float(np.clip(entropy, 0.0, 1.0))
    except Exception:
        pass

    freqs = np.linspace(low_hz, high_hz, 31)
    n = len(x)
    t = np.arange(n) / fs
    psd = []
    for f in freqs:
        cos_part = np.sum(x * np.cos(2 * np.pi * f * t))
        sin_part = np.sum(x * np.sin(2 * np.pi * f * t))
        psd.append((cos_part ** 2 + sin_part ** 2) / n)
    psd = np.asarray(psd)
    tot = np.sum(psd)
    if tot > 0 and len(psd) > 1:
        p_norm = psd / tot
        ent = -np.sum(p_norm * np.log2(p_norm + 1e-12)) / np.log2(len(p_norm))
        return float(np.clip(ent, 0.0, 1.0))

    return 0.5


def _detect_breaths(
    x: np.ndarray, times: np.ndarray, fs: float
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Detect inhalation peaks and exhalation valleys across calm and rapid breathing."""
    dist_samples = max(2, int(MIN_BREATH_DISTANCE_S * fs))
    sd = float(np.std(x))
    prom = PROMINENCE_SD_FRACTION * sd if sd > 0 else 0.001

    peaks = []
    valleys = []

    try:
        from scipy import signal
        p_arr, _ = signal.find_peaks(x, distance=dist_samples, prominence=prom)
        v_arr, _ = signal.find_peaks(-x, distance=dist_samples, prominence=prom)
        peaks = p_arr.tolist()
        valleys = v_arr.tolist()
    except Exception:
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] > x[i + 1] and x[i] > 0.0:
                if not peaks or (i - peaks[-1]) >= dist_samples:
                    peaks.append(i)
            elif x[i] < x[i - 1] and x[i] < x[i + 1] and x[i] < 0.0:
                if not valleys or (i - valleys[-1]) >= dist_samples:
                    valleys.append(i)

    breaths = []
    valleys_arr = np.asarray(valleys)

    for p in peaks:
        prev_v = valleys_arr[valleys_arr < p]
        if len(prev_v) > 0:
            v = int(prev_v[-1])
            mag = float(x[p] - x[v])
            if mag > 0:
                breaths.append({
                    "peak_idx": p,
                    "valley_idx": v,
                    "peak_time": float(times[p]),
                    "valley_time": float(times[v]),
                    "magnitude": mag,
                })

    return breaths, np.asarray(peaks, dtype=int), np.asarray(valleys, dtype=int)


def _empty_result(status: str, duration_s: float = 0.0, n_samples: int = 0) -> dict:
    """Return default empty feature structure."""
    return {
        "status": status,
        "duration_seconds": round(duration_s, 2),
        "n_samples": n_samples,
        "effective_fs": 0.0,
        "rsp_mean_amplitude": 0.0,
        "rsp_std_amplitude": 0.0,
        "rsp_spectral_entropy": 0.0,
        "rsp_mean_peak_valley_magnitude": 0.0,
        "rsp_std_peak_valley_magnitude": 0.0,
        "calculated_bpm": 0.0,
        "n_breaths": 0,
        "peak_indices": [],
        "valley_indices": [],
        "peak_times": [],
        "filtered_signal": [],
    }
