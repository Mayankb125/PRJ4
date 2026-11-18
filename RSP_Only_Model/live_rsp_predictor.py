"""Live Respiration-Only (5-Feature) Random Forest Predictor + Rate-Aware Fusion.

Connects the 5 real-time respiration features + accurate breathing rate (BPM)
from live_respiration_pipeline.py into the trained 5-feature Random Forest models:
  - Binary model (81.06% LOSO accuracy): Baseline (VR=Low) vs. Stress (VR=High)
  - Optional 4-Class model (61.86% LOSO accuracy): Baseline, Stress, Amusement, Meditation

Includes:
  1. Personal resting baseline tracking (5 features + baseline_bpm)
  2. Clean Baseline Quality Check (warns if Window 1 contains belt adjustments / huge spikes)
  3. Rate-Aware Arousal Fusion (combines WESAD 5-feature RF + personal ΔBPM tachypnea check
     so rapid shallow breathing in silent VR is never missed, while logging raw RF output cleanly)
"""
from __future__ import annotations

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "results"
BINARY_MODEL_PATH = MODEL_DIR / "rsp_only_baseline_vs_stress_normalized_final_model.joblib"
FOUR_CLASS_MODEL_PATH = MODEL_DIR / "rsp_only_four_class_normalized_final_model.joblib"

RAW_FEATURE_NAMES = [
    "rsp_mean_amplitude",
    "rsp_std_amplitude",
    "rsp_spectral_entropy",
    "rsp_mean_peak_valley_magnitude",
    "rsp_std_peak_valley_magnitude",
]


def _compute_rate_stress_prob(delta_bpm: float | None) -> float:
    """Map breathing rate elevation above personal baseline (ΔBPM) to a 0-0.95 arousal probability."""
    if delta_bpm is None or not np.isfinite(delta_bpm):
        return 0.0
    if delta_bpm <= 2.5:
        return 0.0
    if delta_bpm <= 5.0:
        # Smooth ramp from 0.0 at +2.5 bpm to 0.55 at +5.0 bpm
        return round(((delta_bpm - 2.5) / 2.5) * 0.55, 4)
    if delta_bpm <= 12.0:
        # Smooth ramp from 0.55 at +5.0 bpm to 0.95 at +12.0 bpm
        return round(0.55 + ((delta_bpm - 5.0) / 7.0) * 0.40, 4)
    return 0.95


def _check_baseline_quality(raw_features: dict[str, float], baseline_bpm: float | None) -> str | None:
    """Inspect resting baseline for unusually large strap/movement spikes or non-resting BPM."""
    issues = []
    amp = float(raw_features.get("rsp_mean_amplitude", 0.0))
    depth = float(raw_features.get("rsp_mean_peak_valley_magnitude", 0.0))

    if depth > 12.0 or amp > 4.2:
        issues.append(
            f"unusually large breath depth ({depth:.1f} N) / amplitude ({amp:.1f} N) — "
            "possible belt adjustment or deep test breaths during baseline"
        )
    if baseline_bpm is not None and baseline_bpm > 22.0:
        issues.append(f"elevated resting rate ({baseline_bpm:.1f} bpm > 22 bpm)")
    elif baseline_bpm is not None and baseline_bpm < 5.0:
        issues.append(f"very low detected rate ({baseline_bpm:.1f} bpm — check belt tension)")

    if issues:
        return "Baseline Quality Warning: " + "; ".join(issues) + ". Consider sitting still and clicking Reset Baseline."
    return None


class LiveRespirationPredictor:
    """Loads the 5-feature RSP-only Random Forest models and performs baseline-normalized inference."""

    def __init__(
        self,
        binary_model_path: Path = BINARY_MODEL_PATH,
        four_class_model_path: Path = FOUR_CLASS_MODEL_PATH,
    ):
        if not binary_model_path.is_file():
            raise FileNotFoundError(f"Trained 5-feature binary model not found: {binary_model_path}")

        self.binary_bundle = joblib.load(binary_model_path)
        self.binary_model = self.binary_bundle["model"]
        self.binary_model.n_jobs = 1  # Instant (<2ms) single-row live inference with zero joblib thread overhead
        self.binary_features = list(self.binary_bundle["features"])
        self.binary_classes = list(self.binary_model.classes_)

        self.four_bundle = None
        self.four_model = None
        self.four_classes = []
        if four_class_model_path.is_file():
            self.four_bundle = joblib.load(four_class_model_path)
            self.four_model = self.four_bundle["model"]
            self.four_model.n_jobs = 1
            self.four_classes = list(self.four_model.classes_)

        self.baseline: dict[str, float] | None = None
        self.baseline_bpm: float | None = None
        self.baseline_source: str | None = None
        self.baseline_warning: str | None = None

    @property
    def has_baseline(self) -> bool:
        return self.baseline is not None

    def clear_baseline(self) -> None:
        self.baseline = None
        self.baseline_bpm = None
        self.baseline_source = None
        self.baseline_warning = None

    def set_baseline(
        self,
        raw_features: dict[str, float],
        source: str = "Live Window #1",
        baseline_bpm: float | None = None,
    ) -> dict[str, float]:
        """Lock a 5-feature raw respiration vector + resting BPM as personal baseline."""
        self.baseline = {k: float(raw_features[k]) for k in RAW_FEATURE_NAMES}
        if baseline_bpm is None and "calculated_bpm" in raw_features:
            baseline_bpm = float(raw_features["calculated_bpm"])
        self.baseline_bpm = round(float(baseline_bpm), 2) if baseline_bpm is not None else None
        self.baseline_source = source
        self.baseline_warning = _check_baseline_quality(self.baseline, self.baseline_bpm)
        return self.baseline

    def load_baseline_from_raw_csv(self, csv_path: Path, window_seconds: float = 180.0) -> dict:
        """Compute and set baseline from the first 180s of a previously recorded Vernier raw CSV."""
        live_dir = Path(__file__).resolve().parent.parent / "PhysioSync_Live"
        if str(live_dir) not in sys.path:
            sys.path.insert(0, str(live_dir))
        from live_respiration_pipeline import extract_live_respiration_features

        df = pd.read_csv(csv_path)
        times = df["elapsed_seconds"].to_numpy(dtype=float)
        forces = df["Force (N)"].to_numpy(dtype=float)

        mask = (times - times[0]) <= window_seconds
        t_sub = times[mask]
        f_sub = forces[mask]
        dur = float(t_sub[-1] - t_sub[0]) if len(t_sub) > 1 else 0.0
        fs_est = len(t_sub) / dur if dur > 0 else 20.0

        res = extract_live_respiration_features(t_sub, f_sub, fs=fs_est, min_window_s=20.0)
        if res["status"] != "OK":
            raise ValueError(f"Could not extract valid baseline from {csv_path.name}: {res['status']}")

        self.set_baseline(
            res,
            source=f"CSV: {csv_path.name} ({res['duration_seconds']}s)",
            baseline_bpm=res["calculated_bpm"],
        )
        return res

    def predict_from_raw(
        self,
        raw_features: dict[str, float],
        live_bpm: float | None = None,
    ) -> dict:
        """Subtract personal baseline and predict Stress vs. Baseline (RF + Rate-Aware Fusion)."""
        if self.baseline is None:
            raise RuntimeError("Baseline is not set yet. Complete Window #1 or load a baseline first.")

        # 1. Baseline subtraction (Bian et al. Section 2.3.2)
        norm_map: dict[str, float] = {}
        for raw_key in RAW_FEATURE_NAMES:
            norm_key = f"norm_{raw_key}"
            norm_map[norm_key] = round(float(raw_features[raw_key]) - self.baseline[raw_key], 5)

        # 2. Build ordered feature vector matching exact training column order
        X = np.array([[norm_map[col] for col in self.binary_features]], dtype=float)

        # 3. Raw 5-Feature WESAD Random Forest Prediction
        rf_pred = str(self.binary_model.predict(X)[0])
        rf_probs = self.binary_model.predict_proba(X)[0]
        rf_prob_map = {str(c): round(float(p), 4) for c, p in zip(self.binary_classes, rf_probs)}
        rf_stress_prob = rf_prob_map.get("Stress", 0.0)

        # 4. Personal Breathing Rate Delta (ΔBPM) Check
        if live_bpm is None and "calculated_bpm" in raw_features:
            live_bpm = float(raw_features["calculated_bpm"])
        delta_bpm = (
            round(float(live_bpm) - self.baseline_bpm, 2)
            if (live_bpm is not None and self.baseline_bpm is not None)
            else None
        )
        rate_stress_prob = _compute_rate_stress_prob(delta_bpm)

        # 5. Rate-Aware Arousal Fusion
        fused_stress_prob = round(max(rf_stress_prob, rate_stress_prob), 4)
        fused_baseline_prob = round(1.0 - fused_stress_prob, 4)

        if fused_stress_prob >= 0.50:
            final_pred = "Stress"
            confidence = fused_stress_prob
            if rf_stress_prob >= 0.50 and rate_stress_prob >= 0.50:
                decision_source = "Both (RF + ΔBPM)"
            elif rate_stress_prob >= 0.50:
                decision_source = "Rate_Elevation (+ΔBPM)"
            else:
                decision_source = "RF_Model"
        else:
            final_pred = "Baseline"
            confidence = fused_baseline_prob
            decision_source = "RF_Model"

        vr_state = "High" if final_pred == "Stress" else "Low"

        # 6. Optional 4-Class Prediction
        four_pred = None
        four_prob_map = {}
        if self.four_model is not None:
            four_pred = str(self.four_model.predict(X)[0])
            f_probs = self.four_model.predict_proba(X)[0]
            four_prob_map = {str(c): round(float(p), 4) for c, p in zip(self.four_classes, f_probs)}

        return {
            "predicted_label": final_pred,
            "vr_state": vr_state,
            "confidence": round(confidence, 4),
            "stress_probability": fused_stress_prob,
            "baseline_probability": fused_baseline_prob,
            "rf_predicted_label": rf_pred,
            "rf_stress_probability": rf_stress_prob,
            "rate_stress_probability": rate_stress_prob,
            "decision_source": decision_source,
            "baseline_bpm": self.baseline_bpm,
            "live_bpm": round(float(live_bpm), 2) if live_bpm is not None else None,
            "delta_bpm": delta_bpm,
            "binary_probabilities": {"Baseline": fused_baseline_prob, "Stress": fused_stress_prob},
            "four_class_label": four_pred,
            "four_class_probabilities": four_prob_map,
            "raw_features": {k: round(float(raw_features[k]), 4) for k in RAW_FEATURE_NAMES},
            "baseline_features": {k: round(float(self.baseline[k]), 4) for k in RAW_FEATURE_NAMES},
            "norm_features": norm_map,
            "baseline_source": self.baseline_source,
            "baseline_warning": self.baseline_warning,
            "model_name": self.binary_bundle.get("experiment", "rsp_only_baseline_vs_stress_normalized"),
        }
