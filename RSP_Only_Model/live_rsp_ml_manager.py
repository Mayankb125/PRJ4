"""Threaded Live Acquisition + 5-Feature Extraction + Random Forest Inference Manager.

Isolated inside D:/PRJ/RSP_Only_Model/ — leaves all old code and Page 6 untouched.
Connects:
  Vernier GDX-RB (BLE)
    -> live_respiration_pipeline.py (5 Bian et al. features)
      -> LiveRespirationPredictor (5-feature 81.06% LOSO Random Forest)
        -> Live Stress / Baseline Predictions + CSV Logging
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import csv
import sys
import threading
import time

RSP_MODEL_DIR = Path(__file__).resolve().parent
LIVE_DIR = RSP_MODEL_DIR.parent / "PhysioSync_Live"

for p in (str(RSP_MODEL_DIR), str(LIVE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gdx import gdx
from live_respiration_pipeline import extract_live_respiration_features
from live_rsp_predictor import LiveRespirationPredictor, RAW_FEATURE_NAMES

SESSIONS_DIR = RSP_MODEL_DIR / "live_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


class LiveRSPMLManager:
    """Singleton background worker orchestrating Live BLE -> 5 Features -> RF Stress Prediction."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "LiveRSPMLManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = LiveRSPMLManager()
            return cls._instance

    def __init__(self):
        self.predictor = LiveRespirationPredictor()

        self.is_running = False
        self.is_connecting = False
        self.connected = False
        self.error_message: str | None = None

        self.device_name = "Not Connected"
        self.battery_level = None
        self.sample_rate_hz = 20
        self.sample_period_ms = 50
        self.window_duration_s = 180.0

        self.raw_file: Path | None = None
        self.predictions_file: Path | None = None
        self.start_time: float | None = None
        self.final_elapsed_seconds = 0.0
        self.sample_count = 0
        self.latest_force = 0.0
        self.latest_bpm: float | None = None

        # Window & ML state
        self.completed_windows = 0
        self.window_start_elapsed = 0.0
        self.current_window_elapsed = 0.0
        self.latest_features: dict = {}
        self.latest_prediction: dict = {}

        self.buffer_lock = threading.Lock()
        self.data_buffer = deque(maxlen=2000)
        self.prediction_history = deque(maxlen=100)

        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def load_baseline_csv(self, csv_path: Path, window_seconds: float = 180.0) -> dict:
        """Load and lock a baseline from an existing Vernier raw CSV before starting live inference."""
        return self.predictor.load_baseline_from_raw_csv(csv_path, window_seconds=window_seconds)

    def clear_baseline(self) -> None:
        self.predictor.clear_baseline()

    def start(self, sample_rate_hz: int = 20, window_duration_s: float = 180.0, keep_baseline: bool = True) -> bool:
        """Start live BLE acquisition + feature extraction + ML inference thread."""
        with self._lock:
            if self.is_running or self.is_connecting:
                return False

            self.sample_rate_hz = sample_rate_hz
            self.sample_period_ms = max(10, 1000 // sample_rate_hz)
            self.window_duration_s = float(window_duration_s)
            self.is_connecting = True
            self.error_message = None
            self.battery_level = None
            self.sample_count = 0
            self.latest_force = 0.0
            self.latest_bpm = None
            self.final_elapsed_seconds = 0.0

            self.completed_windows = 0
            self.window_start_elapsed = 0.0
            self.current_window_elapsed = 0.0
            self.latest_features = {}
            self.latest_prediction = {}

            if not keep_baseline:
                self.predictor.clear_baseline()

            with self.buffer_lock:
                self.data_buffer.clear()
                self.prediction_history.clear()

            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._run_loop, daemon=True, name="LiveRSPMLWorker"
            )
            self._worker_thread.start()
            return True

    def stop(self) -> None:
        """Safely stop live acquisition and disconnect BLE."""
        if self.start_time and self.is_running:
            self.final_elapsed_seconds = time.time() - self.start_time
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        self.is_running = False
        self.is_connecting = False
        self.connected = False

    def _run_loop(self) -> None:
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        sensor = None
        raw_csv_file = None
        pred_csv_file = None

        session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        win_tag = f"{int(self.window_duration_s)}s"
        self.raw_file = SESSIONS_DIR / f"raw_{session_tag}_{self.sample_rate_hz}Hz.csv"
        self.predictions_file = SESSIONS_DIR / f"predictions_{session_tag}_{self.sample_rate_hz}Hz_{win_tag}.csv"

        win_times: list[float] = []
        win_forces: list[float] = []

        try:
            sensor = gdx.gdx()
            sensor.open(connection="ble")

            if not sensor.devices:
                self.error_message = (
                    "No Vernier Go Direct device found. Ensure GDX-RB is ON (LED flashing red), "
                    "Graphical Analysis is closed, and PC Bluetooth is ON."
                )
                self.is_connecting = False
                return

            self.connected = True
            self.is_connecting = False
            self.is_running = True

            dev_info = sensor.device_info()
            if dev_info:
                self.device_name = str(dev_info[0])
                if dev_info[2] and int(dev_info[2]) > 0:
                    self.battery_level = dev_info[2]

            try:
                sensor.select_sensors([1, 2])
            except Exception:
                sensor.select_sensors()

            headers = sensor.enabled_sensor_info() or ["Force (N)", "Respiration Rate (bpm)"]

            raw_csv_file = open(self.raw_file, mode="w", newline="", encoding="utf-8")
            raw_writer = csv.writer(raw_csv_file)
            raw_writer.writerow(["timestamp_utc", "elapsed_seconds"] + headers)
            raw_csv_file.flush()

            pred_csv_file = open(self.predictions_file, mode="w", newline="", encoding="utf-8")
            pred_writer = csv.writer(pred_csv_file)
            pred_writer.writerow([
                "timestamp_utc",
                "window_index",
                "window_start_seconds",
                "window_end_seconds",
                "is_baseline_window",
                "calculated_bpm",
                "baseline_bpm",
                "delta_bpm",
                "n_breaths",
                *RAW_FEATURE_NAMES,
                *[f"norm_{f}" for f in RAW_FEATURE_NAMES],
                "predicted_label",
                "vr_state",
                "confidence",
                "stress_probability",
                "rf_stress_probability",
                "rate_stress_probability",
                "decision_source",
                "baseline_probability",
                "four_class_label",
            ])
            pred_csv_file.flush()

            sensor.start(period=self.sample_period_ms)
            self.start_time = time.time()
            self.window_start_elapsed = 0.0
            sleep_interval = max(0.005, self.sample_period_ms / 2000.0)
            min_valid_s = max(15.0, self.window_duration_s - 5.0)

            while not self._stop_event.is_set():
                readings = sensor.read()

                if readings is not None and len(readings) > 0:
                    self.sample_count += 1
                    now_utc = datetime.now(timezone.utc).isoformat()
                    elapsed = round(time.time() - self.start_time, 3)
                    self.final_elapsed_seconds = elapsed

                    force_val = float(readings[0]) if readings[0] is not None else 0.0
                    bpm_val = None
                    if len(readings) > 1 and readings[1] is not None:
                        try:
                            v = float(readings[1])
                            if v == v:
                                bpm_val = v
                        except (ValueError, TypeError):
                            pass

                    self.latest_force = force_val
                    if bpm_val is not None:
                        self.latest_bpm = bpm_val

                    raw_writer.writerow([now_utc, elapsed] + readings)
                    if self.sample_count % 10 == 0:
                        raw_csv_file.flush()

                    with self.buffer_lock:
                        self.data_buffer.append({"elapsed": elapsed, "force": force_val, "bpm": bpm_val})

                    win_times.append(elapsed)
                    win_forces.append(force_val)
                    self.current_window_elapsed = elapsed - self.window_start_elapsed

                    # Trigger Feature Extraction + ML Prediction on completed window
                    if self.current_window_elapsed >= self.window_duration_s:
                        win_idx = self.completed_windows + 1
                        win_start_s = round(self.window_start_elapsed, 2)
                        win_end_s = round(elapsed, 2)

                        feat_res = extract_live_respiration_features(
                            win_times,
                            win_forces,
                            fs=float(self.sample_rate_hz),
                            min_window_s=min_valid_s,
                        )

                        if feat_res["status"] == "OK":
                            feat_res["window_index"] = win_idx
                            feat_res["window_start_seconds"] = win_start_s
                            feat_res["window_end_seconds"] = win_end_s

                            is_baseline_win = False
                            if not self.predictor.has_baseline:
                                self.predictor.set_baseline(
                                    feat_res,
                                    source=f"Live Window #{win_idx} ({win_start_s:.0f}s-{win_end_s:.0f}s)",
                                    baseline_bpm=feat_res["calculated_bpm"],
                                )
                                is_baseline_win = True

                            pred_res = self.predictor.predict_from_raw(
                                feat_res, live_bpm=feat_res["calculated_bpm"]
                            )
                            pred_res["window_index"] = win_idx
                            pred_res["window_start_seconds"] = win_start_s
                            pred_res["window_end_seconds"] = win_end_s
                            pred_res["is_baseline_window"] = is_baseline_win
                            pred_res["calculated_bpm"] = feat_res["calculated_bpm"]
                            pred_res["n_breaths"] = feat_res["n_breaths"]
                            pred_res["timestamp_utc"] = now_utc

                            self.completed_windows = win_idx
                            self.latest_features = feat_res
                            self.latest_prediction = pred_res

                            row_record = {
                                "timestamp_utc": now_utc,
                                "window_index": win_idx,
                                "window_start_seconds": win_start_s,
                                "window_end_seconds": win_end_s,
                                "is_baseline_window": is_baseline_win,
                                "calculated_bpm": feat_res["calculated_bpm"],
                                "baseline_bpm": pred_res["baseline_bpm"],
                                "delta_bpm": pred_res["delta_bpm"],
                                "n_breaths": feat_res["n_breaths"],
                                **{f: feat_res[f] for f in RAW_FEATURE_NAMES},
                                **{f"norm_{f}": pred_res["norm_features"][f"norm_{f}"] for f in RAW_FEATURE_NAMES},
                                "predicted_label": pred_res["predicted_label"],
                                "vr_state": pred_res["vr_state"],
                                "confidence": pred_res["confidence"],
                                "stress_probability": pred_res["stress_probability"],
                                "rf_stress_probability": pred_res["rf_stress_probability"],
                                "rate_stress_probability": pred_res["rate_stress_probability"],
                                "decision_source": pred_res["decision_source"],
                                "baseline_probability": pred_res["baseline_probability"],
                                "four_class_label": pred_res["four_class_label"],
                            }

                            with self.buffer_lock:
                                self.prediction_history.append(row_record)

                            pred_writer.writerow([
                                now_utc,
                                win_idx,
                                win_start_s,
                                win_end_s,
                                is_baseline_win,
                                feat_res["calculated_bpm"],
                                pred_res["baseline_bpm"],
                                pred_res["delta_bpm"],
                                feat_res["n_breaths"],
                                *[feat_res[f] for f in RAW_FEATURE_NAMES],
                                *[pred_res["norm_features"][f"norm_{f}"] for f in RAW_FEATURE_NAMES],
                                pred_res["predicted_label"],
                                pred_res["vr_state"],
                                pred_res["confidence"],
                                pred_res["stress_probability"],
                                pred_res["rf_stress_probability"],
                                pred_res["rate_stress_probability"],
                                pred_res["decision_source"],
                                pred_res["baseline_probability"],
                                pred_res["four_class_label"],
                            ])
                            pred_csv_file.flush()

                        # Start next non-overlapping window cleanly
                        win_times = []
                        win_forces = []
                        self.window_start_elapsed = elapsed
                        self.current_window_elapsed = 0.0

                time.sleep(sleep_interval)

        except Exception as e:
            self.error_message = f"Live ML pipeline error: {e}"
        finally:
            if raw_csv_file and not raw_csv_file.closed:
                raw_csv_file.flush()
                raw_csv_file.close()
            if pred_csv_file and not pred_csv_file.closed:
                pred_csv_file.flush()
                pred_csv_file.close()
            if sensor:
                try:
                    sensor.stop()
                except Exception:
                    pass
                try:
                    sensor.close()
                except Exception:
                    pass
            self.is_running = False
            self.is_connecting = False
            self.connected = False

    def get_snapshot(self) -> dict:
        with self.buffer_lock:
            buf_copy = list(self.data_buffer)
            hist_copy = list(self.prediction_history)

        elapsed_total = (
            time.time() - self.start_time if (self.start_time and self.is_running) else self.final_elapsed_seconds
        )

        return {
            "is_running": self.is_running,
            "is_connecting": self.is_connecting,
            "connected": self.connected,
            "error_message": self.error_message,
            "device_name": self.device_name,
            "battery_level": self.battery_level,
            "sample_rate_hz": self.sample_rate_hz,
            "window_duration_s": self.window_duration_s,
            "sample_count": self.sample_count,
            "latest_force": self.latest_force,
            "latest_bpm": self.latest_bpm,
            "elapsed_seconds": elapsed_total,
            "completed_windows": self.completed_windows,
            "current_window_elapsed": self.current_window_elapsed,
            "has_baseline": self.predictor.has_baseline,
            "baseline_features": self.predictor.baseline,
            "baseline_bpm": self.predictor.baseline_bpm,
            "baseline_source": self.predictor.baseline_source,
            "baseline_warning": self.predictor.baseline_warning,
            "latest_features": self.latest_features,
            "latest_prediction": self.latest_prediction,
            "prediction_history": hist_copy,
            "raw_file": str(self.raw_file) if self.raw_file else None,
            "predictions_file": str(self.predictions_file) if self.predictions_file else None,
            "data": buf_copy,
        }
