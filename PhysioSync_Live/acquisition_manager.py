"""Background thread manager for Vernier GDX-RB live data acquisition.

Maintains a persistent Bluetooth Low Energy (BLE) connection, continuous
sampling loop, and strict non-overlapping 180-second (3-minute) window
extraction of the 5 Bian et al. (2019) respiration features.
"""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import csv
import sys
import threading
import time

# Ensure local modules are accessible
CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from gdx import gdx
from live_respiration_pipeline import extract_live_respiration_features

DATA_DIR = CURRENT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_DURATION_SECONDS = 180.0  # Strict 3-minute non-overlapping window (Bian et al. 2019)


class VernierAcquisitionManager:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = VernierAcquisitionManager()
            return cls._instance

    def __init__(self):
        self.is_running = False
        self.is_connecting = False
        self.connected = False
        self.error_message = None

        self.device_name = "Not Connected"
        self.battery_level = None
        self.sample_rate_hz = 10
        self.sample_period_ms = 100

        self.output_file = None
        self.features_file = None
        self.start_time = None
        self.final_elapsed_seconds = 0.0
        self.sample_count = 0
        self.latest_force = 0.0
        self.latest_bpm = None

        # Non-overlapping 3-minute window tracking
        self.completed_windows = 0
        self.window_start_elapsed = 0.0
        self.current_window_elapsed = 0.0

        # Holds up to 2000 samples for smooth live rolling 20s chart
        self.buffer_lock = threading.Lock()
        self.data_buffer = deque(maxlen=2000)

        # Official 5 Target Features (updated once per completed 180s window)
        self.latest_features = {}
        self.feature_history = deque(maxlen=100)

        self._stop_event = threading.Event()
        self._worker_thread = None

    def start(self, sample_rate_hz: int = 10) -> bool:
        """Start acquisition in a background thread."""
        with self._lock:
            if self.is_running or self.is_connecting:
                return False

            self.sample_rate_hz = sample_rate_hz
            self.sample_period_ms = max(10, 1000 // sample_rate_hz)
            self.is_connecting = True
            self.error_message = None
            self.sample_count = 0
            self.latest_force = 0.0
            self.latest_bpm = None
            self.final_elapsed_seconds = 0.0

            self.completed_windows = 0
            self.window_start_elapsed = 0.0
            self.current_window_elapsed = 0.0
            self.latest_features = {}

            with self.buffer_lock:
                self.data_buffer.clear()
                self.feature_history.clear()

            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._run_acquisition, daemon=True, name="VernierAcquisitionWorker"
            )
            self._worker_thread.start()
            return True

    def stop(self):
        """Signal acquisition to stop and disconnect."""
        if self.start_time and self.is_running:
            self.final_elapsed_seconds = time.time() - self.start_time
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        self.is_running = False
        self.is_connecting = False
        self.connected = False

    def _run_acquisition(self):
        """Worker thread entry point."""
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        sensor = None
        csv_file = None
        feat_csv_file = None
        session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = DATA_DIR / f"respiration_{session_time}_{self.sample_rate_hz}Hz.csv"
        self.features_file = DATA_DIR / f"respiration_features_{session_time}_{self.sample_rate_hz}Hz.csv"

        # Buffers strictly for the active 180-second non-overlapping block
        win_times = []
        win_forces = []

        try:
            sensor = gdx.gdx()
            sensor.open(connection="ble")

            if not sensor.devices:
                self.error_message = (
                    "No Vernier Go Direct device found. Make sure the belt is powered on (LED flashing red) "
                    "and computer Bluetooth is ON."
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

            headers = sensor.enabled_sensor_info()
            if not headers:
                headers = ["Force (N)", "Respiration Rate (bpm)"]

            # Initialize raw readings CSV file
            csv_file = open(self.output_file, mode="w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["timestamp_utc", "elapsed_seconds"] + headers)
            csv_file.flush()

            # Initialize 3-minute features CSV file
            feat_csv_file = open(self.features_file, mode="w", newline="", encoding="utf-8")
            feat_writer = csv.writer(feat_csv_file)
            feat_writer.writerow([
                "timestamp_utc",
                "window_index",
                "window_start_seconds",
                "window_end_seconds",
                "window_duration_seconds",
                "rsp_mean_amplitude",
                "rsp_std_amplitude",
                "rsp_spectral_entropy",
                "rsp_mean_peak_valley_magnitude",
                "rsp_std_peak_valley_magnitude",
                "calculated_bpm",
                "n_breaths"
            ])
            feat_csv_file.flush()

            # Start sensor streaming
            sensor.start(period=self.sample_period_ms)
            self.start_time = time.time()
            self.window_start_elapsed = 0.0
            sleep_interval = max(0.005, (self.sample_period_ms / 2000.0))

            while not self._stop_event.is_set():
                readings = sensor.read()

                if readings is not None and len(readings) > 0:
                    self.sample_count += 1
                    now_utc = datetime.now(timezone.utc).isoformat()
                    cur_time = time.time()
                    elapsed = round(cur_time - self.start_time, 3)
                    self.final_elapsed_seconds = elapsed

                    force_val = float(readings[0]) if readings[0] is not None else 0.0
                    bpm_val = None
                    if len(readings) > 1 and readings[1] is not None:
                        try:
                            v = float(readings[1])
                            if not (v != v):
                                bpm_val = v
                        except (ValueError, TypeError):
                            pass

                    self.latest_force = force_val
                    if bpm_val is not None:
                        self.latest_bpm = bpm_val

                    # Write raw row to CSV
                    csv_writer.writerow([now_utc, elapsed] + readings)
                    if self.sample_count % 10 == 0:
                        csv_file.flush()

                    # Push to live chart rolling buffer
                    with self.buffer_lock:
                        self.data_buffer.append(
                            {
                                "elapsed": elapsed,
                                "force": force_val,
                                "bpm": bpm_val,
                            }
                        )

                    # Accumulate into active non-overlapping 180s window
                    win_times.append(elapsed)
                    win_forces.append(force_val)
                    self.current_window_elapsed = elapsed - self.window_start_elapsed

                    # --- STRICT 3-MINUTE (180s) FEATURE EXTRACTION ---
                    # Trigger ONLY when a complete 180-second block finishes
                    if self.current_window_elapsed >= WINDOW_DURATION_SECONDS:
                        win_idx = self.completed_windows + 1
                        win_start_s = round(self.window_start_elapsed, 2)
                        win_end_s = round(elapsed, 2)

                        feat_res = extract_live_respiration_features(
                            win_times,
                            win_forces,
                            fs=float(self.sample_rate_hz),
                            min_window_s=170.0,
                        )

                        if feat_res["status"] == "OK":
                            feat_res["window_index"] = win_idx
                            feat_res["window_start_seconds"] = win_start_s
                            feat_res["window_end_seconds"] = win_end_s

                            self.completed_windows = win_idx
                            self.latest_features = feat_res

                            row_record = {
                                "timestamp_utc": now_utc,
                                "window_index": win_idx,
                                "window_start_seconds": win_start_s,
                                "window_end_seconds": win_end_s,
                                "window_duration_seconds": feat_res["duration_seconds"],
                                "rsp_mean_amplitude": feat_res["rsp_mean_amplitude"],
                                "rsp_std_amplitude": feat_res["rsp_std_amplitude"],
                                "rsp_spectral_entropy": feat_res["rsp_spectral_entropy"],
                                "rsp_mean_peak_valley_magnitude": feat_res["rsp_mean_peak_valley_magnitude"],
                                "rsp_std_peak_valley_magnitude": feat_res["rsp_std_peak_valley_magnitude"],
                                "calculated_bpm": feat_res["calculated_bpm"],
                                "n_breaths": feat_res["n_breaths"],
                            }

                            with self.buffer_lock:
                                self.feature_history.append(row_record)

                            feat_writer.writerow([
                                now_utc,
                                win_idx,
                                win_start_s,
                                win_end_s,
                                feat_res["duration_seconds"],
                                feat_res["rsp_mean_amplitude"],
                                feat_res["rsp_std_amplitude"],
                                feat_res["rsp_spectral_entropy"],
                                feat_res["rsp_mean_peak_valley_magnitude"],
                                feat_res["rsp_std_peak_valley_magnitude"],
                                feat_res["calculated_bpm"],
                                feat_res["n_breaths"],
                            ])
                            feat_csv_file.flush()

                        # Reset block accumulators for the next non-overlapping 3-minute window
                        win_times = []
                        win_forces = []
                        self.window_start_elapsed = elapsed
                        self.current_window_elapsed = 0.0

                time.sleep(sleep_interval)

        except Exception as e:
            self.error_message = f"Acquisition error: {str(e)}"
        finally:
            if csv_file and not csv_file.closed:
                csv_file.flush()
                csv_file.close()

            if feat_csv_file and not feat_csv_file.closed:
                feat_csv_file.flush()
                feat_csv_file.close()

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

    def get_snapshot(self):
        """Retrieve thread-safe snapshot of current data, features, and status."""
        with self.buffer_lock:
            buf_copy = list(self.data_buffer)
            feat_hist_copy = list(self.feature_history)

        if self.start_time and self.is_running:
            elapsed_total = time.time() - self.start_time
        else:
            elapsed_total = self.final_elapsed_seconds

        return {
            "is_running": self.is_running,
            "is_connecting": self.is_connecting,
            "connected": self.connected,
            "error_message": self.error_message,
            "device_name": self.device_name,
            "battery_level": self.battery_level,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": self.sample_count,
            "latest_force": self.latest_force,
            "latest_bpm": self.latest_bpm,
            "elapsed_seconds": elapsed_total,
            "completed_windows": self.completed_windows,
            "current_window_elapsed": self.current_window_elapsed,
            "output_file": str(self.output_file) if self.output_file else None,
            "features_file": str(self.features_file) if self.features_file else None,
            "latest_features": self.latest_features,
            "feature_history": feat_hist_copy,
            "data": buf_copy,
        }
