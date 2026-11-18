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
import json
import os
import socket
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

# --- Unity VR link (TCP, same PC by default) ---------------------------------
# Python is the TCP SERVER (listens); Unity is the CLIENT (connects).
# One JSON object per line ("\n"-terminated). Override with env vars
# UNITY_HOST / UNITY_PORT; set UNITY_ENABLED=0 to disable.
UNITY_ADDR = (os.environ.get("UNITY_HOST", "127.0.0.1"), int(os.environ.get("UNITY_PORT", "5005")))
UNITY_ENABLED = os.environ.get("UNITY_ENABLED", "1") != "0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _UnityTcpServer:
    """Listens for Unity and pushes newline-delimited JSON to every connected client.

    * keeps retrying to bind if the port is busy (status "port_busy"), instead of giving up
    * replays the last reading to a client that connects late
    * sends a heartbeat every HEARTBEAT_S so Unity can tell the link is alive
    """

    HEARTBEAT_S = 5.0
    RETRY_S = 2.0

    def __init__(self, addr):
        self.addr = addr
        self.status = "starting"            # starting | listening | port_busy
        self.last_error: "str | None" = None
        self._clients: list = []
        self._last_payload: "dict | None" = None
        self._last_sent_mono: "float | None" = None
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()  # one writer at a time per process (readings, heartbeats, acks)
        # Unity -> Python commands (baseline_start, ...). Set by _get_unity_server(): callback(msg: dict, reply: callable)
        self.on_command = None
        self._srv = None
        threading.Thread(target=self._serve_loop, daemon=True, name="UnityTcpServe").start()
        threading.Thread(target=self._heartbeat_loop, daemon=True, name="UnityTcpBeat").start()

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def _bind(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Windows: SO_REUSEADDR would let a second program share the port; use exclusive use.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            s.bind(self.addr)
            s.listen(1)
        except OSError:
            s.close()
            raise
        return s

    def _serve_loop(self) -> None:
        while True:
            try:
                self._srv = self._bind()
            except OSError as exc:
                self.status = "port_busy"
                self.last_error = str(exc)
                time.sleep(self.RETRY_S)
                continue
            self.status = "listening"
            self.last_error = None
            while True:
                try:
                    conn, _ = self._srv.accept()
                except OSError:
                    break
                self._on_connect(conn)
            self.status = "starting"
            time.sleep(self.RETRY_S)

    def _on_connect(self, conn) -> None:
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        with self._lock:
            self._clients.append(conn)
            last = self._last_payload
            age = (time.monotonic() - self._last_sent_mono) if self._last_sent_mono else None
        threading.Thread(target=self._client_reader, args=(conn,), daemon=True, name="UnityTcpRead").start()
        # immediate proof of life, so Unity counts the link as alive right away (not after the next 5 s heartbeat)
        self._send_to([conn], {"type": "heartbeat", "sent_utc": _utc_now(),
                               "last_reading_age_s": round(age, 1) if age is not None else None})
        if last is not None:                       # late joiner gets the latest reading at once
            msg = dict(last)
            msg["replayed"] = True
            msg["age_s"] = round(age, 1) if age is not None else None
            self._send_to([conn], msg)

    def _client_reader(self, conn) -> None:
        """Reads newline-delimited JSON commands from Unity (e.g. baseline_start) and hands them to on_command."""
        try:
            for raw in conn.makefile("r", encoding="utf-8", newline="\n"):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                cb = self.on_command
                if cb is None or not isinstance(msg, dict):
                    continue
                try:
                    cb(msg, lambda obj, c=conn: self._send_to([c], obj))
                except Exception as exc:  # never let a bad command kill the reader
                    print(f"[Unity TCP] command {msg.get('type')!r} failed: {exc}")
        except (OSError, ValueError):
            pass
        finally:
            with self._lock:
                self._clients = [c for c in self._clients if c is not conn]
            try:
                conn.close()
            except OSError:
                pass

    def _send_to(self, conns, obj: dict) -> None:
        data = (json.dumps(obj) + "\n").encode("utf-8")
        dead = []
        for c in conns:
            try:
                with self._send_lock:
                    c.sendall(data)
            except OSError:
                dead.append(c)
        if dead:
            with self._lock:
                self._clients = [c for c in self._clients if c not in dead]
            for c in dead:
                try:
                    c.close()
                except OSError:
                    pass

    def send(self, obj: dict) -> None:
        """Broadcast a reading to all clients and remember it for late joiners."""
        with self._lock:
            self._last_payload = dict(obj)
            self._last_sent_mono = time.monotonic()
            conns = list(self._clients)
        self._send_to(conns, obj)

    def _heartbeat_loop(self) -> None:
        while True:
            time.sleep(self.HEARTBEAT_S)
            with self._lock:
                conns = list(self._clients)
                age = (time.monotonic() - self._last_sent_mono) if self._last_sent_mono else None
            if conns:
                self._send_to(conns, {
                    "type": "heartbeat",
                    "sent_utc": _utc_now(),
                    "last_reading_age_s": round(age, 1) if age is not None else None,
                })


_unity_server: "_UnityTcpServer | None" = None
_unity_server_lock = threading.Lock()


def _get_unity_server():
    """Start the TCP server once (it retries the port in the background). None if disabled."""
    global _unity_server
    if not UNITY_ENABLED:
        return None
    with _unity_server_lock:
        if _unity_server is None:
            _unity_server = _UnityTcpServer(UNITY_ADDR)
            _unity_server.on_command = _handle_unity_command
        return _unity_server


def get_unity_link_status() -> dict:
    """For the UI: is the Unity link listening, and is a client connected?"""
    server = _get_unity_server()
    if server is None:
        return {"enabled": False, "status": "disabled", "clients": 0, "address": None, "error": None}
    return {
        "enabled": True,
        "status": server.status,
        "clients": server.client_count,
        "address": f"{server.addr[0]}:{server.addr[1]}",
        "error": server.last_error,
    }


def send_to_unity(pred_res: dict) -> None:
    """Send one window's result to Unity as a JSON line over TCP. Never raises.

    status: Valid | Baseline | LowQuality. Only "Valid" carries a usable High/Low state;
    otherwise state is "Unknown" so the adaptive logic ignores the reading.
    """
    server = _get_unity_server()
    if server is None:
        return
    try:
        status, state, reason = "Valid", pred_res["vr_state"], None   # "Low" | "High"
        if pred_res.get("is_baseline_window"):
            status, state, reason = "Baseline", "Unknown", "Window used to set the personal baseline"
        elif int(pred_res.get("n_breaths", 0)) < 3:
            status, state, reason = "LowQuality", "Unknown", "Fewer than 3 breaths in window"
        elif pred_res.get("baseline_warning"):
            status, state, reason = "LowQuality", "Unknown", str(pred_res["baseline_warning"])
        payload = {
            "type": "reading",
            "window_index": pred_res["window_index"],
            "status": status,
            "state": state,
            "confidence": round(float(pred_res["confidence"]), 3) if status == "Valid" else 0.0,
            "stress_prob": round(float(pred_res["stress_probability"]), 3),
            "bpm": float(pred_res["calculated_bpm"]),
            "is_baseline_window": bool(pred_res.get("is_baseline_window", False)),
            "reason": reason,
            "window_end_seconds": pred_res.get("window_end_seconds"),
            "sent_utc": _utc_now(),
            "replayed": False,
            "model": "rsp_only_rf_5feat_v1",
        }
        server.send(payload)
    except Exception:
        pass


# --- Unity -> Python commands ---------------------------------------------------
# The headset tells Python when the experiment's resting baseline begins, so nobody has to click Start on page 7.
#   baseline_start : belt not running -> connect + start recording (window #1 = the personal baseline)
#                    belt already running (e.g. block 2) -> restart the 3-minute window now, and (if
#                    REBASELINE_EACH_BLOCK) forget the old baseline so this rest becomes the new baseline
#   baseline_end   : informational only
#   session_end    : stop recording and close the files
AUTO_SAMPLE_RATE_HZ = int(os.environ.get("AUTO_SAMPLE_RATE_HZ", "20"))
AUTO_WINDOW_S = float(os.environ.get("AUTO_WINDOW_S", "60"))                  # = headset 1-min baseline (SessionController.baselineSeconds)
AUTO_ROLLING_WINDOW_S = float(os.environ.get("AUTO_ROLLING_WINDOW_S", "60"))   # 60s rolling analysis lookback after baseline
AUTO_PRED_STEP_S = float(os.environ.get("AUTO_PRED_STEP_S", "30"))             # emit a fresh ML prediction to Unity every 30s
REBASELINE_EACH_BLOCK = os.environ.get("REBASELINE_EACH_BLOCK", "1") != "0"


def _handle_unity_command(msg: dict, reply) -> None:
    kind = str(msg.get("type", ""))
    mgr = LiveRSPMLManager.get_instance()
    late_s = float(msg.get("elapsed_since_event_s", 0.0) or 0.0)
    block = msg.get("block")
    print(f"[Unity cmd] {kind} block={block} participant={msg.get('participant')} late={late_s:.1f}s")

    if kind == "baseline_start":
        planned = float(msg.get("planned_s", AUTO_WINDOW_S) or AUTO_WINDOW_S)
        t_cmd = time.time() - late_s
        deadline = t_cmd + planned          # when the headset's baseline ends
        if mgr.is_running or mgr.is_connecting:
            mgr.request_window_restart(clear_baseline=REBASELINE_EACH_BLOCK)
            mgr.baseline_deadline = deadline
            action = "window_realigned"
        else:
            action = "recording_starting" if mgr.start(
                sample_rate_hz=AUTO_SAMPLE_RATE_HZ,
                window_duration_s=AUTO_WINDOW_S,
                rolling_window_s=AUTO_ROLLING_WINDOW_S,
                prediction_step_s=AUTO_PRED_STEP_S,
                keep_baseline=False,
            ) else "start_refused"
            if action == "recording_starting":
                mgr.baseline_deadline = deadline   # set after start() (start() clears it)
        note = ""
        if abs(planned - AUTO_WINDOW_S) > 1.0:
            note = f"Unity baseline is {planned:.0f} s but the ML window is {AUTO_WINDOW_S:.0f} s - they should match"
        reply({"type": "ack", "command": kind, "action": action, "late_s": round(late_s, 1), "note": note})

        if action == "recording_starting":
            # second ack once the belt is really streaming (BLE connect can take several seconds)
            def _await_stream():
                deadline = time.time() + 40.0
                while time.time() < deadline and not mgr.is_running and mgr.error_message is None:
                    time.sleep(0.2)
                if mgr.is_running:
                    reply({"type": "ack", "command": kind, "action": "recording_started",
                           "connect_delay_s": round(time.time() - t_cmd, 1), "late_s": round(late_s, 1), "note": ""})
                else:
                    reply({"type": "ack", "command": kind, "action": "recording_failed", "late_s": round(late_s, 1),
                           "note": str(mgr.error_message or "belt did not start within 40 s")})
            threading.Thread(target=_await_stream, daemon=True, name="UnityCmdAck").start()
    elif kind == "session_end":
        mgr.stop()
        reply({"type": "ack", "command": kind, "action": "recording_stopped", "late_s": round(late_s, 1), "note": ""})
    elif kind in ("baseline_end", "ping"):
        reply({"type": "ack", "command": kind, "action": "ok", "late_s": round(late_s, 1), "note": ""})


def send_status_to_unity(status: str, reason: str, window_index: int = 0) -> None:
    """Tell Unity the belt/pipeline is down (status e.g. "Disconnected"); state is Unknown."""
    server = _get_unity_server()
    if server is None:
        return
    try:
        server.send({
            "type": "reading",
            "window_index": window_index,
            "status": status,
            "state": "Unknown",
            "confidence": 0.0,
            "stress_prob": 0.0,
            "bpm": 0.0,
            "is_baseline_window": False,
            "reason": reason,
            "window_end_seconds": None,
            "sent_utc": _utc_now(),
            "replayed": False,
            "model": "rsp_only_rf_5feat_v1",
        })
    except Exception:
        pass


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
        _get_unity_server()   # start listening so Unity can connect before recording starts

        self.is_running = False
        self.is_connecting = False
        self.connected = False
        self.error_message: str | None = None

        self.device_name = "Not Connected"
        self.battery_level = None
        self.sample_rate_hz = 20
        self.sample_period_ms = 50
        self.baseline_duration_s = AUTO_WINDOW_S
        self.rolling_window_s = AUTO_ROLLING_WINDOW_S
        self.prediction_step_s = AUTO_PRED_STEP_S
        self.window_duration_s = AUTO_WINDOW_S

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
        # wall-clock time (epoch s) at which the HEADSET baseline ends; window #1 is closed then, so the ML baseline
        # covers the same rest period as the headset's, not the time after the belt finished connecting.
        self.baseline_deadline: "float | None" = None
        self._window_restart_req: "bool | None" = None   # None = no request; else = also clear the baseline?

    def request_window_restart(self, clear_baseline: bool = True) -> None:
        """Ask the recording loop to start a fresh 3-minute window at the next sample (belt stays connected)."""
        self._window_restart_req = bool(clear_baseline)

    def load_baseline_csv(self, csv_path: Path, window_seconds: float = 180.0) -> dict:
        """Load and lock a baseline from an existing Vernier raw CSV before starting live inference."""
        return self.predictor.load_baseline_from_raw_csv(csv_path, window_seconds=window_seconds)

    def clear_baseline(self) -> None:
        self.predictor.clear_baseline()

    def start(
        self,
        sample_rate_hz: int = 20,
        window_duration_s: float = 120.0,
        keep_baseline: bool = True,
        rolling_window_s: float | None = None,
        prediction_step_s: float | None = None,
    ) -> bool:
        """Start live BLE acquisition + feature extraction + ML inference thread.

        If rolling_window_s and prediction_step_s are provided (e.g. 60s rolling / 30s step),
        Window #1 collects a full `window_duration_s` baseline (e.g. 120s), then subsequent
        predictions run every `prediction_step_s` over the most recent `rolling_window_s` of
        post-baseline active data.
        """
        with self._lock:
            if self.is_running or self.is_connecting:
                return False

            self.sample_rate_hz = sample_rate_hz
            self.sample_period_ms = max(10, 1000 // sample_rate_hz)
            self.baseline_duration_s = float(window_duration_s)
            if rolling_window_s is None and prediction_step_s is None:
                self.rolling_window_s = self.baseline_duration_s
                self.prediction_step_s = self.baseline_duration_s
            else:
                self.rolling_window_s = float(rolling_window_s if rolling_window_s is not None else self.baseline_duration_s)
                self.prediction_step_s = float(prediction_step_s if prediction_step_s is not None else self.rolling_window_s)
            self.window_duration_s = self.baseline_duration_s
            self.is_connecting = True
            self.baseline_deadline = None
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
        self.baseline_deadline = None
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
        if self.rolling_window_s == self.baseline_duration_s and self.prediction_step_s == self.baseline_duration_s:
            win_tag = f"{int(self.baseline_duration_s)}s"
        else:
            win_tag = f"{int(self.baseline_duration_s)}b_{int(self.rolling_window_s)}w_{int(self.prediction_step_s)}s"
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
                send_status_to_unity("Disconnected", self.error_message, self.completed_windows)
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
            last_valid_t = self.start_time
            self.window_start_elapsed = 0.0
            sleep_interval = max(0.005, self.sample_period_ms / 2000.0)
            min_valid_s = max(15.0, self.window_duration_s - 5.0)

            while not self._stop_event.is_set():
                readings = sensor.read()

                # Belt silent (no packets, or only NaN force) for 10 s -> tell Unity and stop.
                if time.time() - last_valid_t > 10.0:
                    self.error_message = (
                        "Belt is connected but sending NO force data. Turn the GDX-RB off and on "
                        "(LED flashing red), close other apps using it, then Start again."
                    )
                    send_status_to_unity("Disconnected", self.error_message, self.completed_windows)
                    break

                if readings is not None and len(readings) > 0:
                    self.sample_count += 1
                    now_utc = datetime.now(timezone.utc).isoformat()
                    elapsed = round(time.time() - self.start_time, 3)
                    self.final_elapsed_seconds = elapsed

                    force_val = float(readings[0]) if readings[0] is not None else float("nan")
                    # Belt connected but sending no force values -> don't log NaN rows; fail loudly.
                    if force_val != force_val:
                        self.sample_count -= 1
                        time.sleep(sleep_interval)
                        continue
                    last_valid_t = time.time()
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

                    # Unity says a new resting baseline just began: start window #N fresh from this sample
                    if self._window_restart_req is not None:
                        clear_bl = self._window_restart_req
                        self._window_restart_req = None
                        win_times = []
                        win_forces = []
                        self.window_start_elapsed = elapsed
                        self.current_window_elapsed = 0.0
                        if clear_bl:
                            self.predictor.clear_baseline()
                        print(f"[Live ML] window restarted at {elapsed:.1f}s (baseline cleared: {clear_bl})")

                    win_times.append(elapsed)
                    win_forces.append(force_val)
                    self.current_window_elapsed = elapsed - self.window_start_elapsed

                    # Determine active timer target: full baseline_duration_s (e.g. 120s) before
                    # baseline is locked, then prediction_step_s (e.g. 30s) during live play.
                    collecting_baseline = not self.predictor.has_baseline
                    target_step_s = self.baseline_duration_s if collecting_baseline else self.prediction_step_s
                    if collecting_baseline and self.baseline_deadline is not None:
                        # close window #1 exactly when the headset's baseline ends (belt connect time is already used up)
                        wall_window_start = self.start_time + self.window_start_elapsed
                        target_step_s = max(20.0, min(self.baseline_duration_s, self.baseline_deadline - wall_window_start))
                    self.window_duration_s = target_step_s

                    # In rolling post-baseline mode, keep only the most recent rolling_window_s (e.g. 60s)
                    if (not collecting_baseline) and (self.rolling_window_s > self.prediction_step_s):
                        cutoff_t = elapsed - self.rolling_window_s
                        while len(win_times) > 1 and win_times[0] < cutoff_t:
                            win_times.pop(0)
                            win_forces.pop(0)

                    # Trigger Feature Extraction + ML Prediction when baseline or step interval completes
                    if self.current_window_elapsed >= target_step_s:
                        win_idx = self.completed_windows + 1
                        win_start_s = round(win_times[0], 2) if win_times else round(self.window_start_elapsed, 2)
                        win_end_s = round(elapsed, 2)
                        min_req_s = max(15.0, min(target_step_s, self.rolling_window_s) - 5.0)

                        feat_res = extract_live_respiration_features(
                            win_times,
                            win_forces,
                            fs=float(self.sample_rate_hz),
                            min_window_s=min_req_s,
                        )

                        is_baseline_win = False
                        if feat_res["status"] == "OK":
                            feat_res["window_index"] = win_idx
                            feat_res["window_start_seconds"] = win_start_s
                            feat_res["window_end_seconds"] = win_end_s

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

                            send_to_unity(pred_res)

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

                        # Advance window:
                        # - Always clear after baseline (never mix resting baseline into active play)
                        # - Also clear in non-overlapping fixed mode (rolling_window_s <= prediction_step_s)
                        # - In rolling mode (e.g. 60s window / 30s step), keep buffer & reset only the 30s step timer
                        if is_baseline_win or (self.rolling_window_s <= self.prediction_step_s):
                            win_times = []
                            win_forces = []
                        self.window_start_elapsed = elapsed
                        self.current_window_elapsed = 0.0
                        self.window_duration_s = (
                            self.baseline_duration_s if not self.predictor.has_baseline else self.prediction_step_s
                        )

                time.sleep(sleep_interval)

        except Exception as e:
            self.error_message = f"Live ML pipeline error: {e}"
            send_status_to_unity("Disconnected", self.error_message, self.completed_windows)
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

        active_target_s = (
            self.window_duration_s if self.is_running else
            (self.baseline_duration_s if not self.predictor.has_baseline else self.prediction_step_s)
        )

        return {
            "is_running": self.is_running,
            "is_connecting": self.is_connecting,
            "connected": self.connected,
            "error_message": self.error_message,
            "device_name": self.device_name,
            "battery_level": self.battery_level,
            "sample_rate_hz": self.sample_rate_hz,
            "window_duration_s": active_target_s,
            "baseline_duration_s": self.baseline_duration_s,
            "rolling_window_s": self.rolling_window_s,
            "prediction_step_s": self.prediction_step_s,
            "sample_count": self.sample_count,
            "latest_force": self.latest_force,
            "latest_bpm": self.latest_bpm,
            "elapsed_seconds": elapsed_total,
            "completed_windows": self.completed_windows,
            "current_window_elapsed": self.current_window_elapsed,
            "unity_link": get_unity_link_status(),
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
