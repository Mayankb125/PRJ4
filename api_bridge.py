"""Safe, Non-Intrusive FastAPI Bridge for PRJ4.

Imports LiveRSPMLManager without modifying any core logic or models.
Provides lightweight JSON endpoints for the React frontend workstation,
including Sources selection (Quest standalone vs PC editor) and explicit
data storage path inspection.
"""
from __future__ import annotations

import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent
RSP_MODEL_DIR = ROOT_DIR / "RSP_Only_Model"
LIVE_DIR = ROOT_DIR / "PhysioSync_Live"
SESSIONS_DIR = RSP_MODEL_DIR / "live_sessions"

for p in (str(ROOT_DIR), str(RSP_MODEL_DIR), str(LIVE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
from live_rsp_ml_manager import (
    LiveRSPMLManager,
    get_unity_link_status,
    AUTO_SAMPLE_RATE_HZ,
    AUTO_WINDOW_S,
    AUTO_ROLLING_WINDOW_S,
    AUTO_PRED_STEP_S,
)

LIVE_DATA_DIR = LIVE_DIR / "data"
DEFAULT_UNITY_ROOT = Path.home() / "AppData" / "LocalLow" / "DefaultCompany" / "MazeStudy" / "MazeStudy"
QUEST_PULL_ROOT = Path(os.environ.get("QUEST_PULL_ROOT", r"D:\Vr_Project\MazeStudy\tools\quest_usb\pulled\MazeStudy"))

# 3-Row Rule Mapping matching Page 8 (High Stress, Low Stress, and Perf-Only Fallback)
ORIGINAL_RULES = {
    "HIGH_GOOD": "StepUp",
    "HIGH_POOR": "StepDown",
    "LOW_GOOD": "StepUpIfRepeated",
    "LOW_POOR": "StepDown",
    "PERF_GOOD": "StepUpIfRepeated",
    "PERF_POOR": "StepDown",
}
RULE_KEYS = {
    "HG": "HIGH_GOOD", "HP": "HIGH_POOR",
    "LG": "LOW_GOOD", "LP": "LOW_POOR",
    "PG": "PERF_GOOD", "PP": "PERF_POOR",
}

# In-memory Sources Configuration (matches Streamlit Page 8 sidebar)
UNITY_CONFIG = {
    "source_mode": "quest",  # "quest" (USB pull) or "pc" (PC editor / Quest Link)
    "custom_root": None,
    "follow_newest": True,
    "selected_session": None,
    "selected_prediction_file": "(current live session / newest)",
    "show_practice": False,
}

app = FastAPI(title="PRJ4 Bio-VR Workstation API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sanitize(val: Any) -> Any:
    """Recursively converts NaN, Infinity, and non-serializable objects to JSON-friendly values."""
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 4)
    if isinstance(val, dict):
        return {k: _sanitize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize(v) for v in val]
    if isinstance(val, Path):
        return str(val)
    return val


def _to_ist_str(ts_val) -> str:
    """Converts a UTC timestamp to an IST HH:MM:SS string."""
    try:
        t = pd.to_datetime(ts_val, utc=True)
        return t.tz_convert("Asia/Kolkata").strftime("%H:%M:%S")
    except Exception:
        return str(ts_val) if ts_val else ""


def list_unity_sessions(root: Path) -> list[Path]:
    """Finds all valid Unity session folders under root, sorted newest first."""
    if not root.is_dir():
        return []
    sessions = [p for p in root.iterdir() if p.is_dir() and (p / "trials.csv").is_file()]

    def _key(p: Path):
        m = re.search(r"_(\d{8}_\d{6})$", p.name)
        mt = max((p / n).stat().st_mtime for n in ("trials.csv", "events.csv") if (p / n).is_file())
        return (m.group(1) if m else "", mt)

    return sorted(sessions, key=_key, reverse=True)


def get_manager() -> LiveRSPMLManager:
    return LiveRSPMLManager.get_instance()


@app.on_event("startup")
def startup_event():
    # Calling get_instance initializes Unity TCP server (port 5005) in background
    LiveRSPMLManager.get_instance()


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/status")
def get_status():
    """Returns the live snapshot of belt telemetry, 5 Bian et al. features, and ML inference."""
    mgr = get_manager()
    snap = mgr.get_snapshot()

    raw_data = snap.get("data", [])
    recent_waveform = raw_data[-400:] if len(raw_data) > 400 else raw_data

    clean_snap = {
        "is_running": snap["is_running"],
        "is_connecting": snap["is_connecting"],
        "connected": snap["connected"],
        "error_message": snap["error_message"],
        "device_name": snap["device_name"],
        "battery_level": snap["battery_level"],
        "sample_rate_hz": snap["sample_rate_hz"],
        "window_duration_s": snap["window_duration_s"],
        "baseline_duration_s": snap["baseline_duration_s"],
        "rolling_window_s": snap["rolling_window_s"],
        "prediction_step_s": snap["prediction_step_s"],
        "sample_count": snap["sample_count"],
        "latest_force": snap["latest_force"],
        "latest_bpm": snap["latest_bpm"],
        "elapsed_seconds": snap["elapsed_seconds"],
        "completed_windows": snap["completed_windows"],
        "current_window_elapsed": snap["current_window_elapsed"],
        "unity_link": snap["unity_link"],
        "has_baseline": snap["has_baseline"],
        "baseline_bpm": snap["baseline_bpm"],
        "baseline_source": snap["baseline_source"],
        "baseline_warning": snap["baseline_warning"],
        "baseline_features": snap["baseline_features"],
        "latest_features": snap["latest_features"],
        "latest_prediction": snap["latest_prediction"],
        "prediction_history": snap["prediction_history"][-30:],
        "recent_waveform": recent_waveform,
        "raw_file": snap.get("raw_file"),
        "predictions_file": snap.get("predictions_file"),
        "sessions_dir": str(SESSIONS_DIR),
        "live_data_dir": str(LIVE_DATA_DIR),
    }
    return _sanitize(clean_snap)


class StartRequest(BaseModel):
    device_name: str = "GDX-RB 0K700080"
    sample_rate_hz: int = AUTO_SAMPLE_RATE_HZ
    baseline_duration_s: float = AUTO_WINDOW_S
    rolling_window_s: float = AUTO_ROLLING_WINDOW_S
    prediction_step_s: float = AUTO_PRED_STEP_S
    keep_baseline: bool = False


@app.post("/api/control/start")
def start_recording(req: StartRequest):
    mgr = get_manager()
    success = mgr.start(
        sample_rate_hz=req.sample_rate_hz,
        window_duration_s=req.baseline_duration_s,
        rolling_window_s=req.rolling_window_s,
        prediction_step_s=req.prediction_step_s,
        keep_baseline=req.keep_baseline,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Cannot start: acquisition already active or connecting.")
    return {"status": "started", "config": req.dict()}


@app.get("/api/baseline/candidates")
def list_baseline_candidates():
    candidates = []
    if LIVE_DATA_DIR.is_dir():
        for f in sorted(LIVE_DATA_DIR.glob("respiration_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
            candidates.append({
                "filename": f.name,
                "path": str(f),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
    return {"candidates": candidates}


class LoadBaselineRequest(BaseModel):
    filename: str
    window_seconds: float = 60.0


@app.post("/api/baseline/load_csv")
def load_baseline_csv(req: LoadBaselineRequest):
    mgr = get_manager()
    target_path = LIVE_DATA_DIR / req.filename
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"File {req.filename} not found.")
    try:
        res = mgr.load_baseline_csv(target_path, window_seconds=req.window_seconds)
        return {"status": "baseline_loaded", "features": _sanitize(res)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load baseline: {exc}")


@app.post("/api/baseline/clear")
def clear_baseline():
    mgr = get_manager()
    mgr.clear_baseline()
    return {"status": "baseline_cleared"}


@app.post("/api/control/stop")
def stop_recording():
    mgr = get_manager()
    mgr.stop()
    return {"status": "stopped"}


@app.post("/api/control/rebaseline")
def trigger_rebaseline():
    mgr = get_manager()
    mgr.request_window_restart(clear_baseline=True)
    return {"status": "rebaseline_requested"}


# --- Sources & Data Storage Configuration ---

class SourcesUpdateRequest(BaseModel):
    source_mode: str | None = None
    custom_root: str | None = None
    follow_newest: bool | None = None
    selected_session: str | None = None
    selected_prediction_file: str | None = None
    show_practice: bool | None = None


@app.get("/api/sources")
def get_sources_info():
    """Returns current sources settings, available Unity sessions, and exact storage paths on disk."""
    mode = UNITY_CONFIG["source_mode"]
    default_root = DEFAULT_UNITY_ROOT if mode == "pc" else QUEST_PULL_ROOT
    root_path = Path(UNITY_CONFIG["custom_root"] or default_root)

    sessions = list_unity_sessions(root_path)
    session_names = [p.name for p in sessions]

    pred_files = []
    if SESSIONS_DIR.is_dir():
        pred_files = [
            p.name for p in sorted(SESSIONS_DIR.glob("predictions_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        ]

    mgr = get_manager()
    snap = mgr.get_snapshot()

    active_session = None
    if UNITY_CONFIG["follow_newest"] and session_names:
        active_session = session_names[0]
    elif UNITY_CONFIG["selected_session"] in session_names:
        active_session = UNITY_CONFIG["selected_session"]
    elif session_names:
        active_session = session_names[0]

    return _sanitize({
        "source_mode": mode,
        "unity_root": str(root_path),
        "default_quest_root": str(QUEST_PULL_ROOT),
        "default_pc_root": str(DEFAULT_UNITY_ROOT),
        "follow_newest": UNITY_CONFIG["follow_newest"],
        "selected_session": active_session,
        "available_sessions": session_names,
        "selected_prediction_file": UNITY_CONFIG["selected_prediction_file"],
        "available_prediction_files": ["(current live session / newest)"] + pred_files,
        "show_practice": UNITY_CONFIG["show_practice"],
        "storage_paths": {
            "live_sessions_dir": str(SESSIONS_DIR),
            "baseline_candidates_dir": str(LIVE_DATA_DIR),
            "active_raw_file": snap.get("raw_file"),
            "active_predictions_file": snap.get("predictions_file"),
            "active_unity_session_dir": str(root_path / active_session) if active_session else None,
        },
    })


@app.post("/api/sources")
def update_sources(req: SourcesUpdateRequest):
    """Updates sources selection (Quest USB vs PC LocalLow, session choice, prediction file)."""
    if req.source_mode is not None:
        UNITY_CONFIG["source_mode"] = req.source_mode
    if req.custom_root is not None:
        UNITY_CONFIG["custom_root"] = req.custom_root.strip() or None
    if req.follow_newest is not None:
        UNITY_CONFIG["follow_newest"] = req.follow_newest
    if req.selected_session is not None:
        UNITY_CONFIG["selected_session"] = req.selected_session
    if req.selected_prediction_file is not None:
        UNITY_CONFIG["selected_prediction_file"] = req.selected_prediction_file
    if req.show_practice is not None:
        UNITY_CONFIG["show_practice"] = req.show_practice
    return {"status": "ok", "config": UNITY_CONFIG}


@app.get("/api/unity/latest")
def get_unity_latest():
    """Reads Unity session logs from the configured source folder (Quest USB or PC LocalLow)."""
    mode = UNITY_CONFIG["source_mode"]
    default_root = DEFAULT_UNITY_ROOT if mode == "pc" else QUEST_PULL_ROOT
    root_path = Path(UNITY_CONFIG["custom_root"] or default_root)

    sessions = list_unity_sessions(root_path)

    # Fallback search if current root has no sessions yet
    if not sessions:
        fallback_root = QUEST_PULL_ROOT if mode == "pc" else DEFAULT_UNITY_ROOT
        if fallback_root.is_dir():
            fallback_sessions = list_unity_sessions(fallback_root)
            if fallback_sessions:
                root_path = fallback_root
                sessions = fallback_sessions

    if not sessions:
        return _sanitize({
            "session_name": None,
            "session_path": None,
            "unity_root": str(root_path),
            "source_mode": mode,
            "trials": [],
            "recent_events": [],
            "latest_trial": None,
            "rules": ORIGINAL_RULES,
            "cell_counts": {},
            "active_cell": None,
            "hold_no_physio": False,
        })

    active_session = None
    if UNITY_CONFIG["follow_newest"]:
        active_session = sessions[0]
    else:
        for s in sessions:
            if s.name == UNITY_CONFIG["selected_session"]:
                active_session = s
                break
        if not active_session:
            active_session = sessions[0]

    # Parse rules in use from session.csv
    rules = dict(ORIGINAL_RULES)
    session_csv = active_session / "session.csv"
    if session_csv.is_file():
        try:
            sdf = pd.read_csv(session_csv, encoding="utf-8-sig", on_bad_lines="skip")
            if not sdf.empty and "key" in sdf.columns:
                rows_ = sdf[sdf["key"].astype(str).str.startswith("adaptive_block")]
                if not rows_.empty:
                    m = re.search(r"rules=([^;]+)", str(rows_.iloc[-1]["value"]))
                    if m:
                        for part in m.group(1).split(","):
                            k, _, v = part.partition(":")
                            if k in RULE_KEYS:
                                rules[RULE_KEYS[k]] = v
        except Exception:
            pass

    # Load trials
    trials_csv = active_session / "trials.csv"
    trials_list = []
    latest_trial = None
    cell_counts = {}
    hold_no_physio = False

    if trials_csv.is_file():
        try:
            tdf = pd.read_csv(trials_csv, encoding="utf-8-sig", on_bad_lines="skip")
            if not tdf.empty:
                # Check practice filter
                if not UNITY_CONFIG["show_practice"] and "practice" in tdf.columns:
                    tdf = tdf[tdf["practice"].astype(str) != "1"]

                for _, row in tdf.iterrows():
                    rc = str(row.get("rule_cell", "")).strip()
                    if rc:
                        cell_counts[rc] = cell_counts.get(rc, 0) + 1
                    
                    amode = str(row.get("adapt_mode", ""))
                    if amode == "hold_no_physio":
                        hold_no_physio = True

                    item = {
                        "global_trial": int(row.get("global_trial", 0)),
                        "block_trial": int(row.get("block_trial", 0)),
                        "prev_level": int(row.get("prev_level", 0)) if pd.notna(row.get("prev_level")) else 0,
                        "next_level": int(row.get("next_level", 0)) if pd.notna(row.get("next_level")) else 0,
                        "prev_guidance": int(row.get("prev_guidance", 0)) if pd.notna(row.get("prev_guidance")) else 0,
                        "next_guidance": int(row.get("next_guidance", 0)) if pd.notna(row.get("next_guidance")) else 0,
                        "change": str(row.get("change", "none")),
                        "perf_class": str(row.get("perf_class", "")),
                        "physio_state": str(row.get("physio_state", "")),
                        "physio_confidence": float(row.get("physio_confidence", 0.0)) if pd.notna(row.get("physio_confidence")) else 0.0,
                        "rule_cell": rc,
                        "reason": str(row.get("reason", "")),
                        "start_ist": _to_ist_str(row.get("start_utc")),
                        "end_ist": _to_ist_str(row.get("end_utc")),
                        "outcome": str(row.get("outcome", "")),
                    }
                    trials_list.append(item)
                if trials_list:
                    latest_trial = trials_list[-1]
        except Exception:
            pass

    # Load recent events
    events_csv = active_session / "events.csv"
    events_list = []
    if events_csv.is_file():
        try:
            edf = pd.read_csv(events_csv, encoding="utf-8-sig", on_bad_lines="skip")
            if not edf.empty:
                # filter out noisy move events
                edf = edf[edf["type"] != "move"].tail(35)
                for _, erow in edf.iterrows():
                    events_list.append({
                        "ist": _to_ist_str(erow.get("utc")),
                        "type": str(erow.get("type", "")),
                        "detail": str(erow.get("detail", "")),
                        "session_s": round(float(erow.get("session_s", 0)), 1) if pd.notna(erow.get("session_s")) else 0.0,
                    })
        except Exception:
            pass

    return _sanitize({
        "session_name": active_session.name,
        "session_path": str(active_session),
        "unity_root": str(root_path),
        "source_mode": mode,
        "trials": trials_list,
        "latest_trial": latest_trial,
        "recent_events": events_list[::-1],
        "rules": rules,
        "cell_counts": cell_counts,
        "active_cell": latest_trial.get("rule_cell") if latest_trial else None,
        "hold_no_physio": hold_no_physio,
    })


@app.get("/api/sessions")
def list_recorded_sessions():
    """Lists saved raw CSVs and ML prediction files."""
    files = []
    if SESSIONS_DIR.is_dir():
        for f in sorted(SESSIONS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
            files.append({
                "name": f.name,
                "path": str(f),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "modified_ist": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
    return {"files": files}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_bridge:app", host="127.0.0.1", port=8000, reload=False)
