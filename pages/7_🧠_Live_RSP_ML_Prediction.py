"""Page 7: Live End-to-End Respiration ML Stress Predictor (5-Feature RSP-Only Model)."""
from datetime import timedelta
from pathlib import Path
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add isolated RSP_Only_Model directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
RSP_MODEL_DIR = ROOT_DIR / "RSP_Only_Model"
LIVE_DATA_DIR = ROOT_DIR / "PhysioSync_Live" / "data"

if str(RSP_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(RSP_MODEL_DIR))

from live_rsp_ml_manager import LiveRSPMLManager
from live_rsp_predictor import RAW_FEATURE_NAMES

st.set_page_config(
    page_title="Live Respiration ML Stress Predictor",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Live End-to-End Respiration Stress Predictor (5-Feature Model)")
st.write(
    "Full real-time pipeline: **Vernier GDX-RB (BLE)** $\\rightarrow$ **5 Bian et al. Respiration Features** "
    "$\\rightarrow$ **Personal Baseline Subtraction ($\\Delta$)** $\\rightarrow$ "
    "**5-Feature Random Forest (81.06% LOSO Accuracy)** $\\rightarrow$ **VR-Ready State (`Low` / `High`)**."
)

st.divider()

manager = LiveRSPMLManager.get_instance()
status = manager.get_snapshot()

# ----------------------------------------------------
# 1. BASELINE CALIBRATION SETUP (OPTIONAL PRE-LOAD OR LIVE)
# ----------------------------------------------------
with st.expander("🎯 Step 1: Personal Baseline Calibration (Required for Normalized ML)", expanded=not status["has_baseline"]):
    if status["has_baseline"]:
        st.success(f"✅ **Baseline Locked:** `{status['baseline_source']}`")
        b_cols = st.columns(5)
        b_feats = status["baseline_features"]
        labels = ["Mean Amp (N)", "Std Amp (N)", "Entropy", "Depth Mean (N)", "Depth Std (N)"]
        for col_ui, key, lbl in zip(b_cols, RAW_FEATURE_NAMES, labels):
            col_ui.metric(lbl, f"{b_feats[key]:.3f}")
        if not status["is_running"]:
            if st.button("🔄 Clear Baseline (Record Fresh Window #1 as Baseline)"):
                manager.clear_baseline()
                st.rerun()
    else:
        st.info(
            "No baseline locked yet. **Option A:** Simply click **Start** below—Window #1 will automatically lock as your personal baseline. "
            "**Option B:** Pre-load a previously recorded 3-minute Vernier CSV below to start predicting immediately on Window #1!"
        )
        csv_candidates = sorted(
            [p for p in LIVE_DATA_DIR.glob("respiration_*.csv") if "features" not in p.name and p.stat().st_size > 10000],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if LIVE_DATA_DIR.exists() else []

        if csv_candidates:
            c1, c2 = st.columns([2.5, 1.2])
            with c1:
                chosen_csv = st.selectbox(
                    "Select Previously Recorded Raw Respiration CSV",
                    options=csv_candidates,
                    format_func=lambda p: f"{p.name} ({p.stat().st_size / 1024:.0f} KB)",
                    disabled=status["is_running"],
                )
            with c2:
                st.write("")
                st.write("")
                if st.button("📥 Lock CSV as Baseline", use_container_width=True, disabled=status["is_running"]):
                    try:
                        manager.load_baseline_csv(chosen_csv, window_seconds=120.0)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to load baseline: {exc}")

# ----------------------------------------------------
# 2. LIVE ACQUISITION & INFERENCE CONTROLS
# ----------------------------------------------------
c_freq, c_win, c_btn, c_stat = st.columns([1.0, 1.5, 1.0, 1.4])

with c_freq:
    freq_map = {
        "10 Hz (100 ms)": 10,
        "20 Hz (50 ms) — Recommended": 20,
        "50 Hz (20 ms)": 50,
    }
    freq_label = st.selectbox(
        "Sampling Rate",
        options=list(freq_map.keys()),
        index=1,
        disabled=status["is_running"] or status["is_connecting"],
    )
    selected_hz = freq_map[freq_label]

with c_win:
    # (baseline_duration_s, rolling_window_s, prediction_step_s)
    win_map = {
        "120s Baseline → 60s Rolling / 30s Updates (VR Recommended)": (120.0, 60.0, 30.0),
        "120s Baseline → 30s Non-Overlapping Updates": (120.0, 30.0, 30.0),
        "120s Fixed (2 Min Baseline & Predictions)": (120.0, 120.0, 120.0),
        "180s Fixed (3 Min — Official WESAD)": (180.0, 180.0, 180.0),
        "30s Fixed (Quick Live Test)": (30.0, 30.0, 30.0),
    }
    win_label = st.selectbox(
        "Baseline & Prediction Timing",
        options=list(win_map.keys()),
        index=0,
        disabled=status["is_running"] or status["is_connecting"],
        help="Default: 2-min (120s) resting baseline, then smooth 60s rolling predictions pushed to VR every 30s.",
    )
    sel_base_s, sel_roll_s, sel_step_s = win_map[win_label]

with c_btn:
    st.write("")
    st.write("")
    if not status["is_running"] and not status["is_connecting"]:
        if st.button("🟢 Start Live ML", use_container_width=True, type="primary"):
            if manager.start(
                sample_rate_hz=selected_hz,
                window_duration_s=sel_base_s,
                rolling_window_s=sel_roll_s,
                prediction_step_s=sel_step_s,
                keep_baseline=status["has_baseline"],
            ):
                st.rerun()
    else:
        if st.button("🛑 Stop & Save", use_container_width=True, type="secondary"):
            manager.stop()
            st.rerun()

with c_stat:
    st.write("")
    st.write("")
    if status["is_connecting"]:
        st.warning("🟡 Connecting to Vernier GDX-RB over BLE...")
    elif status["is_running"]:
        batt = f"🔋 {status['battery_level']}%" if status["battery_level"] is not None else ""
        st.success(f"🟢 **Live ML Active:** {status['device_name']} {batt}")
    elif status["error_message"]:
        st.error(f"❌ {status['error_message']}")
    else:
        st.info("⚪ Disconnected. Turn on GDX-RB and click **Start Live ML**.")

# ----------------------------------------------------
# 3. LIVE ML PREDICTION OUTPUT PANEL (HERO SECTION)
# ----------------------------------------------------
_link = status["unity_link"]
if not _link["enabled"]:
    st.caption("🔌 Unity link: disabled")
elif _link["status"] == "port_busy":
    st.warning(f"🔌 Unity link: port {_link['address']} is busy ({_link['error']}). Retrying every 2 s...")
elif _link["clients"] > 0:
    st.caption(f"🔌 Unity link: **connected** ({_link['address']})")
else:
    st.caption(f"🔌 Unity link: listening on {_link['address']} — waiting for Unity to connect")

st.divider()
st.markdown("### 🤖 Live Random Forest Prediction Output (VR-Ready)")

win_dur = status["window_duration_s"]
roll_dur = status.get("rolling_window_s", win_dur)
step_dur = status.get("prediction_step_s", win_dur)
base_dur = status.get("baseline_duration_s", 120.0)
completed_wins = status["completed_windows"]
cur_win_elapsed = status["current_window_elapsed"]
elapsed_sec = status["elapsed_seconds"]
elapsed_fmt = str(timedelta(seconds=int(elapsed_sec)))[2:] if elapsed_sec > 0 else "00:00"

if status["is_running"]:
    next_win = completed_wins + 1
    p_val = min(1.0, cur_win_elapsed / win_dur) if win_dur > 0 else 0.0
    if not status["has_baseline"]:
        role_txt = f"Baseline Calibration ({base_dur:.0f}s)"
    elif roll_dur > step_dur:
        role_txt = f"Next Rolling {roll_dur:.0f}s Prediction (#{next_win})"
    else:
        role_txt = f"Inference Window #{next_win}"
    st.progress(
        p_val,
        text=f"⏳ {role_txt}: {cur_win_elapsed:.0f}s / {win_dur:.0f}s | Total Elapsed: {elapsed_fmt} | Samples: {status['sample_count']:,}",
    )
elif completed_wins > 0:
    st.progress(1.0, text=f"✅ Session Complete — {completed_wins} Window(s) Evaluated | Total Duration: {elapsed_fmt}")
else:
    st.progress(
        0.0,
        text=f"Ready — Baseline: {sel_base_s:.0f}s, then {sel_roll_s:.0f}s rolling predictions every {sel_step_s:.0f}s.",
    )

latest_pred = status["latest_prediction"]

col_pred, col_gauge = st.columns([1.3, 1.2])

with col_pred:
    if latest_pred:
        pred_lbl = latest_pred["predicted_label"]
        vr_state = latest_pred["vr_state"]
        conf_pct = latest_pred["confidence"] * 100.0
        stress_pct = latest_pred["stress_probability"] * 100.0
        is_base_win = latest_pred.get("is_baseline_window", False)

        if pred_lbl == "Stress":
            st.error(
                f"## 🔴 STRESS DETECTED (`VR State: High`)\n"
                f"**Confidence:** `{conf_pct:.1f}%` | **Stress Probability:** `{stress_pct:.1f}%` | "
                f"**Window #{latest_pred['window_index']}** (`{latest_pred['window_start_seconds']:.0f}s – {latest_pred['window_end_seconds']:.0f}s`)"
            )
        else:
            tag = " *(Reference Baseline Window Locked)*" if is_base_win else ""
            st.success(
                f"## 🟢 CALM / BASELINE (`VR State: Low`){tag}\n"
                f"**Confidence:** `{conf_pct:.1f}%` | **Stress Probability:** `{stress_pct:.1f}%` | "
                f"**Window #{latest_pred['window_index']}** (`{latest_pred['window_start_seconds']:.0f}s – {latest_pred['window_end_seconds']:.0f}s`)"
            )

        # VR JSON Payload preview
        vr_payload = {
            "type": "reading",
            "window_index": latest_pred["window_index"],
            "status": "Valid",
            "state": vr_state,
            "confidence": round(latest_pred["confidence"], 3),
            "stress_prob": round(latest_pred["stress_probability"], 3),
            "bpm": latest_pred["calculated_bpm"],
            "model": "rsp_only_rf_5feat_v1",
        }
        st.caption("📡 **Exact JSON Payload Ready for Unity VR (`PhysiologicalDataReceiver`):**")
        st.json(vr_payload)
    else:
        st.info(
            "### ⏳ Waiting for First Completed Window...\n"
            f"Once **{win_dur:.0f} seconds** of live breathing completes, the 5 features will be extracted, "
            "subtracted from your baseline, and classified by the **81.06% Random Forest model**."
        )

with col_gauge:
    stress_pct_val = latest_pred["stress_probability"] * 100.0 if latest_pred else 0.0
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=stress_pct_val,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "<b>Live Stress Probability</b><br><span style='font-size:0.8em;color:gray'>5-Feature RSP Random Forest</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#ef4444" if stress_pct_val >= 50 else "#10b981"},
                "steps": [
                    {"range": [0, 40], "color": "#dcfce7"},
                    {"range": [40, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#1e293b", "width": 3}, "thickness": 0.75, "value": 50},
            },
        )
    )
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=55, b=15))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ----------------------------------------------------
# 4. LIVE WAVEFORM + FEATURE DELTA BREAKDOWN
# ----------------------------------------------------
col_wave, col_delta = st.columns([1.4, 1.1])

with col_wave:
    buffer_data = status["data"]
    if buffer_data:
        df_live = pd.DataFrame(buffer_data)
        max_t = df_live["elapsed"].max()
        min_t = max(0.0, max_t - 20.0)
        df_win = df_live[df_live["elapsed"] >= min_t]

        fig_w = go.Figure()
        fig_w.add_trace(
            go.Scatter(
                x=df_win["elapsed"],
                y=df_win["force"],
                mode="lines",
                name="Force (N)",
                line=dict(color="#0284c7", width=2.5, shape="spline"),
                fill="tozeroy",
                fillcolor="rgba(2, 132, 199, 0.08)",
            )
        )
        fig_w.update_layout(
            title=f"<b>Live Respiration Signal</b> ({status['sample_rate_hz']} Hz) | Instant: {status['latest_force']:.2f} N",
            xaxis=dict(title="Elapsed Time (s)", range=[min_t, max(min_t + 5.0, max_t + 0.5)]),
            yaxis=dict(title="Force (N)"),
            template="plotly_white",
            height=300,
            margin=dict(l=35, r=20, t=45, b=35),
        )
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.info("Live 20-second waveform appears here once recording starts.")

with col_delta:
    st.markdown("#### 🧬 Model Input Breakdown (`Live - Baseline = Δ`)")
    if latest_pred:
        rows_tbl = []
        for f in RAW_FEATURE_NAMES:
            rows_tbl.append({
                "Feature": f.replace("rsp_", ""),
                "Live Raw": latest_pred["raw_features"][f],
                "Baseline": latest_pred["baseline_features"][f],
                "Model Input (Δ)": latest_pred["norm_features"][f"norm_{f}"],
            })
        st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True, hide_index=True)
        st.caption(
            f"Detected Breaths: **{latest_pred['n_breaths']}** | "
            f"Breathing Rate: **{latest_pred['calculated_bpm']} bpm** | "
            f"4-Class Hint: **{latest_pred['four_class_label']}**"
        )
    else:
        st.caption("Feature breakdown table will populate as soon as the first window completes.")

# ----------------------------------------------------
# 5. PREDICTION HISTORY & EXPORTS
# ----------------------------------------------------
pred_hist = status["prediction_history"]
if pred_hist:
    with st.expander(f"📋 Window-by-Window ML Prediction Log ({len(pred_hist)} windows)", expanded=True):
        df_hist = pd.DataFrame(pred_hist)
        show_cols = [
            "window_index",
            "window_start_seconds",
            "window_end_seconds",
            "predicted_label",
            "vr_state",
            "confidence",
            "stress_probability",
            "calculated_bpm",
            "n_breaths",
        ]
        st.dataframe(df_hist[[c for c in show_cols if c in df_hist.columns]], use_container_width=True, hide_index=True)

if not status["is_running"] and status["sample_count"] > 0:
    raw_p = Path(status["raw_file"]) if status["raw_file"] else None
    pred_p = Path(status["predictions_file"]) if status["predictions_file"] else None
    d1, d2, _ = st.columns([1.4, 1.4, 2.2])
    if raw_p and raw_p.is_file():
        with d1:
            with open(raw_p, "rb") as f:
                st.download_button("📥 Download Raw CSV", f, file_name=raw_p.name, mime="text/csv", use_container_width=True)
    if pred_p and pred_p.is_file() and completed_wins > 0:
        with d2:
            with open(pred_p, "rb") as f:
                st.download_button("📥 Download ML Predictions CSV", f, file_name=pred_p.name, mime="text/csv", use_container_width=True)

# ----------------------------------------------------
# 6. AUTO-REFRESH WHILE STREAMING
# ----------------------------------------------------
if status["is_running"] or status["is_connecting"]:
    time.sleep(0.35)
    st.rerun()
