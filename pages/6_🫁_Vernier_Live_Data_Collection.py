"""Page 6: Vernier Go Direct Respiration — Live Data Collection & 3-Min Feature Pipeline."""
from datetime import timedelta
from pathlib import Path
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure PhysioSync_Live is in Python path
LIVE_DIR = Path(__file__).parent.parent / "PhysioSync_Live"
if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))

from acquisition_manager import VernierAcquisitionManager, WINDOW_DURATION_SECONDS

st.set_page_config(
    page_title="Vernier Live Respiration & 3-Min Feature Pipeline",
    page_icon="🫁",
    layout="wide",
)

st.title("🫁 Vernier Go Direct Respiration — Live Data Collection & Pipeline")
st.write(
    "Connects directly to your **Vernier GDX-RB Respiration Belt** over Bluetooth Low Energy (BLE), "
    "streams real-time chest biomechanics, and computes all **5 Bian et al. (2019 Table 3) respiration features** "
    "strictly at the completion of every **3-minute (180-second) window**."
)

st.divider()

manager = VernierAcquisitionManager.get_instance()
status = manager.get_snapshot()

# ----------------------------------------------------
# 1. ACQUISITION CONTROLS
# ----------------------------------------------------
col_freq, col_btn, col_status = st.columns([1.2, 1.2, 1.6])

with col_freq:
    freq_map = {
        "10 Hz (100 ms) — Standard BLE": 10,
        "20 Hz (50 ms) — High Precision": 20,
        "50 Hz (20 ms) — Ultra-Fine": 50,
    }
    freq_choice = st.selectbox(
        "Sampling Frequency",
        options=list(freq_map.keys()),
        index=1,  # Default to 20 Hz
        disabled=status["is_running"] or status["is_connecting"],
        help="20 Hz is the sweet spot for smooth chest biomechanics and low BLE jitter.",
    )
    selected_hz = freq_map[freq_choice]

with col_btn:
    st.write("")
    st.write("")
    if not status["is_running"] and not status["is_connecting"]:
        if st.button("🟢 Start Live Recording", use_container_width=True, type="primary"):
            success = manager.start(sample_rate_hz=selected_hz)
            if success:
                st.rerun()
    else:
        if st.button("🛑 Stop & Save Recording", use_container_width=True, type="secondary"):
            manager.stop()
            st.rerun()

with col_status:
    st.write("")
    st.write("")
    if status["is_connecting"]:
        st.warning("🟡 Searching & Connecting to GDX-RB over BLE...")
    elif status["is_running"]:
        battery_str = f"🔋 {status['battery_level']}%" if status["battery_level"] is not None else ""
        st.success(f"🟢 **Recording Live:** {status['device_name']}  {battery_str}")
    elif status["error_message"]:
        st.error(f"❌ {status['error_message']}")
    else:
        st.info("⚪ Sensor Disconnected. Turn on GDX-RB and click **Start**.")

# ----------------------------------------------------
# 2. LIVE SIGNAL METRICS (SENSOR LEVEL)
# ----------------------------------------------------
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

cur_force = status["latest_force"]
cur_bpm = status["latest_bpm"]
elapsed_sec = status["elapsed_seconds"]
elapsed_fmt = str(timedelta(seconds=int(elapsed_sec)))[2:] if elapsed_sec > 0 else "00:00"
has_session_data = status["is_running"] or status["sample_count"] > 0

with m_col1:
    st.metric(
        label="Instant Force (N)",
        value=f"{cur_force:.2f} N" if status["is_running"] else "--",
        help="Real-time chest expansion/contraction force measured by the belt.",
    )

with m_col2:
    st.metric(
        label="Sensor BPM (On-board)",
        value=f"{cur_bpm:.1f} bpm" if (status["is_running"] and cur_bpm) else "--",
        help="Vernier internal calculated breaths per minute (updates every 10s).",
    )

with m_col3:
    st.metric(
        label="Total Elapsed Time",
        value=elapsed_fmt if has_session_data else "00:00",
        help="Total recording duration (MM:SS).",
    )

with m_col4:
    st.metric(
        label="Samples Logged",
        value=f"{status['sample_count']:,}" if has_session_data else "0",
        help=f"Continuous data points captured at {status['sample_rate_hz']} Hz.",
    )

# ----------------------------------------------------
# 3. REAL-TIME WAVEFORM GRAPH (ROLLING 20s VIEW)
# ----------------------------------------------------
buffer_data = status["data"]

if buffer_data and len(buffer_data) > 0:
    df_live = pd.DataFrame(buffer_data)

    # Rolling 20-second viewport
    max_t = df_live["elapsed"].max()
    min_t = max(0.0, max_t - 20.0)
    df_window = df_live[df_live["elapsed"] >= min_t]

    fig = go.Figure()

    # Primary Line: Continuous Respiration Force (N)
    fig.add_trace(
        go.Scatter(
            x=df_window["elapsed"],
            y=df_window["force"],
            mode="lines",
            name="Force (N)",
            line=dict(color="#0284c7", width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(2, 132, 199, 0.08)",
        )
    )

    fig.update_layout(
        title={
            "text": f"<b>Live Respiration Waveform</b> — Rolling 20s View ({status['sample_rate_hz']} Hz)",
            "x": 0.02,
            "y": 0.95,
        },
        xaxis=dict(
            title="Elapsed Time (seconds)",
            range=[min_t, max(min_t + 5.0, max_t + 0.5)],
            gridcolor="#f1f5f9",
            zeroline=False,
        ),
        yaxis=dict(
            title="Force (Newtons)",
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#cbd5e1",
        ),
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    fig_empty = go.Figure()
    fig_empty.update_layout(
        title="Live Respiration Waveform — Ready to Record",
        xaxis=dict(title="Elapsed Time (seconds)", range=[0, 20]),
        yaxis=dict(title="Force (Newtons)", range=[-5, 25]),
        template="plotly_white",
        height=340,
    )
    fig_empty.add_annotation(
        text="Click '🟢 Start Live Recording' above to begin streaming data",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#64748b"),
    )
    st.plotly_chart(fig_empty, use_container_width=True)

# ----------------------------------------------------
# 4. STRICT 3-MINUTE (180s) BIAN ET AL. TARGET FEATURES
# ----------------------------------------------------
st.markdown("### 🧬 3-Minute Respiration Pipeline Output (Bian et al. 2019 Table 3)")

completed_wins = status["completed_windows"]
cur_win_elapsed = status["current_window_elapsed"]
latest_feats = status["latest_features"]

# Progress Bar Logic (Strict 180s Blocks)
if status["is_running"]:
    next_win_num = completed_wins + 1
    p_val = min(1.0, cur_win_elapsed / WINDOW_DURATION_SECONDS)
    win_label = "Baseline Window #1" if next_win_num == 1 else f"Window #{next_win_num}"
    if completed_wins == 0:
        st.progress(
            p_val,
            text=f"⏳ Collecting {win_label}: {cur_win_elapsed:.0f}s / 180s — Features will be calculated automatically at 03:00",
        )
    else:
        st.progress(
            p_val,
            text=f"✅ Completed {completed_wins} Window(s) | ⏳ Collecting {win_label}: {cur_win_elapsed:.0f}s / 180s",
        )
elif completed_wins > 0:
    st.progress(
        1.0,
        text=f"✅ Recording Finished — {completed_wins} Complete 3-Minute Window(s) Extracted (Total Duration: {elapsed_fmt})",
    )
elif status["sample_count"] > 0:
    st.warning(
        f"⚠️ Session stopped at {elapsed_fmt} before completing a full 180-second window. "
        "Record at least 3 full minutes to generate official Bian et al. features."
    )
else:
    st.progress(0.0, text="Ready — Features are computed strictly at the end of each 3-minute (180s) window.")

# 5 Feature Metric Cards
f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)

if latest_feats and latest_feats.get("status") == "OK":
    with f_col1:
        st.metric(
            label="Mean Amplitude",
            value=f"{latest_feats['rsp_mean_amplitude']:.3f} N",
            help="1. rsp_mean_amplitude: Mean magnitude of the filtered breathing wave.",
        )
    with f_col2:
        st.metric(
            label="Std Amplitude",
            value=f"{latest_feats['rsp_std_amplitude']:.3f} N",
            help="2. rsp_std_amplitude: Standard deviation of breathing amplitude.",
        )
    with f_col3:
        st.metric(
            label="Spectral Entropy",
            value=f"{latest_feats['rsp_spectral_entropy']:.3f}",
            help="3. rsp_spectral_entropy: Welch PSD normalized entropy in the 0.05-0.35 Hz band.",
        )
    with f_col4:
        st.metric(
            label="Breath Depth (Mean)",
            value=f"{latest_feats['rsp_mean_peak_valley_magnitude']:.3f} N",
            help="4. rsp_mean_peak_valley_magnitude: Average inhalation peak minus preceding trough.",
        )
    with f_col5:
        st.metric(
            label="Breath Depth (Std)",
            value=f"{latest_feats['rsp_std_peak_valley_magnitude']:.3f} N",
            help="5. rsp_std_peak_valley_magnitude: Standard deviation of breath depths.",
        )

    win_idx = latest_feats.get("window_index", 1)
    win_s = latest_feats.get("window_start_seconds", 0.0)
    win_e = latest_feats.get("window_end_seconds", 180.0)
    st.caption(
        f"🎯 **Showing Official Window #{win_idx} ({win_s:.0f}s – {win_e:.0f}s):** "
        f"Duration: **{latest_feats['duration_seconds']}s** | "
        f"Detected Breaths: **{latest_feats['n_breaths']}** | "
        f"Calculated Rate: **{latest_feats['calculated_bpm']} bpm** | "
        f"Filter: **0.05 – 0.35 Hz Butterworth Bandpass**"
    )
else:
    with f_col1:
        st.metric(label="Mean Amplitude", value="-- N")
    with f_col2:
        st.metric(label="Std Amplitude", value="-- N")
    with f_col3:
        st.metric(label="Spectral Entropy", value="--")
    with f_col4:
        st.metric(label="Breath Depth (Mean)", value="-- N")
    with f_col5:
        st.metric(label="Breath Depth (Std)", value="-- N")

# ----------------------------------------------------
# 5. COMPLETED 3-MINUTE WINDOWS LOG & EXPORTS
# ----------------------------------------------------
feat_history = status["feature_history"]
if feat_history and len(feat_history) > 0:
    with st.expander(f"📊 Completed 3-Minute Feature Windows ({len(feat_history)} total)", expanded=True):
        df_feats = pd.DataFrame(feat_history)
        cols_order = [
            "window_index",
            "window_start_seconds",
            "window_end_seconds",
            "calculated_bpm",
            "n_breaths",
            "rsp_mean_amplitude",
            "rsp_std_amplitude",
            "rsp_spectral_entropy",
            "rsp_mean_peak_valley_magnitude",
            "rsp_std_peak_valley_magnitude",
        ]
        available_cols = [c for c in cols_order if c in df_feats.columns]
        st.dataframe(df_feats[available_cols], use_container_width=True)

# Post-Session Downloads when Stopped
if not status["is_running"] and status["sample_count"] > 0:
    raw_path = Path(status["output_file"]) if status["output_file"] else None
    feat_path = Path(status["features_file"]) if status["features_file"] else None

    if raw_path and raw_path.is_file():
        st.success(f"💾 **Recording Finished:** `{raw_path.name}` ({status['sample_count']:,} samples)")
        col_d1, col_d2, col_sp = st.columns([1.5, 1.5, 2.0])

        with col_d1:
            with open(raw_path, "rb") as f:
                st.download_button(
                    label="📥 Download Raw Readings CSV",
                    data=f,
                    file_name=raw_path.name,
                    mime="text/csv",
                    use_container_width=True,
                )

        if feat_path and feat_path.is_file() and completed_wins > 0:
            with col_d2:
                with open(feat_path, "rb") as f:
                    st.download_button(
                        label="📥 Download 3-Min Features CSV",
                        data=f,
                        file_name=feat_path.name,
                        mime="text/csv",
                        use_container_width=True,
                    )

# ----------------------------------------------------
# 6. AUTO-REFRESH STREAMING LOOP
# ----------------------------------------------------
if status["is_running"] or status["is_connecting"]:
    time.sleep(0.3)
    st.rerun()
