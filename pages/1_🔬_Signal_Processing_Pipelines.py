"""Page 1: Signal Processing Pipelines & Feature Extraction."""
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Signal Processing Pipelines", page_icon="🔬", layout="wide")

st.title("🔬 Multimodal Signal Processing Pipelines")
st.write("Detailed architecture of how raw sensor streams (PPG, GSR, RSP) are conditioned, audited, and transformed into the **10 features** from Bian et al. (2019) Table 3.")

st.divider()

# Modality Tabs
tab_table3, tab_ppg, tab_gsr, tab_rsp = st.tabs([
    "📑 The 10 Features (Table 3)",
    "❤️ Photoplethysmogram (PPG)",
    "⚡ Galvanic Skin Response (GSR / EDA)",
    "🫁 Respiration (RSP)",
])

with tab_table3:
    st.subheader("Table 3 Reproduction: Physiological Features in the Model")
    table3_data = [
        {"Modality": "PPG", "Feature Name": "Mean of heart rate variability", "Column Name": "paper_mean_ibi_ms", "Unit": "ms", "Physiological Role": "Measures mean duration between arterial pulse peaks (IBI). Parasympathetic brake reflects stress state."},
        {"Modality": "PPG", "Feature Name": "Mean of amplitude of the peak values", "Column Name": "paper_mean_pulse_amplitude", "Unit": "μV / a.u.", "Physiological Role": "Measures peripheral vasoconstriction; arterial pulse amplitude shrinks under acute stress."},
        {"Modality": "GSR", "Feature Name": "Rate of phasic activity", "Column Name": "paper_phasic_response_rate_per_sec", "Unit": "peaks / s", "Physiological Role": "Frequency of sympathetic skin conductance responses (SCR). Direct index of sympathetic nervous system arousal."},
        {"Modality": "GSR", "Feature Name": "Standard deviation of tonic activity level", "Column Name": "paper_tonic_std_uS", "Unit": "No unit / μS", "Physiological Role": "Slow autonomic baseline variability; tracks sustained emotional and cognitive workload."},
        {"Modality": "GSR", "Feature Name": "Slope of tonic activity", "Column Name": "paper_tonic_slope_uS_per_s", "Unit": "μS / s", "Physiological Role": "Linear drift rate of skin conductance level (SCL) across the 3-minute window."},
        {"Modality": "RSP", "Feature Name": "Mean amplitude", "Column Name": "rsp_mean_amplitude", "Unit": "a.u.", "Physiological Role": "Average absolute excursion of the chest expansion signal across breathing cycles."},
        {"Modality": "RSP", "Feature Name": "Standard deviation of amplitude", "Column Name": "rsp_std_amplitude", "Unit": "a.u.", "Physiological Role": "Variability in tidal volume / breath depth over time."},
        {"Modality": "RSP", "Feature Name": "Subband spectral entropy (0.05–0.35 Hz)", "Column Name": "rsp_spectral_entropy", "Unit": "No unit", "Physiological Role": "Quantifies breath pattern complexity and irregularity in the eupneic frequency band."},
        {"Modality": "RSP", "Feature Name": "Mean of peak-valley magnitude", "Column Name": "rsp_mean_peak_valley_magnitude", "Unit": "a.u.", "Physiological Role": "Average peak-to-trough inhalation depth for discrete breathing cycles."},
        {"Modality": "RSP", "Feature Name": "Standard deviation of peak-valley magnitude", "Column Name": "rsp_std_peak_valley_magnitude", "Unit": "a.u.", "Physiological Role": "Breath-to-breath tidal volume variability; irregular under cognitive stress."},
    ]
    st.dataframe(pd.DataFrame(table3_data), hide_index=True, use_container_width=True)

with tab_ppg:
    st.subheader("Photoplethysmography (PPG) Processing Pipeline")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown(r"""
        * **Hardware**: Empatica E4 wristband optical sensor.
        * **Sampling Rate**: 64 Hz.
        * **Bandpass Filter**: 3rd-order Butterworth bandpass, **1.1 Hz to 3.6 Hz** (66 to 216 bpm physiological passband).
        * **Peak Detection**: `scipy.signal.find_peaks` with minimum inter-peak distance of **0.35 seconds** (maximum physiological heart rate of 171.4 bpm).
        * **Unified IBI Quality Audit**:
          - Calculates beat-to-beat Inter-Beat Intervals ($IBI_i = t_{i} - t_{i-1}$).
          - Detects non-physiologic outliers (<350 ms or >1800 ms).
          - Detects ectopic / missed beats: flags IBIs that deviate by $>20\%$ from a rolling median neighborhood.
          - Replaces flagged intervals with `NaN` so outlier beats never corrupt window feature calculations.
        * **Window Aggregation (180s)**:
          - Validates that $\ge 90$ valid heartbeats exist per 3-minute window (minimum 30 bpm).
          - Computes `paper_mean_ibi_ms` using only the audited, cleaned intervals.
        """)
    with col2:
        img_path = Path(__file__).resolve().parent.parent / "ppg_output_final" / "02_peaks_zoom_300s.png"
        if img_path.is_file():
            st.image(str(img_path), caption="Audited PPG Pulse Waveform & Detected Systolic Peaks (Zoomed 300s window)", use_container_width=True)
        else:
            st.info("Peak audit image will appear here.")

with tab_gsr:
    st.subheader("Galvanic Skin Response (GSR / EDA) Pipeline")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown(r"""
        * **Hardware**: Empatica E4 wristband electrodermal electrodes.
        * **Sampling Rate**: 4 Hz.
        * **Filtering & Decomposition**:
          - Lowpass filter at **0.5 Hz** (eliminates high-frequency sensor noise and line interference).
          - **Tonic (SCL) Decomposition**: 2nd-order Butterworth lowpass filter with cutoff at **0.05 Hz**.
          - **Phasic (SCR) Subtraction**: $GSR_{phasic} = GSR_{filtered} - GSR_{tonic}$.
        * **Phasic Peak Detection**:
          - Minimum peak height threshold: **$\ge 0.01\ \mu\text{S}$**.
          - Minimum refractory period: **$\ge 1.0\text{ s}$**.
        * **Extracted Window Features**:
          - `paper_phasic_response_rate_per_sec`: Number of phasic peaks divided by window duration (180s).
          - `paper_tonic_std_uS`: Standard deviation of the slow tonic component across the window.
          - `paper_tonic_slope_uS_per_s`: Linear trend / drift of the tonic baseline.
        """)
    with col2:
        st.markdown("""
        <div style="background-color: #f1f5f9; padding: 1.2rem; border-radius: 8px; border-left: 4px solid #0284c7;">
            <h4 style="color: #0369a1; margin-top:0;">Autonomic Arousal Significance</h4>
            <p style="font-size: 0.95rem; color: #334155;">
                Eccrine sweat glands on the wrist are exclusively innervated by the <b>sympathetic nervous system</b>.
                Under acute stress (e.g. mental arithmetic), sympathetic discharge triggers rapid sweat duct filling,
                producing sharp phasic spikes. This explains why <b>GSR Phasic Rate is our #1 ranked feature (26.1% importance)</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab_rsp:
    st.subheader("Respiration (RSP) Processing Pipeline")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown(r"""
        * **Hardware**: RespiBAN chest inductive plethysmography band.
        * **Original Sampling Rate**: 700 Hz $\rightarrow$ Decimated with anti-aliasing FIR filter to **100 Hz**.
        * **Bandpass Filter**: 2nd-order Butterworth bandpass, **0.05 Hz to 0.35 Hz** (3 to 21 breaths/minute).
        * **Breath Cycle Detection**:
          - Zero-crossing and local extremum detection.
          - Minimum breath distance: **2.85 seconds** (maximum respiratory rate of 21 breaths/min).
        * **Spectral Entropy Calculation**:
          - Welch PSD calculated over the eupneic frequency band ($0.05 - 0.35\text{ Hz}$).
          - Shannon entropy normalized over the spectral bins:
            $$H = -\sum_{f} p(f) \log_2 p(f)$$
          - Higher entropy reflects chaotic, irregular breathing under cognitive load.
        """)
    with col2:
        img_path = Path(__file__).resolve().parent.parent / "rsp_output" / "01_rsp_raw_vs_filtered_120s.png"
        if img_path.is_file():
            st.image(str(img_path), caption="Raw Chest Respiration vs. 0.05–0.35 Hz Filtered Signal (120s Zoom)", use_container_width=True)
        else:
            st.info("Respiration waveform plot available in rsp_output.")
