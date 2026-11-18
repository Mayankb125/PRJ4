"""Page 5: Interactive Live Inference Sandbox & What-If Simulator."""
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Interactive Inference Sandbox", page_icon="🧪", layout="wide")

st.title("🧪 Interactive Live Inference Sandbox")
st.write("Test the trained Random Forest model in real time! Adjust the sliders for the **10 physiological features** (normalized against baseline) or click a preset to see the model's instant prediction.")

st.divider()

RESULTS_DIR = Path(__file__).resolve().parent.parent / "master_dataset" / "loso_results"
bin_model_path = RESULTS_DIR / "baseline_vs_stress_normalized_final_model.joblib"
four_model_path = RESULTS_DIR / "four_class_normalized_final_model.joblib"

if not bin_model_path.is_file() or not four_model_path.is_file():
    st.error("Trained models not found.")
    st.stop()

@st.cache_resource
def load_models():
    bin_b = joblib.load(bin_model_path)
    four_b = joblib.load(four_model_path)
    return bin_b, four_b

binary_bundle, four_bundle = load_models()
bin_model = binary_bundle["model"]
bin_features = binary_bundle["features"]
four_model = four_bundle["model"]
four_features = four_bundle["features"]

# Presets
st.markdown("#### Choose a Simulation Preset or Customize Below:")
col_p1, col_p2, col_p3 = st.columns(3)

default_vals = {
    "norm_paper_mean_ibi_ms": 0.0,
    "norm_paper_mean_pulse_amplitude": 0.0,
    "norm_paper_phasic_response_rate_per_sec": 0.0,
    "norm_paper_tonic_std_uS": 0.0,
    "norm_paper_tonic_slope_uS_per_s": 0.0,
    "norm_rsp_mean_amplitude": 0.0,
    "norm_rsp_std_amplitude": 0.0,
    "norm_rsp_spectral_entropy": 0.0,
    "norm_rsp_mean_peak_valley_magnitude": 0.0,
    "norm_rsp_std_peak_valley_magnitude": 0.0,
}

if col_p1.button("🧘 Calm Resting Baseline", use_container_width=True):
    st.session_state["vals"] = {
        "norm_paper_mean_ibi_ms": 25.0,
        "norm_paper_mean_pulse_amplitude": 5.0,
        "norm_paper_phasic_response_rate_per_sec": -0.05,
        "norm_paper_tonic_std_uS": -0.05,
        "norm_paper_tonic_slope_uS_per_s": -0.005,
        "norm_rsp_mean_amplitude": 0.1,
        "norm_rsp_std_amplitude": -0.05,
        "norm_rsp_spectral_entropy": -0.15,
        "norm_rsp_mean_peak_valley_magnitude": 0.1,
        "norm_rsp_std_peak_valley_magnitude": -0.05,
    }
elif col_p2.button("🔥 Acute TSST Mental Stress", use_container_width=True):
    st.session_state["vals"] = {
        "norm_paper_mean_ibi_ms": -120.0,     # Heart rate jumped 15-20 bpm
        "norm_paper_mean_pulse_amplitude": -15.0, # Peripheral vasoconstriction
        "norm_paper_phasic_response_rate_per_sec": 0.18, # 18 more sweat peaks per 100s
        "norm_paper_tonic_std_uS": 0.65,      # High tonic fluctuations
        "norm_paper_tonic_slope_uS_per_s": 0.012, # Upward drift
        "norm_rsp_mean_amplitude": -0.6,      # Shallow chest expansion
        "norm_rsp_std_amplitude": 0.45,       # High amplitude variability
        "norm_rsp_spectral_entropy": 0.35,    # High spectral disorder
        "norm_rsp_mean_peak_valley_magnitude": -0.5,
        "norm_rsp_std_peak_valley_magnitude": 0.4,
    }
elif col_p3.button("🔄 Neutral Baseline State", use_container_width=True):
    st.session_state["vals"] = default_vals.copy()

current_vals = st.session_state.get("vals", default_vals)

st.divider()

col_inputs, col_outputs = st.columns([1.3, 1.2])

with col_inputs:
    st.subheader("Adjust Physiological Parameters (Baseline-Normalized)")
    
    with st.expander("❤️ Heart Rate & Pulse (PPG)", expanded=True):
        ibi = st.slider("Mean IBI Deviation (ms)", -250.0, 250.0, float(current_vals["norm_paper_mean_ibi_ms"]), step=5.0, help="Negative means faster heart rate (stress)")
        pulse_amp = st.slider("Pulse Amplitude Deviation", -40.0, 40.0, float(current_vals["norm_paper_mean_pulse_amplitude"]), step=1.0, help="Negative means vasoconstriction (stress)")

    with st.expander("⚡ Autonomic Sweat (GSR / EDA)", expanded=True):
        phasic_rate = st.slider("Phasic Response Rate (peaks/s)", -0.15, 0.35, float(current_vals["norm_paper_phasic_response_rate_per_sec"]), step=0.01, help="Positive means frequent sweat bursts (stress)")
        tonic_std = st.slider("Tonic Std Deviation (μS)", -0.5, 2.5, float(current_vals["norm_paper_tonic_std_uS"]), step=0.05)
        tonic_slope = st.slider("Tonic Slope (μS/s)", -0.03, 0.03, float(current_vals["norm_paper_tonic_slope_uS_per_s"]), step=0.001)

    with st.expander("🫁 Respiration Dynamics (RSP)", expanded=False):
        rsp_mean_amp = st.slider("RSP Mean Amplitude", -1.5, 1.5, float(current_vals["norm_rsp_mean_amplitude"]), step=0.05)
        rsp_std_amp = st.slider("RSP Std Amplitude", -1.0, 1.5, float(current_vals["norm_rsp_std_amplitude"]), step=0.05)
        rsp_entropy = st.slider("RSP Spectral Entropy", -0.8, 0.8, float(current_vals["norm_rsp_spectral_entropy"]), step=0.02)
        rsp_mean_pv = st.slider("RSP Mean Peak-Valley Mag", -1.5, 1.5, float(current_vals["norm_rsp_mean_peak_valley_magnitude"]), step=0.05)
        rsp_std_pv = st.slider("RSP Std Peak-Valley Mag", -1.0, 1.5, float(current_vals["norm_rsp_std_peak_valley_magnitude"]), step=0.05)

# Vector construction for inference
input_features = {
    "norm_paper_mean_ibi_ms": ibi,
    "norm_paper_mean_pulse_amplitude": pulse_amp,
    "norm_paper_phasic_response_rate_per_sec": phasic_rate,
    "norm_paper_tonic_std_uS": tonic_std,
    "norm_paper_tonic_slope_uS_per_s": tonic_slope,
    "norm_rsp_mean_amplitude": rsp_mean_amp,
    "norm_rsp_std_amplitude": rsp_std_amp,
    "norm_rsp_spectral_entropy": rsp_entropy,
    "norm_rsp_mean_peak_valley_magnitude": rsp_mean_pv,
    "norm_rsp_std_peak_valley_magnitude": rsp_std_pv,
}

X_input = np.array([[input_features[f] for f in bin_features]], dtype=float)

# Run Inference
bin_pred = bin_model.predict(X_input)[0]
bin_probs = bin_model.predict_proba(X_input)[0]
bin_classes = list(bin_model.classes_)
stress_idx = bin_classes.index("Stress")
stress_prob = bin_probs[stress_idx]

four_pred = four_model.predict(X_input)[0]
four_probs = four_model.predict_proba(X_input)[0]
four_classes = list(four_model.classes_)

with col_outputs:
    st.subheader("Model Real-Time Inference Results")
    
    # Stress Gauge
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stress_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Stress Probability", 'font': {'size': 20, 'color': "#1e3a8a"}},
        number={'suffix': "%", 'font': {'size': 36, 'color': "#b91c1c" if stress_prob >= 0.5 else "#15803d"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#b91c1c" if stress_prob >= 0.5 else "#22c55e"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#dcfce7'},
                {'range': [30, 60], 'color': '#fef9c3'},
                {'range': [60, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    gauge_fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Classification Badge
    if bin_pred == "Stress":
        st.error(f"🚨 **Binary Classification**: **STRESS / HIGH ACTIVATION** (Confidence: {stress_prob:.1%})")
    else:
        st.success(f"🧘 **Binary Classification**: **BASELINE / CALM REST** (Confidence: {(1-stress_prob):.1%})")

    st.markdown("---")
    st.markdown("#### 4-Class Probability Distribution:")
    prob_df = pd.DataFrame({
        "Condition": four_classes,
        "Probability": [p * 100 for p in four_probs]
    }).sort_values("Probability", ascending=True)
    
    fig_bars = px.bar(
        prob_df,
        x="Probability",
        y="Condition",
        orientation="h",
        text=prob_df["Probability"].apply(lambda x: f"{x:.1f}%"),
        color="Condition",
        color_discrete_map={
            "Baseline": "#22c55e",
            "Stress": "#ef4444",
            "Amusement": "#3b82f6",
            "Meditation": "#a855f7"
        }
    )
    fig_bars.update_layout(height=220, showlegend=False, xaxis_range=[0, 100], margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_bars, use_container_width=True)
