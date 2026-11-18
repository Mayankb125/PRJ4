"""Page 3: Machine Learning & Leave-One-Subject-Out (LOSO) Benchmark Results."""
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Machine Learning LOSO Benchmark", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning & LOSO Cross-Validation Benchmark")
st.write("Evaluating Random Forest models using strict **Leave-One-Subject-Out (LOSO)** cross-validation across all 15 subjects to ensure zero inter-subject data leakage.")

st.divider()

RESULTS_DIR = Path(__file__).resolve().parent.parent / "master_dataset" / "loso_results"
if not RESULTS_DIR.is_dir():
    st.error("LOSO results directory not found.")
    st.stop()

# 1. Experiments Summary Table
st.subheader("1. Comprehensive Model Evaluation Matrix")
summary_csv = RESULTS_DIR / "loso_experiment_summary.csv"
if summary_csv.is_file():
    sum_df = pd.read_csv(summary_csv)
    display_cols = [
        "experiment", "target", "windows", "accuracy_oof", 
        "balanced_accuracy_oof", "macro_f1_oof", "mean_fold_accuracy"
    ]
    formatted_df = sum_df[display_cols].copy()
    formatted_df["accuracy_oof"] = formatted_df["accuracy_oof"].apply(lambda x: f"{x*100:.2f}%")
    formatted_df["balanced_accuracy_oof"] = formatted_df["balanced_accuracy_oof"].apply(lambda x: f"{x*100:.2f}%")
    formatted_df["macro_f1_oof"] = formatted_df["macro_f1_oof"].apply(lambda x: f"{x:.3f}")
    formatted_df["mean_fold_accuracy"] = formatted_df["mean_fold_accuracy"].apply(lambda x: f"{x*100:.2f}%")
    formatted_df.columns = ["Experiment", "Target Setup", "Windows", "Out-of-Fold Acc", "Balanced Acc", "Macro-F1", "Mean Fold Acc"]
    st.dataframe(formatted_df, hide_index=True, use_container_width=True)

# Callout on Baseline Subtraction
st.success("""
🎯 **Key Scientific Insight — Power of Baseline Subtraction**:
Subtracting each subject's 3-minute resting baseline features (as proposed by Bian et al. 2019) raises cross-subject binary accuracy from **83.33% to 90.91% (+7.58% boost)** and 4-class accuracy from **68.60% to 72.73% (+4.13% boost)**!
""")

st.divider()

# 2. Confusion Matrices & Classification Reports
st.subheader("2. Confusion Matrix & Error Analysis")
tab_bin, tab_four = st.tabs(["🔥 Binary Model (Baseline vs. Stress - 90.91%)", "🌐 4-Class Model (72.73%)"])

with tab_bin:
    col_cm, col_rep = st.columns([1.2, 1.8])
    with col_cm:
        cm_csv = RESULTS_DIR / "baseline_vs_stress_normalized_confusion_matrix.csv"
        if cm_csv.is_file():
            cm_df = pd.read_csv(cm_csv, index_col=0)
            z = cm_df.values
            x_labels = [c.replace("predicted_", "") for c in cm_df.columns]
            y_labels = [r.replace("actual_", "") for r in cm_df.index]
            
            fig_cm = px.imshow(
                z,
                labels=dict(x="Predicted Condition", y="True Condition", color="Windows"),
                x=x_labels,
                y=y_labels,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Binary Confusion Matrix (Normalized LOSO)"
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            total_corr = z[0, 0] + z[1, 1]
            total_all = z.sum()
            st.caption(f"Correctly Classified: **{total_corr} / {total_all} windows ({total_corr/total_all*100:.2f}%)**")
            
    with col_rep:
        st.markdown("#### Per-Class Performance Breakdown")
        rep_csv = RESULTS_DIR / "baseline_vs_stress_normalized_classification_report.csv"
        if rep_csv.is_file():
            rep_df = pd.read_csv(rep_csv, index_col=0)
            st.dataframe(rep_df.style.format("{:.3f}"), use_container_width=True)
        st.markdown("""
        * **Baseline Sensitivity (Recall)**: $82 / 89 = \\mathbf{92.1\\%}$
        * **Stress Sensitivity (Recall)**: $38 / 43 = \\mathbf{88.4\\%}$
        * **Precision for Stress**: $38 / (38 + 7) = \\mathbf{84.4\\%}$
        """)

with tab_four:
    col_cm4, col_rep4 = st.columns([1.2, 1.8])
    with col_cm4:
        cm4_csv = RESULTS_DIR / "four_class_normalized_confusion_matrix.csv"
        if cm4_csv.is_file():
            cm4_df = pd.read_csv(cm4_csv, index_col=0)
            z4 = cm4_df.values
            x4 = [c.replace("predicted_", "") for c in cm4_df.columns]
            y4 = [r.replace("actual_", "") for r in cm4_df.index]
            
            fig_cm4 = px.imshow(
                z4,
                labels=dict(x="Predicted Condition", y="True Condition", color="Windows"),
                x=x4,
                y=y4,
                text_auto=True,
                color_continuous_scale="Greens",
                title="4-Class Confusion Matrix (Normalized LOSO)"
            )
            fig_cm4.update_layout(height=400)
            st.plotly_chart(fig_cm4, use_container_width=True)
            
    with col_rep4:
        st.markdown("#### 4-Class Classification Metrics")
        rep4_csv = RESULTS_DIR / "four_class_normalized_classification_report.csv"
        if rep4_csv.is_file():
            rep4_df = pd.read_csv(rep4_csv, index_col=0)
            st.dataframe(rep4_df.style.format("{:.3f}"), use_container_width=True)
        st.info("Notice that the main confusion in the 4-class setup is between **Baseline** and **Meditation**, both of which involve calm breathing and low autonomic arousal.")

st.divider()

# 3. Feature Importance Analysis
st.subheader("3. Feature Importance Ranking (What Drives the Random Forest?)")
col_feat, col_feat_text = st.columns([1.5, 1])

with col_feat:
    feat_csv = RESULTS_DIR / "baseline_vs_stress_normalized_feature_importance.csv"
    if feat_csv.is_file():
        feat_df = pd.read_csv(feat_csv).sort_values("importance", ascending=True)
        feat_df["feature_clean"] = feat_df["feature"].str.replace("norm_", "")
        fig_bar = px.bar(
            feat_df,
            x="importance",
            y="feature_clean",
            orientation="h",
            color="importance",
            color_continuous_scale="Reds",
            title="Feature Importance in Baseline vs. Stress Classification",
            labels={"importance": "Gini Importance", "feature_clean": "Physiological Feature"}
        )
        fig_bar.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

with col_feat_text:
    st.markdown("""
    #### Modality Contribution Breakdown
    1. **Autonomic Sweat Activity (GSR - #1 Top Feature)**:
       * `paper_phasic_response_rate_per_sec` accounts for **26.1% of all tree splits**.
       * Sympathetic electrodermal bursts provide the sharpest distinction between resting baseline and acute math/speech stress.
    2. **Respiratory Mechanics (RSP - #2 Top Feature)**:
       * `rsp_mean_amplitude` (20.3%) and `rsp_spectral_entropy` (8.6%) reveal shallow, erratic chest wall motion under cognitive strain.
    3. **Cardiovascular Regulation (PPG - #3 Top Feature)**:
       * `paper_mean_ibi_ms` (13.0%) captures the parasympathetic vagal withdrawal that accelerates heart rate during stress.
    """)
