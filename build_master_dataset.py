"""
BUILD MASTER MULTIMODAL DATASET — WESAD
=======================================
Batch processing script that coordinates all 3 validated pipelines:
  - PPG (ppg_wesad_pipeline_final.py)
  - GSR (gsr_wesad_pipeline_final.py)
  - Respiration (rsp_wesad_pipeline_final.py)

Operations:
1. Dynamically discovers all available WESAD subjects (S2 through S17).
2. Loads each subject pickle once (conserving RAM).
3. Executes the exact 3 validated signal cleaning pipelines.
4. Merges the 10 paper features (Table 3, Bian et al. 2019) on window_index.
5. Computes subject-specific Baseline Normalization (norm_ features).
6. Exports:
     - master_dataset/subjects/Sx_multimodal.csv (individual subject datasets)
     - master_dataset/wesad_multimodal_master_180s.csv (all subjects unified)
     - master_dataset/master_dataset_summary.txt (dataset audit and class counts)
"""

import sys
import gc
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Import functions directly from our 3 validated pipeline modules
import ppg_wesad_pipeline_final as ppg_pipe
import gsr_wesad_pipeline_final as gsr_pipe
import rsp_wesad_pipeline_final as rsp_pipe

# =============================================================================
# CONFIGURATION
# =============================================================================
WORKSPACE_DIR = Path(__file__).resolve().parent
WESAD_DIR = WORKSPACE_DIR / "WESAD" / "WESAD"
OUTPUT_DIR = WORKSPACE_DIR / "master_dataset"
SUBJECTS_OUT_DIR = OUTPUT_DIR / "subjects"

PAPER_10_FEATURES = [
    # PPG (2 features)
    "paper_mean_ibi_ms",
    "paper_mean_pulse_amplitude",
    # GSR (3 features)
    "paper_phasic_response_rate_per_sec",
    "paper_tonic_std_uS",
    "paper_tonic_slope_uS_per_s",
    # RSP (5 features)
    "rsp_mean_amplitude",
    "rsp_std_amplitude",
    "rsp_spectral_entropy",
    "rsp_mean_peak_valley_magnitude",
    "rsp_std_peak_valley_magnitude"
]


def process_subject(pkl_path: Path, subject_id: str, subj_out_dir: Path) -> pd.DataFrame:
    """Process a single subject through all 3 pipelines and return unified multimodal DataFrame."""
    subj_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load pickle file once
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    raw_bvp = data["signal"]["wrist"]["BVP"].flatten().astype(np.float64)
    raw_eda = data["signal"]["wrist"]["EDA"].flatten().astype(np.float64)
    raw_resp = np.asarray(data["signal"]["chest"]["Resp"]).ravel().astype(np.float64)
    labels = data["label"].flatten().astype(np.int32)
    
    # Clean pickle from memory
    del data
    gc.collect()

    # 2. PPG Pipeline (64 Hz BVP -> 1.1-3.6 Hz bandpass -> peaks -> audit -> 180s windows)
    filtered_bvp = ppg_pipe.apply_bandpass(
        raw_bvp, ppg_pipe.FS_BVP, ppg_pipe.FILTER_LOW, ppg_pipe.FILTER_HIGH
    )
    bvp_peaks = ppg_pipe.detect_peaks(filtered_bvp)
    ibi_audit_df = ppg_pipe.clean_ibi_intervals(bvp_peaks, filtered_bvp, subj_out_dir)
    df_ppg = ppg_pipe.extract_features(
        ibi_audit_df, labels, len(filtered_bvp), subject_id, subj_out_dir
    )

    # 3. GSR Pipeline (4 Hz EDA -> 0.5/0.05 Hz decomposition -> phasic peaks -> 180s windows)
    smoothed_eda, tonic_eda, phasic_eda = gsr_pipe.decompose_eda(raw_eda, gsr_pipe.FS_EDA)
    eda_peaks, eda_audit_df = gsr_pipe.detect_and_audit_phasic_peaks(
        phasic_eda, tonic_eda, subj_out_dir
    )
    df_gsr, _ = gsr_pipe.extract_gsr_features_and_window_audit(
        raw_eda, tonic_eda, phasic_eda, eda_peaks, labels, subject_id, subj_out_dir
    )

    # 4. RSP Pipeline (700 Hz -> 0.05-0.35 Hz bandpass -> 100 Hz decimation -> breaths -> 180s windows)
    clean_resp, bad_mask = rsp_pipe.clean_raw_signal(raw_resp)
    filtered100_resp = rsp_pipe.filter_and_downsample_rsp(
        clean_resp, rsp_pipe.RAW_FS, rsp_pipe.DOWNSAMPLE_FACTOR
    )
    df_rsp, _, _, _ = rsp_pipe.process_respiration_windows(
        raw_resp, clean_resp, bad_mask, filtered100_resp, labels, subject_id, subj_out_dir
    )

    # 5. Merge all 3 modalities on window_index
    # Common metadata keys
    meta_cols = [
        "subject_id", "window_index", "window_start_seconds", "window_end_seconds",
        "wesad_condition_label", "label_purity_percent", "training_label_usable"
    ]
    
    # Extract modality-specific columns
    ppg_cols = ["paper_mean_ibi_ms", "paper_mean_pulse_amplitude", "ext_mean_heart_rate_bpm", "ext_sd_ibi_ms"]
    gsr_cols = ["paper_phasic_response_rate_per_sec", "paper_tonic_std_uS", "paper_tonic_slope_uS_per_s", "mean_tonic_uS"]
    rsp_cols = [
        "rsp_mean_amplitude", "rsp_std_amplitude", "rsp_spectral_entropy",
        "rsp_mean_peak_valley_magnitude", "rsp_std_peak_valley_magnitude", "rsp_breath_rate_bpm_qc"
    ]

    merged = df_ppg[meta_cols + ppg_cols].copy()
    for col in gsr_cols:
        merged[col] = df_gsr[col].values
    for col in rsp_cols:
        merged[col] = df_rsp[col].values

    # Quality counts
    merged["qc_kept_beats"] = df_ppg["qc_kept_ibi_count"].values
    merged["qc_phasic_peaks"] = df_gsr["qc_phasic_peak_count"].values
    merged["qc_breaths"] = df_rsp["n_breaths"].values

    # 6. Baseline Normalization (Bian et al. Section 2.3.2)
    # Find first pure baseline window
    baseline_mask = (merged["wesad_condition_label"] == "Baseline") & (merged["label_purity_percent"] >= 90.0)
    base_indices = merged.index[baseline_mask].tolist()

    if len(base_indices):
        ref_idx = base_indices[0]
        merged["is_baseline_reference"] = False
        merged.loc[ref_idx, "is_baseline_reference"] = True
        
        # Subtract reference baseline feature vector
        for col in PAPER_10_FEATURES:
            base_val = merged.loc[ref_idx, col]
            merged[f"norm_{col}"] = round(merged[col] - base_val, 5)
    else:
        merged["is_baseline_reference"] = False
        for col in PAPER_10_FEATURES:
            merged[f"norm_{col}"] = np.nan

    # Save subject multimodal CSV
    merged.to_csv(subj_out_dir / f"{subject_id}_multimodal_features.csv", index=False)
    
    # Free arrays from RAM
    del raw_bvp, raw_eda, raw_resp, labels, filtered_bvp, smoothed_eda, tonic_eda, phasic_eda, clean_resp, filtered100_resp
    gc.collect()

    return merged


def main():
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBJECTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(">>> BUILDING MASTER MULTIMODAL WESAD DATASET (10 PAPER FEATURES)")
    print("=" * 80)

    # Discover all subjects
    subject_dirs = sorted([d for d in WESAD_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    if not subject_dirs:
        raise FileNotFoundError(f"No subject folders found in: {WESAD_DIR}")

    print(f"Found {len(subject_dirs)} subjects: {[d.name for d in subject_dirs]}")

    master_list = []
    subject_summaries = []

    for i, s_dir in enumerate(subject_dirs, 1):
        s_id = s_dir.name
        pkl_path = s_dir / f"{s_id}.pkl"
        if not pkl_path.exists():
            print(f"[{i}/{len(subject_dirs)}] SKIP: {pkl_path} does not exist.")
            continue

        subj_out = SUBJECTS_OUT_DIR / s_id
        t0 = time.time()
        print(f"\n[{i}/{len(subject_dirs)}] Processing {s_id} from {pkl_path.name}...")
        
        try:
            df_subj = process_subject(pkl_path, s_id, subj_out)
            master_list.append(df_subj)
            elapsed = time.time() - t0
            
            n_win = len(df_subj)
            n_usable = df_subj["training_label_usable"].sum()
            has_base = df_subj["is_baseline_reference"].any()
            
            print(f"  -> Extracted {n_win} windows ({n_usable} training-usable, BaselineRef: {has_base}) in {elapsed:.1f}s")
            subject_summaries.append({
                "subject": s_id,
                "total_windows": n_win,
                "training_usable_windows": n_usable,
                "has_baseline_ref": has_base,
                "elapsed_seconds": round(elapsed, 1)
            })
        except Exception as e:
            print(f"  -> ERROR processing {s_id}: {e}")
            import traceback
            traceback.print_exc()

    # Combine into Master DataFrame
    master_df = pd.concat(master_list, ignore_index=True)
    master_csv_path = OUTPUT_DIR / "wesad_multimodal_master_180s.csv"
    master_df.to_csv(master_csv_path, index=False)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(">>> MASTER DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.2f} mins)")
    print(f"Total instances (rows): {len(master_df)}")
    print(f"Total features (columns): {len(master_df.columns)}")
    print(f"Saved master dataset to: {master_csv_path}")

    # Check for NaNs across the 10 paper features
    print("\n" + "=" * 80)
    print("10 PAPER FEATURES QUALITY AUDIT (NaN Check across all instances):")
    print("=" * 80)
    nan_counts = master_df[PAPER_10_FEATURES].isna().sum()
    for col, count in nan_counts.items():
        print(f"  {col:<36} : {count} NaNs")

    # Condition distribution summary
    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION (All Windows vs Training-Eligible Windows):")
    print("=" * 80)
    class_dist = master_df.groupby(["wesad_condition_label", "training_label_usable"]).size().unstack(fill_value=0)
    class_dist.columns = ["Excluded_or_Mixed", "Training_Usable"]
    class_dist["Total"] = class_dist["Excluded_or_Mixed"] + class_dist["Training_Usable"]
    print(class_dist.to_string())

    # Save summary report text file
    summary_text = [
        "WESAD MASTER MULTIMODAL DATASET SUMMARY",
        "========================================",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Subjects: {len(subject_summaries)}",
        f"Total Instances (180s windows): {len(master_df)}",
        f"Training-Usable Instances (purity >= 90%): {master_df['training_label_usable'].sum()}",
        f"Features: 10 Raw Paper Features + 10 Baseline-Normalized Features + Metadata",
        "\nPer-Subject Breakdown:",
        pd.DataFrame(subject_summaries).to_string(index=False),
        "\nClass Distribution:",
        class_dist.to_string(),
        "\n10 Paper Features NaN Audit:",
        nan_counts.to_string(),
    ]
    (OUTPUT_DIR / "master_dataset_summary.txt").write_text("\n".join(summary_text), encoding="utf-8")
    print(f"\nSaved summary audit to: {OUTPUT_DIR / 'master_dataset_summary.txt'}")


if __name__ == "__main__":
    main()
