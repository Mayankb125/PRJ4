# PRJ4 — Multimodal Physiological Signal Processing & State Detection

A reproducible, quality-audited physiological signal processing and machine-learning feature extraction pipeline inspired by **Bian et al. (2019)** (*ACM Transactions on Accessible Computing*, DOI: [10.1145/3301498](https://doi.org/10.1145/3301498)) and evaluated on the **WESAD** multimodal dataset (Schmidt et al. 2018, DOI: [10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985)).

---

## 📌 Project Overview
This repository implements the end-to-end signal conditioning, quality filtering, and feature extraction for 3 physiological modalities:
1. **PPG / BVP** (Blood Volume Pulse from Empatica E4 wristband at 64 Hz)
2. **GSR / EDA** (Electrodermal Activity from Empatica E4 wristband at 4 Hz)
3. **Respiration (RSP)** (Chest Respiration from RespiBAN at 700 Hz, decimated to 100 Hz)

The pipeline extracts the **10 target features** per 180-second window described in Table 3 of Bian et al. (2019) across all 15 subjects (`S2` through `S17`), synchronizes ground-truth protocol labels (Baseline, Stress, Amusement, Meditation), and computes subject-specific **Baseline Normalization**.

---

## 🧬 Extracted Features (Bian et al. 2019 Table 3)

| # | Modality | Feature Name | Operational Definition / Unit |
|---|---|---|---|
| 1 | **PPG** | Mean of heart rate variability | Mean Inter-Beat Interval (IBI) in milliseconds (`paper_mean_ibi_ms`) |
| 2 | **PPG** | Mean amplitude of peak values | Peak-to-preceding-trough systolic pulse amplitude (`paper_mean_pulse_amplitude`) |
| 3 | **RSP** | Mean amplitude | Mean absolute amplitude of filtered breathing wave (`rsp_mean_amplitude`) |
| 4 | **RSP** | Standard deviation of amplitude | Standard deviation of filtered breathing waveform (`rsp_std_amplitude`) |
| 5 | **RSP** | Subband spectral entropy | Welch PSD normalized entropy in 0.05–0.35 Hz band (`rsp_spectral_entropy`) |
| 6 | **RSP** | Mean of peak–valley magnitude | Average breath depth (inhalation peak minus preceding trough) (`rsp_mean_peak_valley_magnitude`) |
| 7 | **RSP** | Std dev of peak–valley magnitude | Standard deviation of breath depths (`rsp_std_peak_valley_magnitude`) |
| 8 | **GSR** | Rate of phasic activity | Phasic SCR response peaks per second (`paper_phasic_response_rate_per_sec`) |
| 9 | **GSR** | Std dev of tonic activity level | Standard deviation of 0.05 Hz lowpass tonic SCL (`paper_tonic_std_uS`) |
| 10 | **GSR** | Slope of tonic activity | First-order linear regression slope in $\mu\text{S/s}$ (`paper_tonic_slope_uS_per_s`) |

---

## 📁 Repository Structure

```
PRJ4/
├── .gitignore                          # Excludes raw multi-gigabyte WESAD files and caches
├── README.md                           # Project documentation
├── ppg_wesad_pipeline_final.py         # PPG signal conditioning, peak audit & feature extraction
├── gsr_wesad_pipeline_final.py         # EDA tonic/phasic decomposition & peak extraction
├── rsp_wesad_pipeline_final.py         # Respiration 100 Hz decimation, breath audit & entropy
├── build_master_dataset.py             # Batch runner orchestrating all 15 subjects
└── master_dataset/
    ├── wesad_multimodal_master_180s.csv # Master dataset (476 rows x 35 columns, 0 NaNs)
    ├── master_dataset_summary.txt      # Class distributions and subject summaries
    └── subjects/                       # Subject-level multimodal CSVs (S2 - S17)
```

---

## 🚀 Quickstart

### 1. Requirements
```bash
pip install numpy pandas scipy matplotlib
```

### 2. Run Individual Modality Pipelines (e.g. for S2)
```bash
python ppg_wesad_pipeline_final.py
python gsr_wesad_pipeline_final.py
python rsp_wesad_pipeline_final.py
```

### 3. Generate Master Multimodal Dataset (All 15 Subjects)
```bash
python build_master_dataset.py
```
Outputs the unified master dataset at `master_dataset/wesad_multimodal_master_180s.csv`.

---

## 📊 Dataset Summary

* **Cohort**: 15 participants (`S2`–`S17`)
* **Total Instances**: **476 windows** (180 s non-overlapping)
* **Training-Eligible Windows** ($\ge 90\%$ label purity, non-transient): **194 pure windows**
  - Baseline: 89 windows
  - Stress: 43 windows
  - Meditation: 42 windows
  - Amusement: 20 windows
* **Data Integrity**: **0 missing values (zero NaNs)** across all 10 features for all 476 instances.
