"""
RSP (respiration) pipeline for WESAD, following Bian et al. 2019 (VDEAR), Sec 2.2-2.3.

Steps (paper -> here):
  1. outlier removal (spikes replaced by neighbouring values)  -> remove_outliers()
  2. notch filter at mains frequency                            -> notch()
  3. band-pass 0.05-0.35 Hz (high-pass 0.05, low-pass 0.35)     -> bandpass()
  4. down-sample to 100 Hz                                      -> downsample()
  5. 3-min windows, 1-min step                                  -> window_features()
  6. sub-band spectral entropy (0.05-0.35 Hz)                   -> spectral_entropy()
  7. baseline subtraction (3-min calm baseline)                 -> in run_subject()

Usage:
  python rsp_pipeline.py --data C:/PRJ-4/WESAD/WESAD --out C:/PRJ-4/rsp_pipeline/output
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

FS_RAW = 700          # RespiBAN sampling rate
FS_OUT = 100          # paper down-samples to 100 Hz
BAND = (0.05, 0.35)   # paper's RSP band-pass
MAINS_HZ = 50.0       # WESAD recorded in Germany; the paper used 60 Hz (US). Configurable.
WIN_S, STEP_S = 180, 60
LABELS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def load_subject(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    resp = np.asarray(d["signal"]["chest"]["Resp"]).ravel().astype(float)  # already in % (see readme)
    label = np.asarray(d["label"]).ravel().astype(int)
    return resp, label


def remove_outliers(x, fs, k=4.0):
    """Replace spikes by interpolation of adjacent good samples (paper: mean of adjacent points).
    A sample is a spike if it deviates from a 0.2 s rolling median by more than k x the 99th
    percentile of such deviations (a robust scale; MAD is ~0 on this quantised signal)."""
    win = int(0.2 * fs) | 1
    med = pd.Series(x).rolling(win, center=True, min_periods=1).median().to_numpy()
    dev = np.abs(x - med)
    bad = dev > k * np.percentile(dev, 99)
    y = x.copy()
    if bad.any():
        idx = np.arange(len(x))
        y[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return y, bad.mean()


def notch(x, fs, f0=MAINS_HZ, q=30.0):
    b, a = signal.iirnotch(f0, q, fs)
    return signal.filtfilt(b, a, x)


def bandpass(x, fs, band=BAND, order=2):
    sos = signal.butter(order, band, btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def downsample(x, fs_in, fs_out):
    return signal.decimate(x, fs_in // fs_out, ftype="fir", zero_phase=True)


def spectral_entropy(x, fs, band=BAND, nperseg_s=60):
    """Normalised Shannon entropy (0..1) of the Welch PSD restricted to `band`."""
    f, p = signal.welch(x, fs, nperseg=int(nperseg_s * fs), noverlap=int(nperseg_s * fs / 2))
    m = (f >= band[0]) & (f <= band[1])
    p = p[m]
    p = p / p.sum()
    return -np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p))


def peak_valley_magnitudes(x, fs):
    """Magnitude of each breath = peak value minus the valley that precedes it.
    Peaks/valleys must be >= 2 s apart (band-pass caps breathing at 0.35 Hz = 2.9 s/cycle)."""
    dist = int(2 * fs)
    sd = np.std(x)
    pk, _ = signal.find_peaks(x, distance=dist, prominence=0.1 * sd)
    vl, _ = signal.find_peaks(-x, distance=dist, prominence=0.1 * sd)
    mags = []
    for p in pk:
        prev = vl[vl < p]
        if len(prev):
            mags.append(x[p] - x[prev[-1]])
    return np.asarray(mags)


FEATURES = ["rsp_mean_amplitude", "rsp_std_amplitude", "rsp_spec_entropy",
            "rsp_mean_peak_valley", "rsp_std_peak_valley"]  # paper Table 3, RSP rows


def extract_features(seg, fs):
    """The 5 RSP features of Bian et al. 2019, Table 3, for one window of filtered RSP."""
    amp = np.abs(seg)  # amplitude = magnitude of the (zero-mean) filtered signal
    pv = peak_valley_magnitudes(seg, fs)
    return dict(rsp_mean_amplitude=amp.mean(), rsp_std_amplitude=amp.std(),
                rsp_spec_entropy=spectral_entropy(seg, fs),
                rsp_mean_peak_valley=pv.mean() if len(pv) else np.nan,
                rsp_std_peak_valley=pv.std() if len(pv) else np.nan,
                n_breaths=len(pv))


def window_features(x, lab, fs):
    rows = []
    w, s = WIN_S * fs, STEP_S * fs
    for start in range(0, len(x) - w + 1, s):
        seg_lab = lab[start:start + w]
        if len(np.unique(seg_lab)) != 1 or seg_lab[0] not in LABELS:
            continue  # keep only windows lying fully inside one protocol condition
        rows.append(dict(t_start_s=start / fs, t_end_s=(start + w) / fs, label=int(seg_lab[0]),
                         condition=LABELS[int(seg_lab[0])], **extract_features(x[start:start + w], fs)))
    return pd.DataFrame(rows)


def run_subject(pkl_path, out_dir, mains=MAINS_HZ):
    sid = pkl_path.stem
    raw, lab = load_subject(pkl_path)

    # 1-3 at native rate
    clean, frac_out = remove_outliers(raw, FS_RAW)
    notched = notch(clean, FS_RAW, mains)
    filt = bandpass(notched, FS_RAW)
    # 4 down-sample
    r_ds = downsample(raw, FS_RAW, FS_OUT)
    f_ds = downsample(filt, FS_RAW, FS_OUT)
    l_ds = lab[::FS_RAW // FS_OUT][:len(f_ds)]
    n = min(len(f_ds), len(l_ds), len(r_ds))
    r_ds, f_ds, l_ds = r_ds[:n], f_ds[:n], l_ds[:n]

    # 5-6 windowed features
    feats = window_features(f_ds, l_ds, FS_OUT)

    # 7 baseline: first 3 min of the baseline condition; subtract its feature values
    b_idx = np.flatnonzero(l_ds == 1)
    if len(b_idx) >= WIN_S * FS_OUT and len(feats):
        b0 = b_idx[0]
        base = extract_features(f_ds[b0:b0 + WIN_S * FS_OUT], FS_OUT)
        for k in FEATURES:
            feats[k + "_bc"] = feats[k] - base[k]
        # windows overlapping the reference are flagged so they can be dropped in modelling
        feats["is_baseline_ref"] = (feats.t_start_s * FS_OUT < b0 + WIN_S * FS_OUT) &                                    (feats.t_end_s * FS_OUT > b0)
    feats.insert(0, "subject", sid)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(time_s=np.arange(n) / FS_OUT, rsp_raw_pct=r_ds, rsp_filtered=f_ds, label=l_ds)) \
        .to_csv(out_dir / f"{sid}_rsp_filtered_100Hz.csv", index=False, float_format="%.6f")
    feats.to_csv(out_dir / f"{sid}_rsp_features.csv", index=False, float_format="%.6f")
    plot_subject(sid, raw, notched, filt, lab, f_ds, l_ds, feats, out_dir)
    return feats, dict(subject=sid, outlier_frac=frac_out, minutes=len(raw) / FS_RAW / 60)


def plot_subject(sid, raw, notched, filt, lab, f_ds, l_ds, feats, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(13, 10))
    # (a) raw vs filtered, 2 minutes inside the baseline segment
    b0 = np.flatnonzero(lab == 1)[0] + 60 * FS_RAW
    sl = slice(b0, b0 + 120 * FS_RAW)
    t = np.arange(sl.stop - sl.start) / FS_RAW
    ax[0].plot(t, raw[sl] - raw[sl].mean(), lw=.7, label="raw (mean removed)", alpha=.7)
    ax[0].plot(t, filt[sl], lw=1.2, label="filtered 0.05-0.35 Hz")
    ax[0].set(title=f"{sid}: raw vs filtered RSP (2 min, baseline)", xlabel="s", ylabel="% (belt)")
    ax[0].legend(loc="upper right")
    # (b) PSD before/after over the whole recording
    f1, p1 = signal.welch(raw - raw.mean(), FS_RAW, nperseg=FS_RAW * 120)
    f2, p2 = signal.welch(filt, FS_RAW, nperseg=FS_RAW * 120)
    ax[1].semilogy(f1, p1, label="raw"); ax[1].semilogy(f2, p2, label="filtered")
    ax[1].axvspan(*BAND, color="g", alpha=.1)
    ax[1].set(xlim=(0, 2), ylim=(1e-8, None), title="Power spectrum (green = pass band)", xlabel="Hz", ylabel="PSD")
    ax[1].legend()
    # (c) whole session with protocol conditions + spectral entropy
    t2 = np.arange(len(f_ds)) / FS_OUT / 60
    ax[2].plot(t2, f_ds, lw=.5, color="k")
    colors = {1: "tab:blue", 2: "tab:red", 3: "tab:orange", 4: "tab:green"}
    for k, c in colors.items():
        m = l_ds == k
        if m.any():
            ax[2].fill_between(t2, f_ds.min(), f_ds.max(), where=m, color=c, alpha=.15, label=LABELS[k])
    if len(feats):
        ax2 = ax[2].twinx()
        ax2.plot(feats.t_end_s / 60, feats.rsp_spec_entropy, "o", color="m", ms=3, label="spectral entropy")
        ax2.set_ylabel("spectral entropy (0-1)")
    ax[2].set(title="Filtered RSP over the session, protocol conditions, window entropy",
              xlabel="min", ylabel="filtered RSP")
    ax[2].legend(loc="upper left", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sid}_rsp_overview.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:/PRJ-4/WESAD/WESAD")
    ap.add_argument("--out", default=r"C:/PRJ-4/rsp_pipeline/output")
    ap.add_argument("--mains", type=float, default=MAINS_HZ)
    ap.add_argument("--subjects", nargs="*", help="e.g. S10 S11 (default: all found)")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    pkls = sorted(data.glob("S*/S*.pkl"), key=lambda p: int(p.stem[1:]))
    if a.subjects:
        pkls = [p for p in pkls if p.stem in a.subjects]
    allf, summ = [], []
    for p in pkls:
        try:
            f, s = run_subject(p, out, a.mains)
        except Exception as e:  # keep going if one subject is still being extracted/corrupt
            print(f"[skip] {p.stem}: {e!r}")
            continue
        allf.append(f); summ.append(s)
        print(f"{p.stem}: {s['minutes']:.1f} min, outliers {s['outlier_frac']*100:.2f}%, {len(f)} windows")
    if allf:
        pd.concat(allf).to_csv(out / "all_subjects_rsp_features.csv", index=False, float_format="%.6f")
        pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
