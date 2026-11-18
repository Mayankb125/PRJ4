"""Deep Physiological Feature Extraction & Audit on Collected Live GDX-RB Data."""
import csv
import math
from pathlib import Path

csv_path = Path(__file__).parent / "data" / "respiration_20261117_123101.csv"

if not csv_path.is_file():
    print(f"File not found: {csv_path}")
    exit(1)

with open(csv_path, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

times = [float(r["elapsed_seconds"]) for r in rows if r["elapsed_seconds"] != ""]
forces = [float(r["Force (N)"]) for r in rows if r["Force (N)"] != ""]

n_total = len(forces)
total_dur = times[-1] - times[0]


def analyze_window(w_times, w_forces, label):
    n = len(w_forces)
    dur = w_times[-1] - w_times[0]
    fs = n / dur if dur > 0 else 0

    # 1. Mean & Std amplitude of breathing signal (detrended / zero-mean)
    mean_val = sum(w_forces) / n
    centered = [x - mean_val for x in w_forces]
    std_val = math.sqrt(sum(x ** 2 for x in centered) / n)
    mean_abs_amp = sum(abs(x) for x in centered) / n

    # 2. Moving average smoothing (window = 5 points)
    smooth = []
    w = 5
    for i in range(n):
        s_idx = max(0, i - w)
        e_idx = min(n, i + w + 1)
        smooth.append(sum(w_forces[s_idx:e_idx]) / (e_idx - s_idx))

    # 3. Peak (inhalation) and Valley (exhalation) detection
    peaks = []
    valleys = []

    for i in range(1, n - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
            if smooth[i] > mean_val:
                if not peaks or (w_times[i] - w_times[peaks[-1]]) >= 2.0:
                    peaks.append(i)
        elif smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1]:
            if smooth[i] < mean_val:
                if not valleys or (w_times[i] - w_times[valleys[-1]]) >= 2.0:
                    valleys.append(i)

    # 4. Peak-valley magnitudes (breath depths)
    pv_magnitudes = []
    for p in peaks:
        pre_valleys = [v for v in valleys if v < p]
        if pre_valleys:
            closest_v = pre_valleys[-1]
            pv_magnitudes.append(smooth[p] - smooth[closest_v])

    mean_pv = sum(pv_magnitudes) / len(pv_magnitudes) if pv_magnitudes else 0.0
    std_pv = (
        math.sqrt(sum((x - mean_pv) ** 2 for x in pv_magnitudes) / len(pv_magnitudes))
        if pv_magnitudes
        else 0.0
    )

    # 5. Breaths per minute
    if len(peaks) > 1:
        peak_intervals = [w_times[peaks[j]] - w_times[peaks[j - 1]] for j in range(1, len(peaks))]
        avg_inter = sum(peak_intervals) / len(peak_intervals)
        bpm = 60.0 / avg_inter
    else:
        bpm = 0.0

    # 6. Discrete Fourier Transform in respiration subband [0.05, 0.35 Hz] for spectral entropy
    step = max(1, int(round(fs / 2.0)))
    sub_centered = centered[::step]
    sub_fs = fs / step
    N_sub = len(sub_centered)

    freqs = []
    psd = []
    for k in range(5, 36):
        f = k * 0.01
        omega = 2.0 * math.pi * f / sub_fs
        real_part = sum(sub_centered[t] * math.cos(omega * t) for t in range(N_sub))
        imag_part = sum(sub_centered[t] * math.sin(omega * t) for t in range(N_sub))
        power = (real_part ** 2 + imag_part ** 2) / N_sub
        freqs.append(f)
        psd.append(power)

    total_p = sum(psd)
    if total_p > 0:
        norm_psd = [p / total_p for p in psd]
        spec_entropy = -sum(p * math.log(p) for p in norm_psd if p > 0) / math.log(len(norm_psd))
    else:
        spec_entropy = 0.0

    return {
        "label": label,
        "duration_s": dur,
        "samples": n,
        "fs": fs,
        "mean_amp": mean_abs_amp,
        "std_amp": std_val,
        "spec_entropy": spec_entropy,
        "mean_pv_mag": mean_pv,
        "std_pv_mag": std_pv,
        "n_breaths": len(peaks),
        "bpm": bpm,
        "min_force": min(w_forces),
        "max_force": max(w_forces),
    }


# Divide into 180s windows (matching the 3-min model design)
w1_end_t = times[0] + 180.0
w2_end_t = times[0] + 360.0

idx_w1 = [i for i, t in enumerate(times) if t <= w1_end_t]
idx_w2 = [i for i, t in enumerate(times) if w1_end_t < t <= w2_end_t]
idx_w3 = [i for i, t in enumerate(times) if t > w2_end_t]

res_all = analyze_window(times, forces, "Full Session")
res_w1 = analyze_window([times[i] for i in idx_w1], [forces[i] for i in idx_w1], "Window 1 (0-180s)")
res_w2 = analyze_window([times[i] for i in idx_w2], [forces[i] for i in idx_w2], "Window 2 (180-360s)")
res_w3 = (
    analyze_window([times[i] for i in idx_w3], [forces[i] for i in idx_w3], "Window 3 (360s-End)")
    if len(idx_w3) > 100
    else None
)

print("=" * 90)
print("     DEEP PHYSIOLOGICAL AUDIT OF COLLECTED VERNIER GDX-RB RESPIRATION DATA")
print("=" * 90)
print(f"Total Session Time: {total_dur:.1f} s ({total_dur/60:.2f} mins) | Total Data Rows: {n_total:,}")
print(f"Active Windows:     {2 if not res_w3 else 3} complete/ongoing 180s windows")
print("-" * 90)
header_fmt = "{:<36} | {:<16} | {:<16} | {:<16}"
print(header_fmt.format("Bian et al. / WESAD Feature", "Window 1 (Base)", "Window 2 (Active)", "Full Session"))
print("-" * 90)

rows_to_print = [
    ("Breathing Rate (BPM)", f"{res_w1['bpm']:.1f} bpm", f"{res_w2['bpm']:.1f} bpm", f"{res_all['bpm']:.1f} bpm"),
    ("Detected Breaths", f"{res_w1['n_breaths']} breaths", f"{res_w2['n_breaths']} breaths", f"{res_all['n_breaths']} breaths"),
    ("rsp_mean_amplitude", f"{res_w1['mean_amp']:.3f} N", f"{res_w2['mean_amp']:.3f} N", f"{res_all['mean_amp']:.3f} N"),
    ("rsp_std_amplitude", f"{res_w1['std_amp']:.3f} N", f"{res_w2['std_amp']:.3f} N", f"{res_all['std_amp']:.3f} N"),
    ("rsp_mean_peak_valley_magnitude", f"{res_w1['mean_pv_mag']:.3f} N", f"{res_w2['mean_pv_mag']:.3f} N", f"{res_all['mean_pv_mag']:.3f} N"),
    ("rsp_std_peak_valley_magnitude", f"{res_w1['std_pv_mag']:.3f} N", f"{res_w2['std_pv_mag']:.3f} N", f"{res_all['std_pv_mag']:.3f} N"),
    ("rsp_spectral_entropy (0.05-0.35 Hz)", f"{res_w1['spec_entropy']:.3f}", f"{res_w2['spec_entropy']:.3f}", f"{res_all['spec_entropy']:.3f}"),
    ("Force Range (Min -> Max)", f"{res_w1['min_force']:.1f} -> {res_w1['max_force']:.1f} N", f"{res_w2['min_force']:.1f} -> {res_w2['max_force']:.1f} N", f"{res_all['min_force']:.1f} -> {res_all['max_force']:.1f} N"),
]

for title, w1, w2, wall in rows_to_print:
    print(header_fmt.format(title, w1, w2, wall))

print("=" * 90)

