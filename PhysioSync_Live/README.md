# PhysioSync Live — Vernier GDX-RB Live Data Acquisition

This module provides real-time Bluetooth Low Energy (BLE) data streaming from the **Vernier Go Direct Respiration Belt (GDX-RB)** into Python and logs timestamped readings continuously to CSV.

---

## 📁 Folder Structure

```
PhysioSync_Live/
├── gdx/                    # Official Vernier Go Direct BLE driver module
│   ├── __init__.py
│   ├── gdx.py
│   └── readme.md
├── data/                   # Live recorded CSV output directory
│   └── respiration_YYYYMMDD_HHMMSS.csv
├── vernier_live.py         # Main real-time acquisition script
├── run_acquisition.bat     # One-click launcher script
└── README.md
```

---

## 🚀 How to Run

### Step 1: Prepare Sensor & PC
1. **Close** Vernier Graphical Analysis completely.
2. **Turn on** the GDX-RB belt (the Bluetooth LED will flash red).
3. Ensure your computer's **Bluetooth is ON**.
   *(Do **not** pair it in Windows Bluetooth settings; the script pairs via BLE automatically).*

### Step 2: Run the Script
Open your terminal in `PhysioSync_Live` and run:

```bash
.\run_acquisition.bat
```

Or run directly with Python:
```bash
python vernier_live.py
```

---

## 📊 Recorded CSV Output Format

Files are saved automatically in `data/respiration_<timestamp>.csv` with the following columns:

| timestamp_utc | elapsed_seconds | Force (N) | Respiration Rate (bpm) |
| :--- | :--- | :--- | :--- |
| 2026-11-17T06:50:00.100Z | 0.1 | 12.45 | 14.0 |
| 2026-11-17T06:50:00.200Z | 0.2 | 12.80 | 14.0 |
| 2026-11-17T06:50:00.300Z | 0.3 | 13.15 | 14.0 |

Press **`Ctrl+C`** at any time to safely stop recording and finalize the file.

