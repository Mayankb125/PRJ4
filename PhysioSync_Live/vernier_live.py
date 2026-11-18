"""Vernier Go Direct Respiration Belt (GDX-RB) Live Data Acquisition.

Streams live respiration force (N) and breathing rate (bpm) over Bluetooth Low Energy (BLE),
displays live progress in the console, and continuously logs timestamped data to CSV.
"""
from datetime import datetime, timezone
from pathlib import Path
import csv
import sys
import time

# Ensure local gdx folder is discoverable
sys.path.insert(0, str(Path(__file__).parent))
from gdx import gdx

# ---------------- CONFIGURATION ----------------
SAMPLE_PERIOD_MS = 100  # 100 ms = 10 Hz acquisition rate
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = OUTPUT_DIR / f"respiration_{session_time}.csv"
# -----------------------------------------------


def main():
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    print("=" * 70, flush=True)
    print("   PhysioSync Live — Vernier GDX-RB BLE Data Acquisition", flush=True)
    print("=" * 70, flush=True)
    print("1. Ensure Vernier Graphical Analysis is closed.", flush=True)
    print("2. Ensure GDX-RB belt is powered on (Bluetooth LED flashing red).", flush=True)
    print("3. Ensure PC Bluetooth is turned on.", flush=True)
    print("=" * 70, flush=True)

    sensor = gdx.gdx()
    connected = False
    recording = False
    csv_file = None

    try:
        print("\n[1/3] Scanning for Vernier Go Direct BLE devices...", flush=True)
        sensor.open(connection="ble")
        
        # Verify device was actually found and opened
        if not sensor.devices:
            print("\n[!] Connection failed: No Go Direct BLE device was found.", flush=True)
            print("Troubleshooting steps:", flush=True)
            print("  1. Check that the GDX-RB is turned on and its LED is FLASHING RED.", flush=True)
            print("  2. If the LED is flashing GREEN, it is already connected to another app (like Graphical Analysis). Close it!", flush=True)
            print("  3. Check that Windows Bluetooth is switched ON in Windows Settings.", flush=True)
            print("  4. If you paired the device in Windows Bluetooth settings, click 'Remove device' (BLE pairing must be done by Python directly).", flush=True)
            return

        connected = True
        print("[✓] Connected to Go Direct device!", flush=True)

        # Print connected device information
        dev_info = sensor.device_info()
        if dev_info:
            print(f"    Device Name: {dev_info[0]}", flush=True)
            print(f"    Description: {dev_info[1]}", flush=True)
            print(f"    Battery:     {dev_info[2]}%", flush=True)

        print("\n[2/3] Configuring sensor channels...", flush=True)
        # Auto-select channels [1, 2] (Force & Respiration Rate) for GDX-RB
        try:
            sensor.select_sensors([1, 2])
        except (EOFError, Exception):
            sensor.select_sensors()

        # Retrieve selected sensor column headers (e.g. ['Force (N)', 'Respiration Rate (bpm)'])
        headers = sensor.enabled_sensor_info()
        if not headers:
            headers = ["Force (N)", "Respiration Rate (bpm)"]
        print(f"[✓] Active Channels: {headers}")

        # Initialize CSV writer
        csv_file = open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        
        # Header row
        csv_header = ["timestamp_utc", "elapsed_seconds"] + headers
        csv_writer.writerow(csv_header)
        csv_file.flush()
        print(f"[✓] Logging live data to: {OUTPUT_FILE}")

        # Start sensor sampling
        print(f"\n[3/3] Starting data stream at {1000 // SAMPLE_PERIOD_MS} Hz (period = {SAMPLE_PERIOD_MS} ms)...")
        print("Press Ctrl+C at any time to stop recording safely.\n")
        sensor.start(period=SAMPLE_PERIOD_MS)
        recording = True

        start_time = time.time()
        sample_count = 0

        while True:
            # sensor.read() returns a list of values for the enabled channels
            readings = sensor.read()

            if readings is not None and len(readings) > 0:
                sample_count += 1
                now_utc = datetime.now(timezone.utc).isoformat()
                elapsed = round(time.time() - start_time, 3)

                # Write row to CSV and flush immediately
                csv_writer.writerow([now_utc, elapsed] + readings)
                csv_file.flush()

                # Print progress update
                if sample_count % 10 == 0 or sample_count <= 5:
                    readings_str = ", ".join([f"{val:.2f}" if isinstance(val, (int, float)) else str(val) for val in readings])
                    print(f"Samples: {sample_count:5d} | Elapsed: {elapsed:6.1f}s | Readings: [{readings_str}]", flush=True)

            # Small sleep to keep pace with BLE sampling
            time.sleep(SAMPLE_PERIOD_MS / 1000.0)

    except KeyboardInterrupt:
        print("\n\n[!] Stop requested by user (Ctrl+C). Finishing recording...")

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")

    finally:
        if csv_file and not csv_file.closed:
            csv_file.flush()
            csv_file.close()
            print(f"[✓] Data safely written to: {OUTPUT_FILE}")

        if recording:
            try:
                sensor.stop()
                print("[✓] Sensor sampling stopped.")
            except Exception:
                pass

        if connected:
            try:
                sensor.close()
                print("[✓] BLE connection closed.")
            except Exception:
                pass

        print("\nSession complete. Data acquisition finished cleanly.")


if __name__ == "__main__":
    main()

