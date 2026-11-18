@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting PhysioSync Live Vernier GDX-RB Acquisition...
echo ========================================================
if exist "..\venv\Scripts\python.exe" (
    "..\venv\Scripts\python.exe" vernier_live.py
) else (
    python vernier_live.py
)
pause

