@echo off
title PRJ4 Workstation Launcher
echo ======================================================================
echo          Starting PRJ4 Affective Computing Workstation
echo ======================================================================
echo.

cd /d D:\PRJ

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at D:\PRJ\venv\
    echo Please make sure your venv is installed.
    pause
    exit /b 1
)

echo [1/3] Starting Python FastAPI Backend Bridge (Port 8000 + Unity TCP 5005)...
start "PRJ4 Backend Bridge (:8000)" cmd /k "cd /d D:\PRJ && .\venv\Scripts\python.exe -m uvicorn api_bridge:app --host 127.0.0.1 --port 8000"

echo [2/3] Starting React Vite Frontend (Port 5173)...
start "PRJ4 Frontend (:5173)" cmd /k "cd /d D:\PRJ\frontend && npm run dev"

echo [3/3] Waiting for servers to initialize...
timeout /t 3 /nobreak >nul

echo Opening workstation in default web browser...
start http://localhost:5173

echo.
echo ======================================================================
echo Workstation is running!
echo   * Web UI:        http://localhost:5173
echo   * API Bridge:    http://127.0.0.1:8000
echo   * Unity TCP:     127.0.0.1:5005
echo.
echo To stop the workstation, close the two terminal windows.
echo ======================================================================
echo.
pause
