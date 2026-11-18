@echo off
title PRJ4 Streamlit Dashboard
echo Starting PRJ4 Streamlit Interface on http://localhost:8501...
cd /d D:\PRJ
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501
pause
