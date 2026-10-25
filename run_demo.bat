@echo off
setlocal
cd /d "%~dp0"
title QAM Test Bench

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Tao moi truong .venv ...
  py -3.13 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo LOI: chua co Python 3.13. Cai tu python.org, tick "Add python.exe to PATH", roi chay lai file nay.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -c "import torch, streamlit, scipy, yaml, pandas, matplotlib" 2>nul
if errorlevel 1 (
  echo [2/3] Cai thu vien, lan dau mat khoang 3-10 phut ...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  if exist "wheels\" (
    ".venv\Scripts\python.exe" -m pip install --no-index --find-links wheels -r requirements-demo.txt
  ) else (
    ".venv\Scripts\python.exe" -m pip install -r requirements-demo.txt
  )
)
".venv\Scripts\python.exe" -c "import torch, streamlit, scipy, yaml, pandas, matplotlib" 2>nul
if errorlevel 1 (
  echo LOI: cai thu vien chua xong. Doc thong bao loi phia tren.
  pause
  exit /b 1
)

if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  mkdir "%USERPROFILE%\.streamlit" 2>nul
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo [3/3] Mo demo tai http://localhost:8501  -  dong cua so nay de tat demo
".venv\Scripts\python.exe" -m streamlit run demo\app.py --server.port 8501 --browser.gatherUsageStats false
pause
