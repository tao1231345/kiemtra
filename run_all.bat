@echo off
setlocal

set "PY_CMD="
py -3 --version >nul 2>nul
if %errorlevel%==0 set "PY_CMD=py -3"
if not defined PY_CMD (
  python --version >nul 2>nul
  if %errorlevel%==0 set "PY_CMD=python"
)
if not defined PY_CMD (
  echo Python is not installed or not in PATH.
  echo Install Python from https://www.python.org/downloads/windows/
  echo During install, tick: Add Python to PATH
  echo Then open a new terminal and run: .\run_all.bat
  exit /b 1
)

echo [1/3] Installing requirements...
%PY_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements.
  exit /b 1
)

echo [2/3] Running data analysis + training pipeline...
%PY_CMD% auto_mobile_price_pipeline.py --dataset "PromptCloudHQ/flipkart-products"
if errorlevel 1 (
  echo Pipeline failed.
  exit /b 1
)

echo [3/3] Running sample prediction...
%PY_CMD% predict_sample.py --input-json sample_input.json
if errorlevel 1 (
  echo Sample prediction failed.
  exit /b 1
)

echo Done. Check artifacts folder for outputs.
exit /b 0
