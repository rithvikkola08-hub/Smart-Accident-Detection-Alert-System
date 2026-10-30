@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python generate_synthetic_dataset.py
if errorlevel 1 exit /b 1
python train_model.py
pause
