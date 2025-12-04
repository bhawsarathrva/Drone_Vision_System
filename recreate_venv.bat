@echo off
echo ======================================================================
echo Recreating Virtual Environment - This will fix all PyTorch issues
echo ======================================================================
echo.

echo [1/5] Backing up requirements...
copy requirements.txt requirements_backup.txt >nul 2>&1

echo [2/5] Removing old virtual environment...
rmdir /s /q venv 2>nul

echo [3/5] Creating fresh virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.10 or 3.11 is installed
    pause
    exit /b 1
)

echo [4/5] Installing PyTorch 2.5.1...
venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo [5/5] Installing other dependencies...
venv\Scripts\python.exe -m pip install ultralytics opencv-python loguru pyyaml roboflow geopy
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo SUCCESS! Virtual environment recreated successfully
echo ======================================================================
echo.
echo Verifying installation...
venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}')"
venv\Scripts\python.exe -c "from ultralytics import YOLO; print('YOLO: OK')"
echo.
echo You can now run training:
echo   venv\Scripts\python.exe train_pipeline.py --epochs 2 --model yolov8n.pt
echo.
pause
