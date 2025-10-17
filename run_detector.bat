@echo off
REM Fake News Detector - Quick Launch Script for Windows

echo.
echo Fake News Detector - Quick Launch
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "notebooks\Fake.csv" (
    echo Error: Fake.csv not found in notebooks directory
    pause
    exit /b 1
)

if not exist "notebooks\True.csv" (
    echo Error: True.csv not found in notebooks directory
    pause
    exit /b 1
)

REM Install requirements if needed
if not exist "venv\" (
    echo Setting up virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Check if model exists
if not exist "fake_news_detector_svm.joblib" (
    echo.
    echo No trained model found. Training new model...
    echo.
    python fake_news_detector.py train
)

REM Launch GUI
echo.
echo Launching Fake News Detector GUI...
echo.
python fake_news_detector_gui.py

pause

