@echo off
REM Cattle Disease Detection System - Quick Start Script (Windows)
REM This script sets up and runs the application

echo.
echo Cattle Disease Detection System - Quick Start
echo ================================================
echo.

set "PYTHON_EXE="

REM Prefer bundled venv if present
if exist ".venv_run\Scripts\python.exe" (
    set "PYTHON_EXE=.venv_run\Scripts\python.exe"
)

REM Fall back to local venv
if not defined PYTHON_EXE if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
)

REM Fall back to system Python
if not defined PYTHON_EXE (
    where python >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
    )
)

if not defined PYTHON_EXE (
    echo ERROR: Python not found. Please install Python 3.8+ or use the bundled .venv_run.
    pause
    exit /b 1
)

echo Checking Python version...
"%PYTHON_EXE%" --version
echo.

REM Create virtual environment if needed and using system Python
if "%PYTHON_EXE%"=="python" (
    if not exist "venv\Scripts\python.exe" (
        echo Creating virtual environment...
        python -m venv venv
        echo Virtual environment created
    )
    set "PYTHON_EXE=venv\Scripts\python.exe"
)

REM Install dependencies
echo Installing dependencies...
"%PYTHON_EXE%" -m pip install --upgrade pip >nul 2>&1
"%PYTHON_EXE%" -m pip install -r requirements.txt
echo Dependencies installed
echo.

REM Check for model file
echo Checking for model file...
if exist "models\cattle_disease_vit_model.pth" (
    echo Model file found
) else (
    echo WARNING: Model file not found!
    echo Please add your trained model to models\cattle_disease_vit_model.pth
    echo See models\README.md for instructions
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)
echo.

REM Create necessary directories
echo Creating necessary directories...
if not exist "static\uploads" mkdir static\uploads
if not exist "models" mkdir models
echo Directories created
echo.

REM Check for .env file
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env >nul
    echo .env file created (please update with your values)
) else (
    echo .env file already exists
)
echo.

REM Run the application
echo Starting application...
echo ================================================
echo.
echo Application will be available at:
echo   http://localhost:5000
echo.
echo Admin credentials:
echo   Username: admin
echo   Password: admin123
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

"%PYTHON_EXE%" app.py

pause