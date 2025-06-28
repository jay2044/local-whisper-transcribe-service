@echo off
echo ========================================
echo Local Whisper Transcriber - Build Tool
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found. Installing build dependencies...
pip install -r build_requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install build dependencies
    pause
    exit /b 1
)

echo.
echo Starting build process...
echo This may take 10-30 minutes depending on your system.
echo.

python build_script.py

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Your application is ready in: dist\LocalWhisperTranscriber\
echo.
echo To distribute:
echo 1. Zip the dist\LocalWhisperTranscriber\ folder
echo 2. Include the install.bat file
echo 3. Share with users
echo.
pause 