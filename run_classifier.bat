@echo off
REM run_classifier_wsl.bat - Run the classifier using WSL
echo Starting Solder Joint Classifier via WSL...
echo.

REM Check if WSL is available
wsl --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: WSL is not installed or not available.
    echo Please install WSL2 from Microsoft Store or enable it in Windows Features.
    pause
    exit /b 1
)

REM Set the project directory path in WSL format
set WSL_PROJECT_PATH=/mnt/c/Users/GPAC/Desktop/project

echo Switching to project directory and running classifier...
echo Project path: %WSL_PROJECT_PATH%
echo.

REM Run the bash script through WSL
wsl bash -c "cd %WSL_PROJECT_PATH% && chmod +x run_classifier.sh && ./run_classifier.sh"

echo.
echo WSL execution completed.
pause