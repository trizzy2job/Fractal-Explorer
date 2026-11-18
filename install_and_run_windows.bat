@echo off
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies. Confirm Python and pip are installed.
    pause
    exit /b 1
)
py cantor_explorer.py
if errorlevel 1 (
    echo The application closed with an error.
    pause
)
