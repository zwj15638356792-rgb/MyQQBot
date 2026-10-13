@echo off
setlocal
cd /d "%~dp0" || exit /b 1
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0start_bot.py"
    exit /b
)
if exist "D:\Anaconda\pythonw.exe" (
    start "" "D:\Anaconda\pythonw.exe" "%~dp0start_bot.py"
    exit /b
)
python "%~dp0start_bot.py"
if errorlevel 1 pause
