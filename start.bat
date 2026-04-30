@echo off
setlocal
cd /d "%~dp0"

set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%APPDATA%\uv\bin;%LOCALAPPDATA%\uv\bin;%LOCALAPPDATA%\Programs\uv;%PATH%"
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 'uv' was not found. Run install.bat first, then reopen this terminal.
    pause
    exit /b 1
)

uv run app.py

pause
