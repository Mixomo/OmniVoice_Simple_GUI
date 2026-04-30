@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

call :ensure_uv
if errorlevel 1 exit /b %errorlevel%

:: 4. Detect GPU and Patch Environment for Pascal/Older GPUs
echo [INFO] Detecting GPU to select optimal PyTorch version...
uv run python patch_env.py

:: 5. Run uv sync
echo [INFO] Running 'uv sync'...
uv sync

if %errorlevel% equ 0 (
    echo [DONE] Process completed successfully.
) else (
    echo [WARNING] 'uv sync' failed. Make sure you are in a directory with a pyproject.toml file.
)

exit /b 0

:refresh_uv_path
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%APPDATA%\uv\bin;%LOCALAPPDATA%\uv\bin;%LOCALAPPDATA%\Programs\uv;%PATH%"
exit /b 0

:ensure_uv
call :refresh_uv_path
where uv >nul 2>nul
if %errorlevel% equ 0 (
    for /f "delims=" %%V in ('uv --version 2^>nul') do echo [INFO] %%V
    exit /b 0
)

echo [INFO] 'uv' not detected. Starting installation...
where winget >nul 2>nul
if %errorlevel% equ 0 (
    winget install --id astral-sh.uv --exact --silent --accept-source-agreements --accept-package-agreements
) else (
    echo [WARNING] winget not found. Falling back to the official uv installer.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)

if %errorlevel% neq 0 (
    echo [WARNING] First uv install attempt failed. Trying the official uv installer...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install 'uv'. Please install it manually from https://astral.sh/uv/
    pause
    exit /b 1
)

call :refresh_uv_path
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 'uv' was installed but is still not visible in PATH for this session.
    echo [ERROR] Close this terminal, open a new one, and run install.bat again.
    pause
    exit /b 1
)

for /f "delims=" %%V in ('uv --version 2^>nul') do echo [SUCCESS] %%V is ready.
exit /b 0
