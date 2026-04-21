@echo off
setlocal enabledelayedexpansion

:: 1. Check if uv is already installed
where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] 'uv' is already installed.
) else (
    echo [INFO] 'uv' not detected. Starting installation via winget...
    
    :: 2. Install uv via winget
    winget install --id astral-sh.uv --silent --accept-source-agreements --accept-package-agreements
    
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install 'uv'. Please ensure winget is up to date.
        pause
        exit /b %errorlevel%
    )
    
    echo [SUCCESS] 'uv' installed successfully.
    
    :: 3. Refresh PATH for the current session
    :: Adding common uv installation paths to the local session environment
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    set "PATH=%APPDATA%\astral-sh\uv;%PATH%"
)

:: 4. Run uv sync
echo [INFO] Running 'uv sync'...
uv sync

if %errorlevel% equ 0 (
    echo [DONE] Process completed successfully.
) else (
    echo [WARNING] 'uv sync' failed. Make sure you are in a directory with a pyproject.toml file.
)

pause