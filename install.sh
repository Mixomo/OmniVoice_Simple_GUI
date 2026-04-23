#!/bin/bash

# 1. Check if uv is already installed
if command -v uv &> /dev/null; then
    echo "[INFO] 'uv' is already installed."
else
    echo "[INFO] 'uv' not detected. Starting installation via curl..."
    
    # 2. Install uv via curl
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install 'uv'."
        exit 1
    fi
    
    echo "[SUCCESS] 'uv' installed successfully."
    
    # 3. Refresh PATH for the current session
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

# 4. Run uv sync
echo "[INFO] Running 'uv sync'..."
uv sync

if [ $? -eq 0 ]; then
    echo "[DONE] Process completed successfully."
else
    echo "[WARNING] 'uv sync' failed. Make sure you are in a directory with a pyproject.toml file."
fi
