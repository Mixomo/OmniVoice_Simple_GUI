#!/bin/bash
cd "$(dirname "$0")" || exit 1

refresh_uv_path() {
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:${XDG_BIN_HOME:-$HOME/.local/bin}:$PATH"
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
}

ensure_uv() {
    refresh_uv_path
    if command -v uv >/dev/null 2>&1; then
        echo "[INFO] $(uv --version)"
        return 0
    fi

    echo "[INFO] 'uv' not detected. Starting installation..."
    if ! command -v curl >/dev/null 2>&1; then
        echo "[ERROR] curl is required to install 'uv'. Please install curl and run this script again."
        exit 1
    fi

    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo "[ERROR] Failed to install 'uv'. Please install it manually from https://astral.sh/uv/"
        exit 1
    fi

    refresh_uv_path
    if ! command -v uv >/dev/null 2>&1; then
        echo "[ERROR] 'uv' was installed but is still not visible in PATH for this session."
        echo "[ERROR] Open a new terminal or source ~/.local/bin/env, then run install.sh again."
        exit 1
    fi

    echo "[SUCCESS] $(uv --version) is ready."
}

ensure_uv

# 4. Detect GPU and Patch Environment for Pascal/Older GPUs
echo "[INFO] Detecting GPU to select optimal PyTorch version..."
uv run python3 patch_env.py

# 5. Run uv sync
echo "[INFO] Running 'uv sync'..."
uv sync

if [ $? -eq 0 ]; then
    echo "[DONE] Process completed successfully."
else
    echo "[WARNING] 'uv sync' failed. Make sure you are in a directory with a pyproject.toml file."
fi
