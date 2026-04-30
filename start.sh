#!/bin/bash
cd "$(dirname "$0")" || exit 1

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:${XDG_BIN_HOME:-$HOME/.local/bin}:$PATH"
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] 'uv' was not found. Run bash install.sh first, then open a new terminal."
    exit 1
fi

uv run app.py
