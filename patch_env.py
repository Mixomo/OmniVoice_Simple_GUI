import subprocess
import re
import sys
import os

def main():
    pyproject_path = "pyproject.toml"
    if not os.path.exists(pyproject_path):
        print("[WARNING] pyproject.toml not found.")
        return

    try:
        # Query nvidia-smi for compute capability
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], stderr=subprocess.STDOUT).decode()
        caps = [float(x.strip()) for x in out.splitlines() if x.strip()]
        
        if not caps:
            print("[INFO] No NVIDIA GPU detected via nvidia-smi. Defaulting to modern PyTorch.")
            is_old_gpu = False
        else:
            # Pascal is 6.1 (GTX 10 series), Maxwell is 5.x, Kepler is 3.x
            # 6.1 and below require older PyTorch (<=2.6.0) for official support in some wheels
            is_old_gpu = any(c <= 6.1 for c in caps)
            print(f"[INFO] Detected GPU Compute Capabilities: {caps}")
    except Exception as e:
        print(f"[WARNING] nvidia-smi failed or not found ({e}). Defaulting to modern PyTorch.")
        is_old_gpu = False

    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()

    if is_old_gpu:
        print("[INFO] Applying Pascal/Older GPU patch -> PyTorch 2.6.0+cu126")
        content = re.sub(r'torch==2\.7\.1', 'torch==2.6.0', content)
        content = re.sub(r'torchaudio==2\.7\.1', 'torchaudio==2.6.0', content)
        content = re.sub(r'torchvision==0\.22\.1', 'torchvision==0.21.0', content)
        content = re.sub(r'cu128', 'cu126', content)
    else:
        print("[INFO] Applying Modern GPU config -> PyTorch 2.7.1+cu128")
        content = re.sub(r'torch==2\.6\.0', 'torch==2.7.1', content)
        content = re.sub(r'torchaudio==2\.6\.0', 'torchaudio==2.7.1', content)
        content = re.sub(r'torchvision==0\.21\.0', 'torchvision==0.22.1', content)
        content = re.sub(r'cu126', 'cu128', content)

    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    main()
