#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install numpy matplotlib scikit-learn umap-learn pyyaml
# Install torch and transformers for real integration (CPU wheels by default).
# For GPU-enabled installs, users should follow PyTorch official instructions.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers
echo "PoC dependencies installed (numpy, matplotlib, scikit-learn, umap-learn, pyyaml, torch, transformers)."


