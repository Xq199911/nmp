#!/usr/bin/env bash
echo "Create python venv and install minimal requirements..."
python -m venv .venv || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib umap-learn
echo "Done. Activate the virtualenv with: source .venv/bin/activate"


