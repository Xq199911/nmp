#!/usr/bin/env python3
"""
Helper to persist a Hugging Face token for this project.

Usage:
  python scripts/save_hf_token.py --token <YOUR_TOKEN> [--persist-env]

By default this writes a file `.hf_token` in the repository root which will be
read by the project's scripts. If `--persist-env` is provided on Windows, the
script will also call `setx HUGGINGFACE_HUB_TOKEN <token>` to persist the token
into the user's environment variables (requires new shell to take effect).
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True, help="Hugging Face token")
    p.add_argument("--persist-env", action="store_true", help="Also persist into user environment (setx on Windows)")
    args = p.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    target = os.path.join(repo_root, ".hf_token")
    with open(target, "w", encoding="utf-8") as f:
        f.write(args.token.strip() + "\n")
    print(f"Wrote token to {target}")

    if args.persist_env:
        if sys.platform.startswith("win"):
            try:
                subprocess.check_call(["setx", "HUGGINGFACE_HUB_TOKEN", args.token.strip()])
                print("Persisted HUGGINGFACE_HUB_TOKEN to Windows user environment (setx). Restart your shell to pick it up.")
            except Exception as e:
                print("Failed to persist with setx:", e)
        else:
            print("Persisting to shell environment is platform-specific; set `HUGGINGFACE_HUB_TOKEN` in your shell profile manually.")

if __name__ == "__main__":
    main()


