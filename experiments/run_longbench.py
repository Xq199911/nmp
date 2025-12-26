"""
Orchestrator for LongBench / RULER style experiments (benchmark template).

This script provides two modes:
 - simulate: run a fast synthetic "needles" experiment using data.needles.run_niah
 - integrate: (placeholder) print recommended commands to run full LongBench/RULER suites

The aim is to provide a reproducible entrypoint for running large-scale benchmarks
and collecting MP-KVM ablation results. For real runs, supply a dataset and model
and adapt the `integration` section below to your environment.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import json
from typing import Optional


def run_simulate(args):
    # import the needles runner and call it programmatically
    try:
        from data.needles.run_niah import run_experiment, parse_args
    except Exception:
        # fallback to subprocess invocation
        script = os.path.join(os.path.dirname(__file__), "..", "data", "needles", "run_niah.py")
        cmd = ["python", script]
        if args.out is not None:
            cmd += ["--out", args.out]
        if args.total_tokens is not None:
            cmd += ["--total-tokens", str(args.total_tokens)]
        print("Running synthetic needles experiment via subprocess:", " ".join(cmd))
        subprocess.check_call(cmd)
        return

    # build synthetic args
    sim_args = parse_args(
        [
            "--total-tokens",
            str(args.total_tokens),
            "--dim",
            str(args.dim),
            "--n-clusters",
            str(args.n_clusters),
            "--n-needles",
            str(args.n_needles),
            "--out",
            args.out or "",
        ]
    )
    run_experiment(sim_args)


def print_integration_instructions(args):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    patch_script = os.path.join(repo_root, "scripts", "patch_llama_example.py")
    example_cmd = ["python", patch_script, "--model", args.model or "facebook/llama-3-8b", "--repeat", "5"]
    print("Integration mode is a placeholder. Recommended command to attach MP-KVM and collect centroids:")
    print(" ".join(example_cmd))
    print("")
    print("For LongBench/RULER integration you should:")
    print("- Provide a task runner that yields long-context prompts and references.")
    print("- Use `attach_mpkvm_to_hf_llama` (adapters.llama_adapter) before generation.")
    print("- After generation, call `mgr.get_layer_centroids(layer_idx)` to collect centroids and compute recall/energy metrics.")
    print("")
    print("This script can be extended to invoke your existing LongBench harness (add dataset/model paths and hooks).")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("simulate", "integrate"), default="simulate")
    p.add_argument("--out", type=str, default=None, help="Output directory for results")
    p.add_argument("--total-tokens", type=int, default=20000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--n-needles", type=int, default=50)
    p.add_argument("--model", type=str, default=None, help="Model id for integration mode")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out or "experiments/out", exist_ok=True)
    if args.mode == "simulate":
        run_simulate(args)
    else:
        print_integration_instructions(args)


if __name__ == "__main__":
    main()


