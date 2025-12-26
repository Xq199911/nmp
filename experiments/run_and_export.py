"""
Run a needles experiment and export plots/results using analysis.visualize.

Usage:
    python experiments/run_and_export.py --out experiments/out --total-tokens 20000
"""
from __future__ import annotations
import os
from data.needles.run_niah import parse_args, run_experiment
from analysis.visualize import generate_all_plots


def main(argv=None):
    args = parse_args(argv)
    # ensure out dir
    out_dir = args.out or "experiments/out"
    args.out = out_dir
    os.makedirs(out_dir, exist_ok=True)
    # run experiment (will write JSON and centroids.npy to out_dir)
    run_experiment(args)
    result_json = os.path.join(out_dir, "needles_result.json")
    plots = generate_all_plots(result_json, out_dir)
    print("Generated plots:", plots)


if __name__ == "__main__":
    main()


