"""
Ablation script: vary similarity_threshold and max_centroids and measure needles recall.
Saves CSV and plots to output directory.
"""
from __future__ import annotations
import os
import json
import argparse
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from data.needles.run_niah import make_synthetic_stream, evaluate_recall
from core.clustering import OnlineManifoldClustering


def run_single(keys, values, needles, similarity_threshold: float, max_centroids: int, args):
    cluster = OnlineManifoldClustering(
        dim=keys.shape[1],
        window_size=args.window_size,
        max_memory_size=args.max_memory_size,
        max_centroids=max_centroids,
        metric="cosine",
        similarity_threshold=similarity_threshold,
    )
    n = keys.shape[0]
    bs = args.batch_size
    for i in range(0, n, bs):
        cluster.add(keys[i : i + bs], values[i : i + bs])
    centroids, counts, weights = cluster.get_centroids()
    recall = evaluate_recall(centroids, needles, threshold=args.recall_threshold)
    return {"similarity_threshold": similarity_threshold, "max_centroids": max_centroids, "num_centroids": int(centroids.shape[0]) if centroids is not None else 0, "recall": float(recall)}


def run_grid(args):
    keys, values, needles = make_synthetic_stream(
        total=args.total_tokens,
        dim=args.dim,
        n_clusters=args.n_clusters,
        cluster_std=args.cluster_std,
        n_needles=args.n_needles,
        seed=args.seed,
        needle_near=getattr(args, "needle_near", False),
        needle_near_scale=getattr(args, "needle_near_scale", 0.5),
    )
    results = []
    for mc in args.max_centroids_list:
        for th in args.sim_thresholds:
            print(f"Running mc={mc} th={th}")
            r = run_single(keys, values, needles, similarity_threshold=th, max_centroids=mc, args=args)
            results.append(r)
    out = {"params": vars(args), "results": results}
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote ablation results to", out_path)
    return results


def plot_results(results, out_dir: str):
    # results: list of dicts with keys similarity_threshold, max_centroids, recall
    os.makedirs(out_dir, exist_ok=True)
    # group by max_centroids
    by_mc = {}
    for r in results:
        mc = r["max_centroids"]
        by_mc.setdefault(mc, []).append(r)
    plt.figure(figsize=(8, 5))
    for mc, items in sorted(by_mc.items()):
        items_sorted = sorted(items, key=lambda x: x["similarity_threshold"])
        xs = [it["similarity_threshold"] for it in items_sorted]
        ys = [it["recall"] for it in items_sorted]
        plt.plot(xs, ys, marker="o", label=f"max_centroids={mc}")
    plt.xlabel("similarity_threshold")
    plt.ylabel("recall")
    plt.title("Ablation: recall vs similarity_threshold")
    plt.legend()
    out_png = os.path.join(out_dir, "ablation_recall_vs_threshold.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved plot to", out_png)
    return out_png


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="experiments/ablation_out")
    p.add_argument("--total-tokens", type=int, default=10000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--cluster-std", type=float, default=0.5)
    p.add_argument("--n-needles", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window-size", type=int, default=4096)
    p.add_argument("--max-memory-size", type=int, default=65536)
    p.add_argument("--recall-threshold", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-thresholds", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    p.add_argument("--max-centroids-list", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--needle-near", dest="needle_near", action="store_true", help="Sample needles near cluster centers (easier to recover).")
    p.add_argument("--needle-near-scale", type=float, default=0.5, help="Scale for needle proximity when --needle-near is used.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # normalize lists
    args.sim_thresholds = list(args.sim_thresholds)
    args.max_centroids_list = list(args.max_centroids_list)
    results = run_grid(args)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()


