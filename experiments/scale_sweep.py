"""
Sweep needle_near_scale to find the closeness threshold where recall rises.
"""
from __future__ import annotations
import os
import json
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from data.needles.run_niah import make_synthetic_stream, evaluate_recall
from core.clustering import OnlineManifoldClustering


def run_scale_sweep(scales: List[float], out_dir: str, total_tokens: int = 10000, dim: int = 64, sim_threshold: float = 0.8, max_centroids: int = 256):
    results = []
    keys_template = None
    values_template = None
    needles_template = None
    for s in scales:
        print(f"Running scale={s}")
        keys, values, needles = make_synthetic_stream(total=total_tokens, dim=dim, n_clusters=20, cluster_std=0.5, n_needles=50, seed=0, needle_near=True, needle_near_scale=s)
        cluster = OnlineManifoldClustering(dim=dim, window_size=4096, max_memory_size=65536, max_centroids=max_centroids, metric="cosine", similarity_threshold=sim_threshold)
        # ingest
        n = keys.shape[0]
        bs = 32
        for i in range(0, n, bs):
            cluster.add(keys[i : i + bs], values[i : i + bs])
        centroids, counts, weights = cluster.get_centroids()
        recall = evaluate_recall(centroids, needles, threshold=0.85)
        results.append({"scale": float(s), "num_centroids": int(centroids.shape[0]) if centroids is not None else 0, "recall": float(recall)})
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scale_sweep_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"scales": scales, "results": results}, f, indent=2)
    # plot
    xs = [r["scale"] for r in results]
    ys = [r["recall"] for r in results]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("needle_near_scale")
    plt.ylabel("recall")
    plt.title(f"Recall vs needle_near_scale (sim_th={sim_threshold}, max_centroids={max_centroids})")
    out_png = os.path.join(out_dir, "scale_sweep_recall.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote results to", out_path, "and", out_png)
    return out_path, out_png


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="experiments/scale_sweep_out")
    p.add_argument("--scales", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0])
    p.add_argument("--total-tokens", type=int, default=10000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--sim-threshold", type=float, default=0.8)
    p.add_argument("--max-centroids", type=int, default=256)
    args = p.parse_args()
    run_scale_sweep(args.scales, args.out, total_tokens=args.total_tokens, dim=args.dim, sim_threshold=args.sim_threshold, max_centroids=args.max_centroids)


