"""
Inspect max cosine similarities between needles and centroids for a single config.
"""
from __future__ import annotations
import numpy as np
from data.needles.run_niah import make_synthetic_stream
from core.clustering import OnlineManifoldClustering
import argparse
import json
import os


def compute_stats(similarity_threshold: float, max_centroids: int, total_tokens: int = 10000, dim: int = 64, seed: int = 0):
    keys, values, needles = make_synthetic_stream(total=total_tokens, dim=dim, n_clusters=20, cluster_std=0.5, n_needles=50, seed=seed, needle_near=False, needle_near_scale=0.5)
    cluster = OnlineManifoldClustering(dim=dim, window_size=4096, max_memory_size=65536, max_centroids=max_centroids, metric="cosine", similarity_threshold=similarity_threshold)
    for i in range(0, keys.shape[0], 32):
        cluster.add(keys[i : i + 32], values[i : i + 32])
    centroids, counts, weights = cluster.get_centroids()
    if centroids is None or centroids.shape[0] == 0:
        print(json.dumps({"num_centroids": 0}))
        return
    # normalize
    def _norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n
    cn = _norm(centroids)
    needles_n = _norm(needles)
    sims = np.dot(needles_n, cn.T)
    max_sims = sims.max(axis=1)
    stats = {
        "num_centroids": int(centroids.shape[0]),
        "max_of_max": float(max_sims.max()),
        "mean_max": float(max_sims.mean()),
        "median_max": float(np.median(max_sims)),
        "counts_above_085": int((max_sims >= 0.85).sum()),
        "counts_above_07": int((max_sims >= 0.7).sum()),
        "counts_above_06": int((max_sims >= 0.6).sum()),
        "per_needle_max": [float(x) for x in max_sims],
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sim-th", type=float, default=0.5)
    p.add_argument("--max-centroids", type=int, default=2048)
    p.add_argument("--total-tokens", type=int, default=10000)
    args = p.parse_args()
    compute_stats(similarity_threshold=args.sim_th, max_centroids=args.max_centroids, total_tokens=args.total_tokens)


