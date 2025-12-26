"""
Measure reconstruction loss (energy) of OnlineManifoldClustering vs random baselines
across varying target centroid counts. Saves results to experiments/energy_vs_random_out/.
"""
from __future__ import annotations
import os
import json
from typing import List, Dict, Any
import numpy as np

from core.clustering import OnlineManifoldClustering
from analysis.energy_loss import assign_by_cosine, reconstruction_loss


def generate_synthetic_clusters(num_clusters: int, points_per_cluster: int, dim: int, cluster_std: float = 0.06, seed: int = 123):
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_clusters, dim)
    keys_list = []
    for c in centers:
        pts = c + rng.normal(scale=cluster_std, size=(points_per_cluster, dim))
        keys_list.append(pts.astype(np.float32))
    keys = np.vstack(keys_list)
    return keys


def measure_energy_vs_random(target_counts: List[int], num_clusters: int = 8, points_per_cluster: int = 100, dim: int = 32, seed: int = 123):
    keys = generate_synthetic_clusters(num_clusters=num_clusters, points_per_cluster=points_per_cluster, dim=dim, cluster_std=0.06, seed=seed)
    results: List[Dict[str, Any]] = []
    N = keys.shape[0]
    for k in target_counts:
        k = max(1, int(k))
        # clustering run (set max_centroids = k)
        cl = OnlineManifoldClustering(dim=dim, max_memory_size=10000, window_size=50, max_centroids=k, metric="cosine", similarity_threshold=0.8)
        cl.add(keys, np.zeros_like(keys))
        centroids, counts, weights = cl.get_centroids()
        C = centroids.shape[0]
        if C == 0:
            recon = None
        else:
            assign = assign_by_cosine(keys, centroids)
            recon = reconstruction_loss(keys, assign, centroids)

        # random-centroid baseline: sample k points from keys
        rng = np.random.RandomState(seed + k)
        pick_k = min(max(1, k), keys.shape[0])
        idx = rng.choice(keys.shape[0], size=pick_k, replace=False)
        random_centroids = keys[idx]
        assign_r = assign_by_cosine(keys, random_centroids)
        rand_recon = reconstruction_loss(keys, assign_r, random_centroids)

        # random-deletion baseline: keep a random subset of size pick_k and use their weighted means per retained point (same as random_centroids here)
        # record results
        results.append({"target_k": k, "centroids_found": int(C), "recon_loss": recon, "random_centroid_loss": rand_recon})
        print(f"k={k} -> found={C} recon={recon} rand={rand_recon}")

    out_dir = os.path.join("experiments", "energy_vs_random_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "energy_vs_random_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")
    return results


if __name__ == "__main__":
    target_counts = [1, 2, 4, 8, 16, 32]
    measure_energy_vs_random(target_counts)


