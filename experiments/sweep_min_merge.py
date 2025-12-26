"""
Quick sweep over min_merge_similarity to observe effect on centroid counts and reconstruction loss.
Saves results to experiments/min_merge_out/.
"""
from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional
import numpy as np

from core.clustering import OnlineManifoldClustering
from analysis.energy_loss import assign_by_cosine, reconstruction_loss


def generate_synthetic_clusters(num_clusters: int, points_per_cluster: int, dim: int, cluster_std: float = 0.06, seed: int = 0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_clusters, dim)
    keys_list = []
    for c in centers:
        pts = c + rng.normal(scale=cluster_std, size=(points_per_cluster, dim))
        keys_list.append(pts.astype(np.float32))
    keys = np.vstack(keys_list)
    return keys


def sweep(min_merge_values: List[Optional[float]], sims: List[float], windows: List[int], seed: int = 0):
    keys = generate_synthetic_clusters(num_clusters=6, points_per_cluster=120, dim=32, cluster_std=0.06, seed=seed)
    results: List[Dict[str, Any]] = []
    for mm in min_merge_values:
        for sim in sims:
            for win in windows:
                cfg = {"min_merge_similarity": mm, "similarity_threshold": sim, "window_size": win}
                cl = OnlineManifoldClustering(dim=32, max_memory_size=10000, window_size=win, max_centroids=128, metric="cosine", similarity_threshold=sim, adaptive_threshold=False, min_merge_similarity=mm)
                cl.add(keys, np.zeros_like(keys))
                centroids, counts, weights = cl.get_centroids()
                C = centroids.shape[0]
                recon = None
                rand = None
                ratio = None
                if C > 0:
                    assign = assign_by_cosine(keys, centroids)
                    recon = reconstruction_loss(keys, assign, centroids)
                    rng = np.random.RandomState(seed + int(win) + int(sim * 100) + (0 if mm is None else int(mm * 100)))
                    pick_k = min(C, keys.shape[0])
                    idx = rng.choice(keys.shape[0], size=pick_k, replace=False)
                    random_centroids = keys[idx]
                    assign_r = assign_by_cosine(keys, random_centroids)
                    rand = reconstruction_loss(keys, assign_r, random_centroids)
                    ratio = float(recon / (rand + 1e-12))

                results.append({"config": cfg, "centroids": int(C), "recon_loss": recon, "random_loss": rand, "ratio": ratio})
                print(f"cfg={cfg} -> C={C} recon={recon} rand={rand} ratio={ratio}")

    out_dir = os.path.join("experiments", "min_merge_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "min_merge_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")
    return results


if __name__ == "__main__":
    mins = [None, 0.5, 0.6, 0.7, 0.8]
    sims = [0.6, 0.8]
    wins = [50, 200]
    sweep(mins, sims, wins)


