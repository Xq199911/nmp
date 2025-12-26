"""
Sweep init_preserve_first_n to force preserving initial items as separate centroids.
Saves visualizations when multiple centroids are produced.
"""
from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional
import numpy as np

from core.clustering import OnlineManifoldClustering
from analysis.energy_loss import assign_by_cosine, reconstruction_loss
from analysis.manifold_viz import visualize_kv_and_centroids


def generate_synthetic_clusters(num_clusters: int, points_per_cluster: int, dim: int, cluster_std: float = 0.06, seed: int = 42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_clusters, dim)
    keys_list = []
    for c in centers:
        pts = c + rng.normal(scale=cluster_std, size=(points_per_cluster, dim))
        keys_list.append(pts.astype(np.float32))
    keys = np.vstack(keys_list)
    return keys


def run_try(preserve_values: List[Optional[int]], similarity_threshold: float = 0.6, window_size: int = 200):
    keys = generate_synthetic_clusters(num_clusters=6, points_per_cluster=120, dim=32, cluster_std=0.06, seed=42)
    out_dir = os.path.join("experiments", "init_preserve_out")
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for pv in preserve_values:
        cfg = {"init_preserve_first_n": pv, "similarity_threshold": similarity_threshold, "window_size": window_size}
        cl = OnlineManifoldClustering(dim=32, max_memory_size=10000, window_size=window_size, max_centroids=128, metric="cosine", similarity_threshold=similarity_threshold, adaptive_threshold=False, min_merge_similarity=None, init_preserve_first_n=pv)
        cl.add(keys, np.zeros_like(keys))
        centroids, counts, weights = cl.get_centroids()
        C = centroids.shape[0]
        recon = None
        rand = None
        if C > 0:
            assign = assign_by_cosine(keys, centroids)
            recon = reconstruction_loss(keys, assign, centroids)
            rng = np.random.RandomState(42 + (0 if pv is None else pv))
            pick_k = min(C, keys.shape[0])
            idx = rng.choice(keys.shape[0], size=pick_k, replace=False)
            random_centroids = keys[idx]
            assign_r = assign_by_cosine(keys, random_centroids)
            rand = reconstruction_loss(keys, assign_r, random_centroids)
        results.append({"config": cfg, "centroids": int(C), "recon_loss": recon, "random_loss": rand})
        print(f"cfg={cfg} -> C={C} recon={recon} rand={rand}")
        if C > 1:
            try:
                vis_path = os.path.join(out_dir, f"viz_preserve{pv}.png")
                visualize_kv_and_centroids(keys, centroids, save_path=vis_path)
                print(f"saved {vis_path}")
            except Exception as e:
                print(f"visualization failed for pv={pv}: {e}")

    out_path = os.path.join(out_dir, "init_preserve_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")
    return results


if __name__ == "__main__":
    preserve_vals = [None, 2, 5, 10, 20]
    run_try(preserve_vals)


