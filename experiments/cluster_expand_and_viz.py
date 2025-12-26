"""
Expanded hyperparameter sweep + visualization.
Tries lower similarity thresholds and larger window sizes to avoid over-merging,
and saves UMAP/PCA visualizations for configs that produce >1 centroids.
"""
from __future__ import annotations
import os
import json
from typing import Any, Dict, List
import numpy as np

from core.clustering import OnlineManifoldClustering
from analysis.energy_loss import assign_by_cosine, reconstruction_loss
from analysis.manifold_viz import visualize_kv_and_centroids


def generate_synthetic_clusters(num_clusters: int, points_per_cluster: int, dim: int, cluster_std: float = 0.06, seed: int = 1234):
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_clusters, dim)
    keys_list = []
    for c in centers:
        pts = c + rng.normal(scale=cluster_std, size=(points_per_cluster, dim))
        keys_list.append(pts.astype(np.float32))
    keys = np.vstack(keys_list)
    return keys


def expand_and_visualize(
    similarity_thresholds: List[float],
    window_sizes: List[int],
    adaptive_flags: List[bool],
    num_clusters: int = 6,
    points_per_cluster: int = 120,
    dim: int = 32,
    seed: int = 1234,
):
    keys = generate_synthetic_clusters(num_clusters=num_clusters, points_per_cluster=points_per_cluster, dim=dim, cluster_std=0.06, seed=seed)
    out_dir = os.path.join("experiments", "cluster_expand_out")
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for sim in similarity_thresholds:
        for win in window_sizes:
            for adaptive in adaptive_flags:
                cfg = {"similarity_threshold": float(sim), "window_size": int(win), "adaptive_threshold": bool(adaptive)}
                cl = OnlineManifoldClustering(dim=dim, max_memory_size=10000, window_size=win, max_centroids=256, metric="cosine", similarity_threshold=sim, adaptive_threshold=adaptive)
                cl.add(keys, np.zeros_like(keys))
                centroids, counts, weights = cl.get_centroids()
                C = centroids.shape[0]
                recon = None
                rand = None
                ratio = None
                if C > 0:
                    assignments = assign_by_cosine(keys, centroids)
                    recon = reconstruction_loss(keys, assignments, centroids)
                    rng = np.random.RandomState(seed + int(win) + int(sim * 100))
                    pick_k = min(C, keys.shape[0])
                    idx = rng.choice(keys.shape[0], size=pick_k, replace=False)
                    random_centroids = keys[idx]
                    assignments_rand = assign_by_cosine(keys, random_centroids)
                    rand = reconstruction_loss(keys, assignments_rand, random_centroids)
                    ratio = float(recon / (rand + 1e-12))

                entry = {"config": cfg, "centroids": int(C), "recon_loss": recon, "random_loss": rand, "ratio": ratio}
                results.append(entry)
                print(f"cfg={cfg} -> C={C} recon={recon} rand={rand} ratio={ratio}")

                # visualize when multiple centroids exist
                if C > 1:
                    try:
                        vis_path = os.path.join(out_dir, f"viz_sim{sim}_win{win}_adapt{int(adaptive)}.png")
                        fig, ax = visualize_kv_and_centroids(keys, centroids, save_path=vis_path)
                        print(f"saved visualization to {vis_path}")
                    except Exception as e:
                        print(f"visualization failed for cfg={cfg}: {e}")

    json_path = os.path.join(out_dir, "expand_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved results to {json_path}")
    return results


if __name__ == "__main__":
    sims = [0.1, 0.2, 0.3, 0.4, 0.5]
    wins = [50, 200, 500]
    adaptive = [False, True]
    expand_and_visualize(sims, wins, adaptive)


