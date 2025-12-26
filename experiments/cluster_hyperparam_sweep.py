"""
Small hyperparameter sweep for OnlineManifoldClustering.
Saves results to experiments/cluster_sweep_results.json and CSV for quick inspection.
"""
from __future__ import annotations
import json
import os
import csv
from typing import List, Dict, Any
import numpy as np

from core.clustering import OnlineManifoldClustering
from analysis.energy_loss import assign_by_cosine, reconstruction_loss


def generate_synthetic_clusters(num_clusters: int, points_per_cluster: int, dim: int, cluster_std: float = 0.05, seed: int = 42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_clusters, dim)
    keys_list = []
    for c in centers:
        pts = c + rng.normal(scale=cluster_std, size=(points_per_cluster, dim))
        keys_list.append(pts.astype(np.float32))
    keys = np.vstack(keys_list)
    values = np.zeros_like(keys, dtype=np.float32)
    return keys, values


def run_sweep(
    similarity_thresholds: List[float],
    max_centroids_list: List[int],
    persistence_decays: List[float],
    num_clusters: int = 5,
    points_per_cluster: int = 120,
    dim: int = 32,
    seed: int = 42,
):
    results: List[Dict[str, Any]] = []
    keys, values = generate_synthetic_clusters(num_clusters=num_clusters, points_per_cluster=points_per_cluster, dim=dim, cluster_std=0.06, seed=seed)
    N = keys.shape[0]
    for sim in similarity_thresholds:
        for maxc in max_centroids_list:
            for decay in persistence_decays:
                cfg = {"similarity_threshold": float(sim), "max_centroids": int(maxc), "persistence_decay": float(decay)}
                # instantiate clustering operator with a modest window to trigger compression
                cl = OnlineManifoldClustering(dim=dim, max_memory_size=10000, window_size=50, max_centroids=maxc, metric="cosine", similarity_threshold=sim, adaptive_threshold=False)
                # If the operator supports persistence_decay, set it (some implementations may ignore)
                try:
                    setattr(cl, "persistence_decay", float(decay))
                except Exception:
                    pass

                # ingest all data
                cl.add(keys, values)

                centroids, counts, weights = cl.get_centroids()
                C = centroids.shape[0]
                if C == 0:
                    recon_loss = None
                    rand_loss = None
                    ratio = None
                else:
                    assignments = assign_by_cosine(keys, centroids)
                    recon_loss = reconstruction_loss(keys, assignments, centroids)
                    # baseline random centroids
                    rng = np.random.RandomState(seed + int(maxc) + int(sim * 100))
                    pick_k = min(C, keys.shape[0])
                    idx = rng.choice(keys.shape[0], size=pick_k, replace=False)
                    random_centroids = keys[idx]
                    assignments_rand = assign_by_cosine(keys, random_centroids)
                    rand_loss = reconstruction_loss(keys, assignments_rand, random_centroids)
                    ratio = float(recon_loss / (rand_loss + 1e-12))

                results.append({"config": cfg, "centroids": int(C), "recon_loss": recon_loss, "random_loss": rand_loss, "ratio": ratio})
                print(f"cfg={cfg} -> C={C} recon={recon_loss} rand={rand_loss} ratio={ratio}")

    # save results
    out_dir = os.path.join("experiments", "cluster_sweep_out")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "cluster_sweep_results.json")
    csv_path = os.path.join(out_dir, "cluster_sweep_results.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["similarity_threshold", "max_centroids", "persistence_decay", "centroids", "recon_loss", "random_loss", "ratio"])
        for r in results:
            cfg = r["config"]
            writer.writerow([cfg["similarity_threshold"], cfg["max_centroids"], cfg["persistence_decay"], r["centroids"], r["recon_loss"], r["random_loss"], r["ratio"]])

    print(f"saved results to {json_path} and {csv_path}")
    return results


if __name__ == "__main__":
    # small default grid that's quick to run
    sims = [0.6, 0.7, 0.8, 0.9]
    maxcs = [5, 10, 20]
    decays = [1.0, 0.95]
    run_sweep(sims, maxcs, decays)


