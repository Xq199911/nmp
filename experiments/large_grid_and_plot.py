"""
Large-ish hyperparameter grid scan combining:
- init_preserve_first_n
- min_merge_similarity
- similarity_threshold
- window_size

Saves results (JSON/CSV) and produces summary plots in experiments/large_grid_out/.
"""
from __future__ import annotations
import os
import json
import csv
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

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


def run_grid(
    init_preserve_values: List[Optional[int]],
    min_merge_values: List[Optional[float]],
    similarity_thresholds: List[float],
    window_sizes: List[int],
    repeats: int = 3,
    out_dir: str = "experiments/large_grid_out",
):
    os.makedirs(out_dir, exist_ok=True)
    keys_cache = {}
    results: List[Dict[str, Any]] = []

    for repeat in range(repeats):
        seed = 100 + repeat
        keys_cache[seed] = generate_synthetic_clusters(num_clusters=6, points_per_cluster=120, dim=32, cluster_std=0.06, seed=seed)

    for init_preserve in init_preserve_values:
        for min_merge in min_merge_values:
            for sim in similarity_thresholds:
                for win in window_sizes:
                    # average over repeats
                    cent_counts = []
                    recon_losses = []
                    rand_losses = []
                    for repeat in range(repeats):
                        seed = 100 + repeat
                        keys = keys_cache[seed]
                        cl = OnlineManifoldClustering(dim=32, max_memory_size=10000, window_size=win, max_centroids=128, metric="cosine", similarity_threshold=sim, adaptive_threshold=False, min_merge_similarity=min_merge, init_preserve_first_n=init_preserve)
                        cl.add(keys, np.zeros_like(keys))
                        centroids, counts, weights = cl.get_centroids()
                        C = int(centroids.shape[0])
                        cent_counts.append(C)
                        if C > 0:
                            assign = assign_by_cosine(keys, centroids)
                            recon = reconstruction_loss(keys, assign, centroids)
                            rng = np.random.RandomState(seed + (0 if init_preserve is None else int(init_preserve)))
                            pick_k = min(C, keys.shape[0])
                            idx = rng.choice(keys.shape[0], size=pick_k, replace=False)
                            random_centroids = keys[idx]
                            assign_r = assign_by_cosine(keys, random_centroids)
                            rand = reconstruction_loss(keys, assign_r, random_centroids)
                        else:
                            recon = None
                            rand = None
                        recon_losses.append(recon)
                        rand_losses.append(rand)

                    # compute averages (ignore None)
                    def mean_ignore_none(xs):
                        xs_f = [x for x in xs if x is not None]
                        return float(np.mean(xs_f)) if xs_f else None

                    avg_C = int(np.mean(cent_counts))
                    avg_recon = mean_ignore_none(recon_losses)
                    avg_rand = mean_ignore_none(rand_losses)
                    ratio = float(avg_recon / (avg_rand + 1e-12)) if avg_recon is not None and avg_rand is not None else None

                    entry = {"init_preserve": init_preserve, "min_merge": min_merge, "similarity": sim, "window": win, "avg_centroids": avg_C, "avg_recon": avg_recon, "avg_rand": avg_rand, "ratio": ratio}
                    results.append(entry)
                    print(f"cfg={entry}")

    # save JSON and CSV
    json_path = os.path.join(out_dir, "large_grid_results.json")
    csv_path = os.path.join(out_dir, "large_grid_results.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["init_preserve", "min_merge", "similarity", "window", "avg_centroids", "avg_recon", "avg_rand", "ratio"])
        for r in results:
            writer.writerow([r["init_preserve"], r["min_merge"], r["similarity"], r["window"], r["avg_centroids"], r["avg_recon"], r["avg_rand"], r["ratio"]])

    # Plotting: show avg_centroids vs avg_recon colored by init_preserve
    try:
        plt.figure(figsize=(9, 6))
        markers = {None: "o", 2: "s", 5: "v", 10: "^", 20: "X"}
        for ip in init_preserve_values:
            xs = [r["avg_centroids"] for r in results if r["init_preserve"] == ip]
            ys = [r["avg_recon"] for r in results if r["init_preserve"] == ip]
            plt.scatter(xs, ys, label=f"preserve={ip}", marker=markers.get(ip, "o"), alpha=0.8)
        plt.xlabel("avg_centroids")
        plt.ylabel("avg_recon_loss")
        plt.legend()
        plt.title("avg_centroids vs avg_recon (colored by init_preserve)")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "centroids_vs_recon_by_init_preserve.png"), dpi=200)
        print("saved centroids_vs_recon_by_init_preserve.png")
    except Exception as e:
        print(f"plot failed: {e}")

    return results


if __name__ == "__main__":
    init_vals = [None, 2, 5, 10]
    min_merge_vals = [None, 0.6, 0.8]
    sims = [0.3, 0.5, 0.7]
    wins = [50, 200]
    run_grid(init_vals, min_merge_vals, sims, wins, repeats=3)


