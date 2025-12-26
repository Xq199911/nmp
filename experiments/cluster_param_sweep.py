"""
Sweep clustering hyperparameters (persistence_decay, sliding_window_size, max_centroids)
to evaluate their effect on needle recall for far-away needles.
"""
from __future__ import annotations
import os
import json
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from data.needles.run_niah import make_synthetic_stream, evaluate_recall
from core.clustering import OnlineManifoldCluster


def run_param_grid(persistence_list: List[float], window_list: List[Optional[int]], max_centroids_list: List[int], out_dir: str, total_tokens: int = 10000, dim: int = 64):
    results = []
    keys, values, needles = make_synthetic_stream(total=total_tokens, dim=dim, n_clusters=20, cluster_std=0.5, n_needles=50, seed=0, needle_near=False)
    for maxc in max_centroids_list:
        for win in window_list:
            for pd in persistence_list:
                print(f"Running maxc={maxc} win={win} pd={pd}")
                cluster = OnlineManifoldCluster(dim=dim, max_centroids=maxc, sliding_window_size=win, persistence_decay=pd)
                # ingest
                n = keys.shape[0]
                bs = 32
                for i in range(0, n, bs):
                    cluster.add(keys[i : i + bs])
                centroids, counts = cluster.get_centroids()
                recall = evaluate_recall(centroids, needles, threshold=0.85)
                results.append({"max_centroids": int(maxc), "window_size": (int(win) if win is not None else None), "persistence_decay": float(pd), "num_centroids": int(centroids.shape[0]) if centroids is not None else 0, "recall": float(recall)})
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cluster_param_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"params": {"persistence_list": persistence_list, "window_list": window_list, "max_centroids_list": max_centroids_list}, "results": results}, f, indent=2)
    # plot example: persistence vs recall for each max_centroids with window fixed (None -> label)
    plt.figure(figsize=(10, 6))
    for maxc in sorted(set([r["max_centroids"] for r in results])):
        for win in sorted(set([r["window_size"] for r in results]), key=lambda x: (x is None, x)):
            sel = [r for r in results if r["max_centroids"] == maxc and r["window_size"] == win]
            sel_sorted = sorted(sel, key=lambda x: x["persistence_decay"])
            xs = [r["persistence_decay"] for r in sel_sorted]
            ys = [r["recall"] for r in sel_sorted]
            plt.plot(xs, ys, marker="o", label=f"maxc={maxc},win={win}")
    plt.xlabel("persistence_decay")
    plt.ylabel("recall")
    plt.title("Recall vs persistence_decay (cluster param sweep)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    out_png = os.path.join(out_dir, "cluster_param_recall.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote results to", out_path, "and", out_png)
    return out_path, out_png


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="experiments/cluster_param_out")
    p.add_argument("--total-tokens", type=int, default=10000)
    args = p.parse_args()
    run_param_grid(persistence_list=[1.0, 0.9, 0.7, 0.5], window_list=[None, 1024, 4096], max_centroids_list=[128, 256, 512], out_dir=args.out, total_tokens=args.total_tokens)


