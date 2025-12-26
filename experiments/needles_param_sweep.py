"""
Parameter sweep for Needles experiment using GPU aggregator.
Varies cluster construction params (init_preserve_first_n, similarity_threshold, min_merge_similarity)
and flush intervals. Saves results to experiments/needles_sweep_out/.
"""
from __future__ import annotations
import os
import json
import csv
import time
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:
    torch = None

from core.integration import MPKVMManager
from core.integration_gpu import MPKVMGPUAggregatorOptimized
from data.needles.run_niah import make_synthetic_stream, evaluate_recall


def run_one(cfg: Dict[str, Any]):
    # unpack
    total_tokens = int(cfg.get("total_tokens", 4000))
    dim = int(cfg.get("dim", 64))
    n_clusters = int(cfg.get("n_clusters", 20))
    cluster_std = float(cfg.get("cluster_std", 0.5))
    n_needles = int(cfg.get("n_needles", 20))
    batch_size = int(cfg.get("batch_size", 32))
    flush_interval = int(cfg.get("flush_interval", 256))
    device = cfg.get("device", "cuda:0")
    cluster_kwargs = cfg.get("cluster_kwargs", {}) or {}

    keys, values, needles = make_synthetic_stream(total=total_tokens, dim=dim, n_clusters=n_clusters, cluster_std=cluster_std, n_needles=n_needles, seed=0)

    # create cpu manager with cluster kwargs so clusters use same params
    cpu_mgr = MPKVMManager(dim=dim, num_layers=1, **{"cluster_kwargs": cluster_kwargs})
    agg = MPKVMGPUAggregatorOptimized(cpu_mgr, dim=dim, device=device, max_gpu_centroids_per_layer=512, similarity_threshold=cfg.get("agg_similarity", 0.7))

    start = time.perf_counter()
    n = keys.shape[0]
    for i in range(0, n, batch_size):
        k_batch = keys[i : i + batch_size]
        v_batch = values[i : i + batch_size]
        kt = torch.from_numpy(k_batch).to(device)
        vt = torch.from_numpy(v_batch).to(device)
        agg.add_kv_torch(0, kt, vt)
        if ((i // batch_size) + 1) % (flush_interval // batch_size) == 0:
            agg.flush_all_to_cpu()
    agg.flush_all_to_cpu()
    total_time = time.perf_counter() - start

    centroids, counts, weights = cpu_mgr.get_layer_centroids(0)
    recall = evaluate_recall(centroids, needles, threshold=cfg.get("recall_threshold", 0.85))

    return {
        "cfg": cfg,
        "total_time_s": total_time,
        "num_centroids": int(centroids.shape[0]) if centroids is not None else 0,
        "recall": float(recall),
    }


def run_sweep():
    if torch is None:
        raise RuntimeError("torch required")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    init_vals: List[Optional[int]] = [None, 2, 5, 10]
    sim_vals: List[float] = [0.4, 0.6, 0.8]
    min_merge_vals: List[Optional[float]] = [None, 0.6, 0.8]
    flush_vals: List[int] = [128, 256, 512]

    out_dir = os.path.join("experiments", "needles_sweep_out")
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for init in init_vals:
        for sim in sim_vals:
            for mm in min_merge_vals:
                for flush in flush_vals:
                    cfg = {
                        "total_tokens": 4000,
                        "dim": 64,
                        "n_clusters": 20,
                        "cluster_std": 0.5,
                        "n_needles": 20,
                        "batch_size": 32,
                        "flush_interval": flush,
                        "device": device,
                        "agg_similarity": 0.7,
                        "recall_threshold": 0.85,
                        "cluster_kwargs": {"init_preserve_first_n": init, "similarity_threshold": float(sim), "min_merge_similarity": mm},
                    }
                    print(f"Running cfg init={init} sim={sim} mm={mm} flush={flush}")
                    res = run_one(cfg)
                    results.append(res)
                    # incremental save
                    with open(os.path.join(out_dir, "needles_sweep_results.json"), "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)
    # write CSV
    csv_path = os.path.join(out_dir, "needles_sweep_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        import csv as _csv

        writer = _csv.writer(cf)
        writer.writerow(["init_preserve", "cluster_sim", "min_merge", "flush_interval", "total_time_s", "num_centroids", "recall"])
        for r in results:
            cfg = r["cfg"]
            ck = cfg["cluster_kwargs"]
            writer.writerow([ck.get("init_preserve_first_n"), ck.get("similarity_threshold"), ck.get("min_merge_similarity"), cfg["flush_interval"], r["total_time_s"], r["num_centroids"], r["recall"]])

    print(f"Saved sweep to {out_dir}")
    return results


if __name__ == "__main__":
    run_sweep()


