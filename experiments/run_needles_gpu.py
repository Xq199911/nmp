"""
Run a Needles-in-a-haystack experiment using the GPU aggregator (optimized).
Feeds synthetic long KV stream to the GPU aggregator, periodically flushes to CPU manager,
and computes recall of needles against CPU centroids.
"""
from __future__ import annotations
import os
import json
import time
import numpy as np

try:
    import torch
except Exception:
    torch = None

from core.integration import MPKVMManager
from core.integration_gpu import MPKVMGPUAggregatorOptimized
from data.needles.run_niah import make_synthetic_stream, evaluate_recall


def run_needles_gpu(total_tokens: int = 4000, dim: int = 64, n_clusters: int = 20, cluster_std: float = 0.5, n_needles: int = 20, batch_size: int = 32, flush_interval: int = 128, device: str = "cuda:0", out_dir: str = "experiments/needles_out"):
    if torch is None:
        raise RuntimeError("torch is required to run GPU experiment")

    keys, values, needles = make_synthetic_stream(total=total_tokens, dim=dim, n_clusters=n_clusters, cluster_std=cluster_std, n_needles=n_needles, seed=0)

    # cpu manager to collect flushed centroids
    cpu_mgr = MPKVMManager(dim=dim, num_layers=1)
    agg = MPKVMGPUAggregatorOptimized(cpu_mgr, dim=dim, device=device, max_gpu_centroids_per_layer=512, similarity_threshold=0.7)

    start = time.perf_counter()
    n = keys.shape[0]
    for i in range(0, n, batch_size):
        k_batch = keys[i : i + batch_size]
        v_batch = values[i : i + batch_size]
        # move to torch on device
        kt = torch.from_numpy(k_batch).to(device)
        vt = torch.from_numpy(v_batch).to(device)
        agg.add_kv_torch(0, kt, vt)
        if ((i // batch_size) + 1) % (flush_interval // batch_size) == 0:
            agg.flush_all_to_cpu()
    # final flush
    agg.flush_all_to_cpu()
    total_time = time.perf_counter() - start

    centroids, counts, weights = cpu_mgr.get_layer_centroids(0)
    recall = evaluate_recall(centroids, needles, threshold=0.85)

    out = {"total_tokens": total_tokens, "dim": dim, "n_clusters": n_clusters, "n_needles": n_needles, "batch_size": batch_size, "flush_interval": flush_interval, "total_time_s": total_time, "num_centroids": int(centroids.shape[0]) if centroids is not None else 0, "recall": float(recall)}
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "needles_gpu_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved results to {out_path}")
    return out


if __name__ == "__main__":
    dev = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
    run_needles_gpu(total_tokens=4000, dim=64, n_clusters=20, cluster_std=0.5, n_needles=20, batch_size=32, flush_interval=256, device=dev)


