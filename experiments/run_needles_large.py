"""
Run larger Needles experiments (total_tokens=20000) on selected configs and save results.
"""
from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List

try:
    import torch
except Exception:
    torch = None

from core.integration import MPKVMManager
from core.integration_gpu import MPKVMGPUAggregatorOptimized
from data.needles.run_niah import make_synthetic_stream, evaluate_recall


def run_cfg(cfg: Dict[str, Any]):
    keys, values, needles = make_synthetic_stream(total=cfg["total_tokens"], dim=cfg["dim"], n_clusters=cfg["n_clusters"], cluster_std=cfg["cluster_std"], n_needles=cfg["n_needles"], seed=cfg.get("seed", 0), needle_near=True, needle_near_scale=cfg.get("needle_near_scale", 0.5))
    cpu_mgr = MPKVMManager(dim=cfg["dim"], num_layers=1, **{"cluster_kwargs": cfg["cluster_kwargs"]})
    agg = MPKVMGPUAggregatorOptimized(cpu_mgr, dim=cfg["dim"], device=cfg["device"], max_gpu_centroids_per_layer=cfg.get("max_gpu_centroids_per_layer", 512), similarity_threshold=cfg.get("agg_similarity", 0.7))
    start = time.perf_counter()
    bs = cfg["batch_size"]
    for i in range(0, keys.shape[0], bs):
        kt = torch.from_numpy(keys[i:i+bs]).to(cfg["device"])
        vt = torch.from_numpy(values[i:i+bs]).to(cfg["device"])
        agg.add_kv_torch(0, kt, vt)
        if ((i // bs) + 1) % (cfg["flush_interval"] // bs) == 0:
            agg.flush_all_to_cpu()
    agg.flush_all_to_cpu()
    total_time = time.perf_counter() - start
    centroids, counts, weights = cpu_mgr.get_layer_centroids(0)
    recall = evaluate_recall(centroids, needles, threshold=cfg.get("recall_threshold", 0.85))
    return {"cfg": cfg, "total_time_s": total_time, "num_centroids": int(centroids.shape[0]) if centroids is not None else 0, "recall": float(recall)}


def main():
    if torch is None:
        raise RuntimeError("torch required")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base = {"total_tokens": 20000, "dim": 64, "n_clusters": 20, "cluster_std": 0.5, "n_needles": 50, "batch_size": 32, "flush_interval": 256, "device": device, "agg_similarity": 0.7, "recall_threshold": 0.85, "max_gpu_centroids_per_layer": 512}
    configs: List[Dict[str, Any]] = []
    configs.append({**base, "cluster_kwargs": {"init_preserve_first_n": None, "similarity_threshold": 0.4, "min_merge_similarity": None, "window_size": 50}})
    configs.append({**base, "cluster_kwargs": {"init_preserve_first_n": 2, "similarity_threshold": 0.4, "min_merge_similarity": None, "window_size": 50}})
    configs.append({**base, "cluster_kwargs": {"init_preserve_first_n": 10, "similarity_threshold": 0.4, "min_merge_similarity": None, "window_size": 50}})

    out = []
    out_dir = os.path.join("experiments", "needles_large_out")
    os.makedirs(out_dir, exist_ok=True)
    for cfg in configs:
        print("Running large:", cfg["cluster_kwargs"])
        res = run_cfg(cfg)
        out.append(res)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()


