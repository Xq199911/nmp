"""
Streaming benchmark: simulate continuous token batches, periodic flushes.
Measures total CPU<->GPU bytes moved (via counting CPU manager) and end-to-end latency.
"""
from __future__ import annotations
import os
import time
import json
from typing import List, Dict, Any

def run_streaming_benchmark(batch_size: int = 1024, dim: int = 64, total_batches: int = 50, flush_interval: int = 10, device: str = "cuda:0"):
    try:
        import torch
    except Exception:
        raise RuntimeError("torch required")

    from core.integration import MPKVMManager
    from core.integration_gpu import MPKVMGPUAggregator, MPKVMGPUAggregatorOptimized

    # counting CPU manager
    class CountingCPUManager(MPKVMManager):
        def __init__(self, dim, num_layers=1):
            super().__init__(dim=dim, num_layers=num_layers)
            self.bytes_received = 0

        def add_kv(self, layer_idx, keys, values, weights=None, similarity_threshold: float = 0.1):
            try:
                self.bytes_received += keys.nbytes + values.nbytes
            except Exception:
                pass
            super().add_kv(layer_idx, keys, values, weights=weights, similarity_threshold=similarity_threshold)

    results = []
    for cls_name in ("baseline", "optimized"):
        cpu_mgr = CountingCPUManager(dim=dim, num_layers=1)
        if cls_name == "baseline":
            AggCls = MPKVMGPUAggregator
        else:
            AggCls = MPKVMGPUAggregatorOptimized

        agg = AggCls(cpu_mgr, dim=dim, device=device, max_gpu_centroids_per_layer=256, similarity_threshold=0.5)

        # run stream
        start = time.perf_counter()
        for b in range(total_batches):
            k = torch.randn((batch_size, dim), device=device)
            v = k.clone()
            t0 = time.perf_counter()
            agg.add_kv_torch(0, k, v)
            t1 = time.perf_counter()
            # optionally periodic flush
            if (b + 1) % flush_interval == 0:
                f0 = time.perf_counter()
                agg.flush_all_to_cpu()
                f1 = time.perf_counter()
                flush_time = f1 - f0
            else:
                flush_time = 0.0
            results.append({"impl": cls_name, "batch": b, "add_time_s": t1 - t0, "flush_time_s": flush_time, "bytes_received_so_far": cpu_mgr.bytes_received})
        total_time = time.perf_counter() - start
        # final flush
        agg.flush_all_to_cpu()
        total_bytes = cpu_mgr.bytes_received
        print(f"{cls_name}: total_time={total_time:.4f}s total_bytes={total_bytes}")
        results.append({"impl": cls_name, "total_time_s": total_time, "total_bytes": total_bytes})

    out_dir = os.path.join("experiments", "benchmark_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "streaming_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved streaming benchmark to {out_path}")
    return results


if __name__ == "__main__":
    run_streaming_benchmark(batch_size=1024, dim=64, total_batches=40, flush_interval=8)


