"""
Export per-layer centroids and simple plots from a local model using MPKVM attach.
"""
from __future__ import annotations
import os
import json
import numpy as np
from typing import Optional

def main(model_path: Optional[str] = None, out_dir: str = "experiments/realmodel_out", repeat: int = 5):
    os.makedirs(out_dir, exist_ok=True)
    # Import lazily to avoid hard dependency at module import time
    try:
        from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required to run this exporter") from e

    # load model (prefer local folder)
    candidate = model_path or "model"
    if not os.path.isdir(candidate):
        raise FileNotFoundError(f"Model folder not found: {candidate}")

    # find folder containing config.json
    found_dir = None
    for root, dirs, files in os.walk(candidate):
        if "config.json" in files:
            found_dir = root
            break
    if found_dir is None:
        raise FileNotFoundError("Could not find a model folder with config.json under " + candidate)

    # load tiny local model (this may be large; we set local_files_only=True)
    try:
        model = LlamaForCausalLM.from_pretrained(found_dir, torch_dtype="auto", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(found_dir, use_fast=True, local_files_only=True)
    except Exception as e:
        raise

    # attach mpkvm manager
    from core.integration import MPKVMManager
    from adapters.llama_adapter import attach_mpkvm_to_hf_llama

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    cpu_manager = MPKVMManager(dim=hidden_size, num_layers=num_layers, cluster_kwargs={"max_centroids": 1024, "sliding_window_size": 16384})
    attach_mpkvm_to_hf_llama(model, cpu_manager, head_mean=True, sample_stride=1, enable_injection=False)

    # run a few generations to collect K/V
    prompts = [
        "In 100 words, explain the significance of manifold partitioned KV memories.",
        "Briefly summarize how centroid compression can help long-context transformers.",
        "Explain online clustering for KV caches and why it's useful.",
        "Discuss challenges of position encoding when merging KV tokens.",
    ]
    import torch
    for _ in range(max(1, int(repeat))):
        for p in prompts:
            try:
                inputs = tokenizer(p, return_tensors="pt")
            except Exception:
                inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}
            _ = model.generate(**inputs, max_new_tokens=64)

    # flush GPU agg if present
    try:
        if hasattr(cpu_manager, "flush_all_to_cpu"):
            cpu_manager.flush_all_to_cpu()
    except Exception:
        pass

    # Force compress any remaining sliding buffers into centroids (best-effort)
    try:
        for li, layer_cluster in cpu_manager.layers.items():
            try:
                # for OnlineManifoldClustering instances, compress any buffered keys
                if hasattr(layer_cluster, "keys_buffer") and hasattr(layer_cluster, "_compress_oldest_batch"):
                    buf_len = len(getattr(layer_cluster, "keys_buffer", []))
                    if buf_len > 0:
                        layer_cluster._compress_oldest_batch(buf_len)
            except Exception:
                # non-fatal
                pass
    except Exception:
        pass

    # collect per-layer centroids and save
    summary = {}
    for li in range(num_layers):
        try:
            centroids, counts, weights = cpu_manager.get_layer_centroids(li)
            if centroids is None:
                centroids = np.zeros((0, hidden_size), dtype=np.float32)
            if counts is None:
                counts = np.array([], dtype=int)
            if weights is None:
                weights = np.array([], dtype=float)
            layer_dir = os.path.join(out_dir, f"layer_{li}")
            os.makedirs(layer_dir, exist_ok=True)
            np.save(os.path.join(layer_dir, "centroids.npy"), centroids)
            np.save(os.path.join(layer_dir, "counts.npy"), counts)
            np.save(os.path.join(layer_dir, "weights.npy"), weights)
            summary[li] = {"centroids": int(centroids.shape[0]), "sum_counts": float(counts.sum()) if counts.size > 0 else 0.0}
        except Exception:
            summary[li] = {"error": True}

    # save summary JSON
    with open(os.path.join(out_dir, "centroids_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"layers": summary}, f, indent=2)

    # aggregate all centroids for embedding visualization
    all_cent = []
    for li in range(num_layers):
        p = os.path.join(out_dir, f"layer_{li}", "centroids.npy")
        if os.path.exists(p):
            arr = np.load(p)
            if arr.size > 0:
                all_cent.append(arr)
    if all_cent:
        all_cent = np.vstack(all_cent)
        # UMAP/PCA fallback
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            z = reducer.fit_transform(all_cent)
        except Exception:
            from sklearn.decomposition import PCA
            z = PCA(n_components=2).fit_transform(all_cent)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(z[:, 0], z[:, 1], s=6, alpha=0.8)
        ax.set_title("All layer centroids embedding")
        fig.savefig(os.path.join(out_dir, "all_centroids_embedding.png"), dpi=200)
        plt.close(fig)

    print("Export complete; summary written to", os.path.join(out_dir, "centroids_summary.json"))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="model")
    p.add_argument("--out", type=str, default="experiments/realmodel_out")
    p.add_argument("--repeat", type=int, default=5)
    args = p.parse_args()
    main(model_path=args.model, out_dir=args.out, repeat=args.repeat)


