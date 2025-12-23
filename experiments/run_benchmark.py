"""
Minimal experiment runner for MP-KVM PoC.

This script creates synthetic KV vectors, feeds them into MPKVMManager and
produces a UMAP plot and energy loss numbers for inspection.
"""
from __future__ import annotations
import argparse
import numpy as np
import yaml
import os
from core import MPKVMManager
from core.clustering import OnlineManifoldClustering
from analysis.energy_loss import reconstruction_loss, assign_by_cosine
from analysis.manifold_viz import visualize_kv_and_centroids


def run_synthetic(cfg_path: str, out_dir: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dim = int(cfg["model"]["dim"])
    maxc = int(cfg["run"]["max_centroids_per_layer"])
    sim_thresh = float(cfg["run"]["similarity_threshold"])
    rng = np.random.RandomState(int(cfg["run"]["seed"]))

    # synthetic: create N points from several gaussian clusters to emulate semantic clusters
    centers = rng.normal(size=(8, dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    points = []
    for c in centers:
        points.append(c + 0.01 * rng.normal(size=(200, dim)))
    points = np.vstack(points).astype(np.float32)

    manager = MPKVMManager(dim=dim, max_centroids_per_layer=maxc, distance="cosine", sliding_window_size=500)
    # add in small batches to simulate streaming
    batch = 64
    for i in range(0, points.shape[0], batch):
        b = points[i : i + batch]
        manager.process_kv("layer_0", b, similarity_threshold=sim_thresh)

    centroids, counts = manager.get_layer_centroids("layer_0")
    os.makedirs(out_dir, exist_ok=True)
    vis_path = os.path.join(out_dir, "manifold.png")
    visualize_kv_and_centroids(points, centroids, save_path=vis_path)
    print(f"Saved manifold visualization to {vis_path}")
    losses = manager.energy_loss(lambda_diversity=float(cfg["run"].get("lambda_diversity", 0.0)))
    for k, v in losses.items():
        print(f"Layer {k} energy loss: {v:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/configs/default.yaml")
    p.add_argument("--out", default="experiments/out")
    args = p.parse_args()
    run_synthetic(args.config, args.out)


if __name__ == "__main__":
    main()

def poc_run(seed: int = 0, n_tokens: int = 2048, dim: int = 128, compress_batch: int = 512):
    rng = np.random.RandomState(seed)
    keys = rng.normal(size=(n_tokens, dim)).astype(np.float32)
    vals = rng.normal(size=(n_tokens, dim)).astype(np.float32)
    # simulate attention weights as softmax over random scores
    attn_scores = rng.rand(n_tokens).astype(float)
    attn_scores /= attn_scores.sum()

    m = OnlineManifoldClustering(dim=dim, window_size=1024, max_centroids=128, metric="cosine", similarity_threshold=0.85)
    # feed in chunks
    for i in range(0, n_tokens, compress_batch):
        k_batch = keys[i : i + compress_batch]
        v_batch = vals[i : i + compress_batch]
        w_batch = attn_scores[i : i + compress_batch]
        m.add(k_batch, v_batch, w_batch)

    centroids, counts, weights = m.get_centroids()
    # compute assignments and reconstruction loss
    assignments = assign_by_cosine(keys, centroids) if centroids.shape[0] > 0 else np.zeros((keys.shape[0],), dtype=int)
    loss_centroid = reconstruction_loss(keys, assignments, centroids) if centroids.shape[0] > 0 else float("inf")

    # baseline: random sampling of same number as centroids
    if centroids.shape[0] > 0:
        rng_idx = rng.choice(np.arange(keys.shape[0]), size=centroids.shape[0], replace=False)
        random_centroids = keys[rng_idx]
        assignments_rand = assign_by_cosine(keys, random_centroids)
        loss_rand = reconstruction_loss(keys, assignments_rand, random_centroids)
    else:
        loss_rand = float("inf")

    return {
        "n_tokens": n_tokens,
        "dim": dim,
        "n_centroids": centroids.shape[0],
        "loss_centroid": loss_centroid,
        "loss_random": loss_rand,
        "centroids": centroids,
        "keys": keys,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_tokens", type=int, default=2048)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=str, default="poc_out.png")
    args = p.parse_args()

    res = poc_run(seed=args.seed, n_tokens=args.n_tokens, dim=args.dim)
    print("Results:")
    print(f"  tokens: {res['n_tokens']}  dim: {res['dim']}  centroids: {res['n_centroids']}")
    print(f"  loss (centroid): {res['loss_centroid']:.4f}")
    print(f"  loss (random baseline): {res['loss_random']:.4f}")

    if args.plot:
        try:
            os.makedirs("figures", exist_ok=True)
            visualize_kv_and_centroids(res["keys"][:1024], res["centroids"], save_path=os.path.join("figures", args.out))
            print(f"Saved manifold plot to figures/{args.out}")
        except Exception as e:
            print("Plotting failed:", e)


if __name__ == "__main__":
    main()


