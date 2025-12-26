"""
Synthetic "Needles-in-a-haystack" experiment for MP-KVM.

This script creates a long stream of KV vectors with a small fraction of
rare "needle" vectors and measures how often those needles are preserved
by the online clustering operator (i.e., whether centroids recover them).

Usage:
    python -m data.needles.run_niah
"""
from __future__ import annotations
import argparse
import os
import json
import numpy as np

from core.clustering import OnlineManifoldClustering


def make_synthetic_stream(total: int, dim: int, n_clusters: int, cluster_std: float, n_needles: int, seed: int = 0,
                          needle_near: bool = False, needle_near_scale: float = 0.5):
    rng = np.random.RandomState(seed)
    # cluster centers
    centers = rng.randn(n_clusters, dim).astype(np.float32) * 5.0
    # assign most tokens to clusters
    cluster_ids = rng.randint(0, n_clusters, size=(total - n_needles,))
    keys = []
    values = []
    for cid in cluster_ids:
        k = centers[cid] + rng.randn(dim).astype(np.float32) * cluster_std
        keys.append(k)
        values.append(k.copy())
    # generate needles: by default far from centers; if needle_near True, sample near random centers
    needles = []
    if needle_near:
        for _ in range(n_needles):
            cid = rng.randint(0, n_clusters)
            n = centers[cid] + rng.randn(dim).astype(np.float32) * (cluster_std * float(needle_near_scale))
            needles.append(n)
            keys.append(n)
            values.append(n.copy())
    else:
        for _ in range(n_needles):
            # sample from a separate high-variance gaussian far away (original behavior)
            n = rng.randn(dim).astype(np.float32) * (cluster_std * 8.0) + rng.randn(dim).astype(np.float32) * 10.0
            needles.append(n)
            keys.append(n)
            values.append(n.copy())

    keys = np.stack(keys, axis=0).astype(np.float32)
    values = np.stack(values, axis=0).astype(np.float32)
    needles = np.stack(needles, axis=0).astype(np.float32)
    return keys, values, needles


def evaluate_recall(centroids: np.ndarray, needles: np.ndarray, threshold: float = 0.85):
    """
    For each needle vector, compute max cosine similarity with centroids.
    Count as recovered if max_sim >= threshold.
    """
    if centroids is None or centroids.shape[0] == 0:
        return 0.0

    # normalize
    def _norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    c_n = _norm(centroids)
    needles_n = _norm(needles)
    sims = np.dot(needles_n, c_n.T)
    max_sims = sims.max(axis=1)
    recovered = (max_sims >= threshold).sum()
    return float(recovered) / float(needles.shape[0])


def run_experiment(args):
    keys, values, needles = make_synthetic_stream(
        total=args.total_tokens,
        dim=args.dim,
        n_clusters=args.n_clusters,
        cluster_std=args.cluster_std,
        n_needles=args.n_needles,
        seed=args.seed,
    )

    # create online clustering operator with sliding window to emulate KV cache
    cluster = OnlineManifoldClustering(dim=args.dim, window_size=args.window_size, max_memory_size=args.max_memory_size,
                                       max_centroids=args.max_centroids, metric="cosine",
                                       similarity_threshold=args.similarity_threshold)

    # ingest stream in small batches to mimic token-by-token arrival
    batch_size = args.batch_size
    n = keys.shape[0]
    for i in range(0, n, batch_size):
        k_batch = keys[i: i + batch_size]
        v_batch = values[i: i + batch_size]
        cluster.add(k_batch, v_batch)

    centroids, counts, weights = cluster.get_centroids()
    recall = evaluate_recall(centroids, needles, threshold=args.recall_threshold)

    out = {
        "params": vars(args),
        "num_centroids": int(centroids.shape[0]) if centroids is not None else 0,
        "recall": float(recall),
    }
    print(json.dumps(out, indent=2))
    if args.out is not None:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "needles_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        # also save centroids and metadata for visualization/export
        try:
            cent_path = os.path.join(args.out, "centroids.npy")
            np.save(cent_path, centroids)
            counts_path = os.path.join(args.out, "centroid_counts.npy")
            np.save(counts_path, counts)
            weights_path = os.path.join(args.out, "centroid_weights.npy")
            np.save(weights_path, weights)
        except Exception:
            # ignore save failures but keep JSON
            pass
        print(f"Wrote results to {out_path} (centroids saved to {args.out})")
        # optionally save centroids and counts as numpy for downstream visualization
        try:
            if getattr(args, "save_centroids", True):
                cent_path = os.path.join(args.out, "centroids.npy")
                counts_path = os.path.join(args.out, "centroid_counts.npy")
                weights_path = os.path.join(args.out, "centroid_weights.npy")
                # ensure arrays are present
                if centroids is None:
                    cent_np = np.zeros((0, args.dim), dtype=np.float32)
                else:
                    cent_np = centroids.astype(np.float32)
                np.save(cent_path, cent_np)
                np.save(counts_path,
                        counts.astype(np.float32) if counts is not None else np.array([], dtype=np.float32))
                np.save(weights_path,
                        weights.astype(np.float32) if weights is not None else np.array([], dtype=np.float32))
                print(f"Wrote centroids to {cent_path} and counts to {counts_path}")
        except Exception:
            pass


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--total-tokens", type=int, default=20000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--cluster-std", type=float, default=0.5)
    p.add_argument("--n-needles", type=int, default=50)
    p.add_argument("--window-size", type=int, default=4096)
    p.add_argument("--max-memory-size", type=int, default=65536)
    p.add_argument("--max-centroids", type=int, default=1024)
    p.add_argument("--similarity-threshold", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--recall-threshold", type=float, default=0.85)
    p.add_argument("--needle-near", dest="needle_near", action="store_true",
                   help="Sample needles near existing cluster centers (easier to recover).")
    p.add_argument("--needle-near-scale", type=float, default=0.5,
                   help="Scale for needle proximity when --needle-near is used.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
