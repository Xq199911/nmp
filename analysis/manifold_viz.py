"""
Visualization utilities for KV manifold.
Produces UMAP projections and basic matplotlib figures.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


def visualize_kv_and_centroids(keys: np.ndarray, centroids: np.ndarray, save_path: Optional[str] = None, title: str = "KV manifold"):
    """
    keys: (N, D)
    centroids: (C, D)
    """
    # prefer UMAP but fall back to PCA if not available
    try:
        import umap  # type: ignore
        import matplotlib.pyplot as plt
        reducer = umap.UMAP(n_components=2, random_state=42)
    except Exception:
        try:
            from sklearn.decomposition import PCA  # type: ignore
            import matplotlib.pyplot as plt
            reducer = PCA(n_components=2)
        except Exception as e:
            raise RuntimeError("Please install 'umap-learn' or 'scikit-learn' and 'matplotlib' to use visualize_kv_and_centroids") from e

    if keys.shape[0] == 0 and centroids.shape[0] == 0:
        raise ValueError("empty inputs")

    X = np.concatenate([keys, centroids], axis=0)
    Z = reducer.fit_transform(X)
    n = keys.shape[0]
    C = centroids.shape[0]
    zk = Z[:n]
    zc = Z[n:]

    plt.figure(figsize=(8, 6))
    plt.scatter(zk[:, 0], zk[:, 1], s=6, alpha=0.6, label="tokens", c="C0")
    if C > 0:
        plt.scatter(zc[:, 0], zc[:, 1], s=60, alpha=0.9, label="centroids", c="C3", marker="X")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()


__all__ = ["visualize_kv_and_centroids"]


