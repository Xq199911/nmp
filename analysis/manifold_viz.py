"""
UMAP-based visualization utilities for KV manifolds and centroids.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def visualize_kv_and_centroids(
    kv_vectors: np.ndarray,
    centroid_vectors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    random_state: int = 42,
):
    """
    Create a 2D UMAP projection showing original KV vectors and centroids.
    If UMAP is not available, fall back to PCA for a workable PoC.
    """
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=random_state)
    except Exception:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)

    all_vecs = kv_vectors
    if centroid_vectors is not None and centroid_vectors.shape[0] > 0:
        all_vecs = np.vstack([kv_vectors, centroid_vectors])

    proj = reducer.fit_transform(all_vecs)
    n = kv_vectors.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj[:n, 0], proj[:n, 1], c="C0", s=6, label="KV tokens", alpha=0.6)
    if centroid_vectors is not None and centroid_vectors.shape[0] > 0:
        m = centroid_vectors.shape[0]
        ax.scatter(proj[n:n + m, 0], proj[n:n + m, 1], c="C1", s=50, marker="X", label="Centroids")
    if labels is not None:
        # optional: color by labels
        pass
    ax.legend()
    ax.set_title("KV manifold (UMAP/PCA)")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax

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
    try:
        import umap
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("Please install 'umap-learn' and 'matplotlib' to use visualize_kv_and_centroids") from e

    if keys.shape[0] == 0 and centroids.shape[0] == 0:
        raise ValueError("empty inputs")

    X = np.concatenate([keys, centroids], axis=0)
    reducer = umap.UMAP(n_components=2, random_state=42)
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


