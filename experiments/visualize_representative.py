"""
Generate representative visualizations:
- true synthetic clusters
- sklearn KMeans baseline centroids
- OnlineManifoldCluster (numpy) centroids
- OnlineManifoldClustering (streaming) centroids (with small chunking)
"""
from __future__ import annotations
import os
import numpy as np

from analysis.manifold_viz import visualize_kv_and_centroids
from core.clustering import OnlineManifoldCluster, OnlineManifoldClustering

def generate_synthetic_clusters(num_clusters: int, points_per_cluster: int, dim: int, cluster_std: float = 0.06, seed: int = 42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_clusters, dim)
    keys_list = []
    for c in centers:
        pts = c + rng.normal(scale=cluster_std, size=(points_per_cluster, dim))
        keys_list.append(pts.astype(np.float32))
    keys = np.vstack(keys_list)
    return keys, centers


def run_and_save(num_clusters: int = 6, points_per_cluster: int = 120, dim: int = 32):
    keys, true_centers = generate_synthetic_clusters(num_clusters=num_clusters, points_per_cluster=points_per_cluster, dim=dim)
    out_dir = os.path.join("experiments", "visual_representative_out")
    os.makedirs(out_dir, exist_ok=True)

    # sklearn baseline
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(keys)
        km_centroids = kmeans.cluster_centers_
        visualize_kv_and_centroids(keys, km_centroids, save_path=os.path.join(out_dir, "kmeans_centroids.png"))
        print("saved kmeans_centroids.png")
    except Exception:
        print("sklearn not available, skipping KMeans baseline")

    # OnlineManifoldCluster (numpy) with max_centroids = num_clusters
    omc = OnlineManifoldCluster(dim=dim, max_centroids=num_clusters, distance="cosine", sliding_window_size=None)
    # add in one batch: this implementation will create up to max_centroids initial centroids
    omc.add(keys, weights=None, similarity_threshold=0.1)
    centroids_omc, counts = omc.get_centroids()
    if centroids_omc.shape[0] > 0:
        visualize_kv_and_centroids(keys, centroids_omc, save_path=os.path.join(out_dir, "omc_centroids.png"))
        print("saved omc_centroids.png")

    # OnlineManifoldClustering (streaming) - add in small chunks to encourage multiple centroids
    omc_stream = OnlineManifoldClustering(dim=dim, max_memory_size=10000, window_size=20, max_centroids= num_clusters, metric="cosine", similarity_threshold=0.6)
    chunk = 20
    for i in range(0, keys.shape[0], chunk):
        omc_stream.add(keys[i : i + chunk], np.zeros((min(chunk, keys.shape[0] - i), dim)))
    cent_s, counts_s, weights_s = omc_stream.get_centroids()
    if cent_s.shape[0] > 0:
        visualize_kv_and_centroids(keys, cent_s, save_path=os.path.join(out_dir, "omc_stream_centroids.png"))
        print("saved omc_stream_centroids.png")


if __name__ == "__main__":
    run_and_save()


