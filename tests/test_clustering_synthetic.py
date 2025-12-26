import numpy as np
from core.clustering import OnlineManifoldClustering
from analysis.energy_loss import assign_by_cosine, reconstruction_loss


def generate_synthetic_clusters(num_clusters: int, points_per_cluster: int, dim: int, cluster_std: float = 0.05, seed: int = 42):
    """
    Generate synthetic clustered data: `num_clusters` gaussian clusters in `dim`-D.
    Returns keys: (N, D) and dummy values: (N, D)
    """
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_clusters, dim)
    keys_list = []
    for c in centers:
        pts = c + rng.normal(scale=cluster_std, size=(points_per_cluster, dim))
        keys_list.append(pts.astype(np.float32))
    keys = np.vstack(keys_list)
    values = np.zeros_like(keys, dtype=np.float32)
    return keys, values


def test_online_clustering_reconstructs_better_than_random():
    """
    Basic sanity test: clustering-based centroids should produce lower reconstruction loss
    than choosing the same number of centroids at random from the data.
    """
    np.random.seed(42)
    num_clusters = 5
    points_per_cluster = 120
    dim = 32

    # generate data
    keys, values = generate_synthetic_clusters(num_clusters=num_clusters, points_per_cluster=points_per_cluster, dim=dim, cluster_std=0.06, seed=42)

    # create online clustering operator with a small window to trigger local compression
    n = keys.shape[0]
    # set window_size smaller than n so _compress_oldest_batch runs during add()
    clustering = OnlineManifoldClustering(dim=dim, max_memory_size=10000, window_size=50, max_centroids= num_clusters, metric="cosine", similarity_threshold=0.8)

    # ingest all keys (values are unused for distance)
    clustering.add(keys, values)

    centroids, counts, weights = clustering.get_centroids()
    # ensure we got some centroids
    assert centroids.shape[0] > 0

    # compute reconstruction using assignments by cosine
    assignments = assign_by_cosine(keys, centroids)
    clustering_loss = reconstruction_loss(keys, assignments, centroids)

    # baseline: pick random points as centroids (same count)
    rng = np.random.RandomState(123)
    k = centroids.shape[0]
    idx = rng.choice(keys.shape[0], size=k, replace=False)
    random_centroids = keys[idx]
    assignments_rand = assign_by_cosine(keys, random_centroids)
    random_loss = reconstruction_loss(keys, assignments_rand, random_centroids)

    # clustering should beat random baseline
    assert clustering_loss < random_loss, f"clustering_loss={clustering_loss} should be < random_loss={random_loss}"


def test_online_manifold_cluster_weighted_mean_and_counts():
    """
    Verify that OnlineManifoldCluster produces weighted centroid equal to weighted mean
    and that centroid counts reflect summed weights when vectors are merged.
    """
    import numpy as np
    from core.clustering import OnlineManifoldCluster

    dim = 4
    v1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    v2 = np.array([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    w1 = 2.0
    w2 = 3.0

    cl = OnlineManifoldCluster(dim=dim, max_centroids=10, distance="euclidean", sliding_window_size=None)
    # add vectors one-by-one so the second add can merge into the existing centroid
    cl.add(v1, weights=np.array([w1], dtype=np.float32), similarity_threshold=1e6)
    cl.add(v2, weights=np.array([w2], dtype=np.float32), similarity_threshold=1e6)

    centroids, counts = cl.get_centroids()
    assert centroids.shape[0] == 1, f"expected single centroid after forced merge, got {centroids.shape[0]}"
    expected = (v1[0] * w1 + v2[0] * w2) / (w1 + w2)
    # centroid vector
    cent = centroids[0]
    assert np.allclose(cent, expected, atol=1e-5), f"centroid {cent} != expected {expected}"
    # counts should sum to total weight
    assert np.isclose(counts.sum(), w1 + w2, atol=1e-5), f"counts.sum()={counts.sum()} != {w1 + w2}"


