import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_result_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_needles_summary(result_json_path: str, out_dir: str):
    res = load_result_json(result_json_path)
    params = res.get("params", {})
    recall = res.get("recall", 0.0)
    num_centroids = res.get("num_centroids", 0)

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["recall", "num_centroids"], [recall, num_centroids], color=["#2ca02c", "#1f77b4"])
    ax.set_ylabel("Value")
    ax.set_title("Needles experiment summary")
    out_path = os.path.join(out_dir, "needles_summary.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_centroids_embedding(centroids_path: str, out_dir: str, method: str = "pca"):
    if not os.path.exists(centroids_path):
        raise FileNotFoundError(centroids_path)
    centroids = np.load(centroids_path)
    if centroids.size == 0:
        return None
    os.makedirs(out_dir, exist_ok=True)
    if method == "umap":
        try:
            import umap

            emb = umap.UMAP(n_components=2).fit_transform(centroids)
        except Exception:
            method = "pca"
    if method == "pca":
        pca = PCA(n_components=2)
        emb = pca.fit_transform(centroids)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.8)
    ax.set_title("Centroids embedding (" + method + ")")
    out_path = os.path.join(out_dir, "centroids_embedding.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_all_plots(result_json_path: str, out_dir: str):
    # produce summary and centroids embedding if available
    summary_path = plot_needles_summary(result_json_path, out_dir)
    cent_path = os.path.join(os.path.dirname(result_json_path), "centroids.npy")
    emb_path = None
    try:
        emb_path = plot_centroids_embedding(cent_path, out_dir)
    except Exception:
        emb_path = None
    return {"summary": summary_path, "embedding": emb_path}


