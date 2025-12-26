"""
Plot per-layer mean attention absolute-difference heatmap from injection comparison summary.
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main(summary_path="experiments/realmodel_out/injection_comparison_summary.json", out_path="experiments/realmodel_out/attention_diff_heatmap.png"):
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        print("no summary found at", summary_path)
        return
    layers = data.get("attention_layers", [])
    if not layers:
        print("no attention_layers in summary")
        return
    vals = []
    labels = []
    for l in layers:
        v = l.get("mean_abs_diff")
        vals.append(np.nan if v is None else float(v))
        labels.append(l.get("layer"))
    arr = np.array(vals).reshape(1, -1)
    plt.figure(figsize=(max(6, len(labels)*0.2), 3))
    # mask NaNs
    mask = np.isnan(arr)
    cmap = plt.cm.viridis
    cmap.set_bad(color="lightgray")
    plt.imshow(arr, aspect="auto", cmap=cmap)
    plt.colorbar(label="mean abs attention diff")
    plt.yticks([])
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.title("Per-layer mean abs attention difference (ON vs OFF)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print("wrote", out_path)

if __name__ == "__main__":
    main()


