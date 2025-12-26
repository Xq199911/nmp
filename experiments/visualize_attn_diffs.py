from __future__ import annotations
import os
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import json

BASE = "experiments/realmodel_out"
ON_DIR = os.path.join(BASE, "inject_on_attn")
OFF_DIR = os.path.join(BASE, "inject_off_attn")


def load_first_attn(layer_dir: str) -> Optional[np.ndarray]:
    if not os.path.isdir(layer_dir):
        return None
    files = sorted([f for f in os.listdir(layer_dir) if f.startswith("attn_") and f.endswith(".npy")])
    if not files:
        return None
    try:
        return np.load(os.path.join(layer_dir, files[0]))
    except Exception:
        return None


def collect_attn(base_dir: str) -> Dict[int, np.ndarray]:
    attn = {}
    if not os.path.isdir(base_dir):
        return attn
    for name in os.listdir(base_dir):
        if not name.startswith("layer_"):
            continue
        li = int(name.split("_")[1])
        arr = load_first_attn(os.path.join(base_dir, name))
        if arr is not None:
            attn[li] = arr
    return attn


def mean_abs_diffs(attn_on: dict, attn_off: dict):
    layers = sorted(set(list(attn_on.keys()) + list(attn_off.keys())))
    diffs = {}
    for li in layers:
        a = attn_on.get(li)
        b = attn_off.get(li)
        if a is None or b is None:
            diffs[li] = None
            continue
        try:
            min_shape = tuple(min(x, y) for x, y in zip(a.shape[-2:], b.shape[-2:]))
            a2 = a[..., :min_shape[0], :min_shape[1]]
            b2 = b[..., :min_shape[0], :min_shape[1]]
            d = np.mean(np.abs(a2 - b2))
            diffs[li] = float(d)
        except Exception:
            diffs[li] = None
    return diffs


def plot_bar(diffs: dict, out_png: str):
    items = sorted([(k, v) for k, v in diffs.items() if v is not None], key=lambda x: x[0])
    if not items:
        return None
    xs = [k for k, _ in items]
    ys = [v for _, v in items]
    plt.figure(figsize=(10, 4))
    plt.bar(xs, ys)
    plt.xlabel("layer")
    plt.ylabel("mean abs attn diff")
    plt.title("Per-layer mean absolute attention difference (on vs off)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png


def plot_heatmap(a, b, out_prefix):
    if a.ndim > 2:
        a = a[0]
    if b.ndim > 2:
        b = b[0]
    maxdim = 128
    a = a[:maxdim, :maxdim]
    b = b[:maxdim, :maxdim]
    diff = a - b
    vmin = min(a.min(), b.min())
    vmax = max(a.max(), b.max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(a, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("on")
    axes[1].imshow(b, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("off")
    axes[2].imshow(diff, cmap="bwr")
    axes[2].set_title("diff (on-off)")
    fig.colorbar(axes[2].images[0], ax=axes.ravel().tolist(), shrink=0.6)
    out = out_prefix + "_heatmaps.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def main():
    on = collect_attn(ON_DIR)
    off = collect_attn(OFF_DIR)
    diffs = mean_abs_diffs(on, off)
    os.makedirs(BASE, exist_ok=True)
    bar_png = os.path.join(BASE, "attn_mean_abs_diff_bar.png")
    plot_bar(diffs, bar_png)
    valid = [(k, v) for k, v in diffs.items() if v is not None]
    if not valid:
        print("No attention arrays recorded")
        return
    top = sorted(valid, key=lambda x: x[1], reverse=True)[:3]
    saved = {"bar": bar_png, "top_layers": []}
    for k, v in top:
        a = on.get(k)
        b = off.get(k)
        outp = os.path.join(BASE, f"layer_{k}_attn_heatmaps.png")
        plot_heatmap(a, b, os.path.join(BASE, f"layer_{k}_attn"))
        saved["top_layers"].append({"layer": k, "mean_abs_diff": v, "heatmap": outp})
    with open(os.path.join(BASE, "attn_diff_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"diffs": diffs, "saved": saved}, f, indent=2)
    print("Wrote attn diff plots and summary to", BASE)

"""
Visualize per-layer attention differences saved under experiments/realmodel_out/inject_on_attn and inject_off_attn.
Produces:
 - bar chart of mean absolute difference per layer
 - heatmaps for top-3 layers (on, off, diff)
"""




def load_first_attn(layer_dir: str):
    # find attn_*.npy under layer_dir; return first if exists
    files = sorted([f for f in os.listdir(layer_dir) if f.startswith("attn_") and f.endswith(".npy")])
    if not files:
        return None
    p = os.path.join(layer_dir, files[0])
    try:
        arr = np.load(p)
        return arr
    except Exception:
        return None


def collect_attn(base_dir: str):
    attn = {}
    if not os.path.isdir(base_dir):
        return attn
    for name in os.listdir(base_dir):
        if not name.startswith("layer_"):
            continue
        li = int(name.split("_")[1])
        layer_dir = os.path.join(base_dir, name)
        arr = load_first_attn(layer_dir)
        if arr is not None:
            attn[li] = arr
    return attn


def mean_abs_diffs(attn_on: dict, attn_off: dict):
    layers = sorted(set(list(attn_on.keys()) + list(attn_off.keys())))
    diffs = {}
    for li in layers:
        a = attn_on.get(li)
        b = attn_off.get(li)
        if a is None or b is None:
            diffs[li] = None
            continue
        # align shapes if needed: take first two dims (..., seq_q, seq_k)
        try:
            # broadcast to same shape or truncate to min shape
            min_shape = tuple(min(x, y) for x, y in zip(a.shape[-2:], b.shape[-2:]))
            a2 = a[..., :min_shape[0], :min_shape[1]]
            b2 = b[..., :min_shape[0], :min_shape[1]]
            d = np.mean(np.abs(a2 - b2))
            diffs[li] = float(d)
        except Exception:
            diffs[li] = None
    return diffs


def plot_bar(diffs: dict, out_png: str):
    items = sorted([(k, v) for k, v in diffs.items() if v is not None], key=lambda x: x[0])
    if not items:
        return None
    xs = [k for k, _ in items]
    ys = [v for _, v in items]
    plt.figure(figsize=(10, 4))
    plt.bar(xs, ys)
    plt.xlabel("layer")
    plt.ylabel("mean abs attn diff")
    plt.title("Per-layer mean absolute attention difference (on vs off)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png


def plot_heatmap(a, b, out_prefix):
    # a,b are 2D (seq_q, seq_k)
    if a.ndim > 2:
        a = a[0]
    if b.ndim > 2:
        b = b[0]
    # ensure small sizes for plotting
    maxdim = 128
    a = a[:maxdim, :maxdim]
    b = b[:maxdim, :maxdim]
    diff = a - b
    import matplotlib.pyplot as plt
    vmin = min(a.min(), b.min())
    vmax = max(a.max(), b.max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(a, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("on")
    im1 = axes[1].imshow(b, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("off")
    im2 = axes[2].imshow(diff, cmap="bwr")
    axes[2].set_title("diff (on-off)")
    fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.6)
    out = out_prefix + "_heatmaps.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def main():
    on = collect_attn(ON_DIR)
    off = collect_attn(OFF_DIR)
    diffs = mean_abs_diffs(on, off)
    os.makedirs(BASE, exist_ok=True)
    bar_png = os.path.join(BASE, "attn_mean_abs_diff_bar.png")
    plot_bar(diffs, bar_png)
    # pick top-3 layers by diff
    valid = [(k, v) for k, v in diffs.items() if v is not None]
    if not valid:
        print("No attention arrays recorded")
        return
    top = sorted(valid, key=lambda x: x[1], reverse=True)[:3]
    saved = {"bar": bar_png, "top_layers": []}
    for k, v in top:
        a = on.get(k)
        b = off.get(k)
        outp = os.path.join(BASE, f"layer_{k}_attn_heatmaps.png")
        plot_heatmap(a, b, os.path.join(BASE, f"layer_{k}_attn"))
        saved["top_layers"].append({"layer": k, "mean_abs_diff": v, "heatmap": outp})
    # write summary
    with open(os.path.join(BASE, "attn_diff_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"diffs": diffs, "saved": saved}, f, indent=2)
    print("Wrote attn diff plots and summary to", BASE)


if __name__ == "__main__":
    main()


