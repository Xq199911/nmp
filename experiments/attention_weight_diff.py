"""
Collect per-layer recorded attention weights for inject_on/off and compute differences.
"""
from __future__ import annotations
import os
import numpy as np
import json

def load_attn_list(base_dir: str, kind: str):
    path = os.path.join(base_dir, kind)
    attn = {}
    for d in os.listdir(path):
        if d.startswith("layer_"):
            li = int(d.split("_")[1])
            # try to load any saved attention arrays if present in manager recording export
            # we saved manager recordings under layer dirs named 'attn_*.npy' optionally
            ad = os.path.join(path, d)
            arrs = []
            for f in os.listdir(ad):
                if f.startswith("attn_") and f.endswith(".npy"):
                    arrs.append(np.load(os.path.join(ad, f)))
            if arrs:
                attn[li] = arrs
    return attn

def compute_mean_abs_diff(attn_on, attn_off):
    # attn_on/off: dict layer->list of arrays
    diffs = {}
    layers = sorted(set(list(attn_on.keys()) + list(attn_off.keys())))
    for li in layers:
        A = attn_on.get(li, [])
        B = attn_off.get(li, [])
        if not A or not B:
            diffs[li] = None
            continue
        # compute mean pairwise abs diff between first arrays
        a = A[0]
        b = B[0]
        # broadcast if necessary
        try:
            d = np.mean(np.abs(a - b))
            diffs[li] = float(d)
        except Exception:
            diffs[li] = None
    return diffs

def main():
    base = "experiments/realmodel_out"
    attn_on = load_attn_list(base, "inject_on")
    attn_off = load_attn_list(base, "inject_off")
    diffs = compute_mean_abs_diff(attn_on, attn_off)
    out = os.path.join(base, "attention_weight_diffs.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(diffs, f, indent=2)
    print("Wrote attention diffs to", out)

if __name__ == "__main__":
    main()


