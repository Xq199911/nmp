"""
Check ON/OFF attention .npy files per-layer for presence and shape compatibility.
Writes a JSON report and prints layers with mismatched counts or shapes.
"""
from __future__ import annotations
import os
import json
import numpy as np
from glob import glob

def inspect_dir(base_dir: str):
    layers = {}
    for p in glob(os.path.join(base_dir, "layer_*")):
        if not os.path.isdir(p):
            continue
        name = os.path.basename(p)
        files = sorted(glob(os.path.join(p, "attn_*.npy")))
        shapes = []
        for f in files:
            try:
                a = np.load(f)
                shapes.append(tuple(a.shape))
            except Exception:
                shapes.append(None)
        layers[name] = {"count": len(files), "shapes": shapes}
    return layers

def compare(on_dir: str, off_dir: str, out_path: str = "experiments/realmodel_out/attn_pair_report.json"):
    on = inspect_dir(on_dir)
    off = inspect_dir(off_dir)
    all_layers = sorted(set(list(on.keys()) + list(off.keys())))
    report = {}
    mismatched = {"count_diff": [], "shape_mismatch": []}
    for layer in all_layers:
        on_info = on.get(layer, {"count":0, "shapes":[]})
        off_info = off.get(layer, {"count":0, "shapes":[]})
        report[layer] = {"on_count": on_info["count"], "off_count": off_info["count"], "on_shapes": on_info["shapes"], "off_shapes": off_info["shapes"]}
        if on_info["count"] != off_info["count"]:
            mismatched["count_diff"].append(layer)
        else:
            # check shapes pairwise
            for i, (sa, sb) in enumerate(zip(on_info["shapes"], off_info["shapes"])):
                if sa != sb:
                    mismatched["shape_mismatch"].append({"layer": layer, "index": i, "on_shape": sa, "off_shape": sb})
                    break
    # save report
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"report": report, "mismatched": mismatched}, f, indent=2)
    print("Wrote report to", out_path)
    print("Layers with count differences:", mismatched["count_diff"])
    print("Number of shape mismatches:", len(mismatched["shape_mismatch"]))
    return report, mismatched


def _ensure_layer_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _pad_missing_files(side_dir: str, layer_name: str, target_count: int, example_shape):
    """
    Create placeholder attn_{idx}.npy files under side_dir/layer_name until count == target_count.
    example_shape should be a tuple describing the (S, S_k) shape to create zeros for.
    """
    layer_dir = os.path.join(side_dir, layer_name)
    _ensure_layer_dir(layer_dir)
    # collect existing indices
    existing = sorted([f for f in os.listdir(layer_dir) if f.startswith("attn_") and f.endswith(".npy")])
    existing_idxs = set()
    for fn in existing:
        try:
            num = int(fn.split("_")[1].split(".")[0])
            existing_idxs.add(num)
        except Exception:
            continue
    # determine next free index starting from 0
    next_idx = 0
    while next_idx in existing_idxs:
        next_idx += 1
    # create files until we reach target_count
    current_count = len(existing_idxs)
    to_create = max(0, target_count - current_count)
    for i in range(to_create):
        fname = os.path.join(layer_dir, f"attn_{next_idx:06d}.npy")
        try:
            arr = np.zeros(example_shape, dtype=np.float32)
            np.save(fname, arr)
        except Exception:
            # best-effort: if saving fails, try a minimal shape
            try:
                np.save(fname, np.zeros((1, 1), dtype=np.float32))
            except Exception:
                pass
        next_idx += 1


def pad_pairs(on_dir: str, off_dir: str, out_path: str = "experiments/realmodel_out/attn_pair_report.json"):
    """
    Inspect ON/OFF directories and pad the side with fewer files per-layer by creating
    zero-filled .npy placeholders to match counts. Uses the other side's first valid
    shape as the example; falls back to (1,1).
    """
    on = inspect_dir(on_dir)
    off = inspect_dir(off_dir)
    all_layers = sorted(set(list(on.keys()) + list(off.keys())))
    report = {}
    for layer in all_layers:
        on_info = on.get(layer, {"count": 0, "shapes": []})
        off_info = off.get(layer, {"count": 0, "shapes": []})
        # determine target count (max of the two)
        target = max(on_info["count"], off_info["count"])
        # find example shape to use when padding: prefer the other side's first non-None shape
        example_shape = None
        for s in on_info["shapes"] + off_info["shapes"]:
            if s is not None and isinstance(s, (tuple, list)) and len(s) == 2:
                example_shape = tuple(s)
                break
        if example_shape is None:
            example_shape = (1, 1)
        # pad on side
        if on_info["count"] < target:
            _pad_missing_files(on_dir, layer, target, example_shape)
        if off_info["count"] < target:
            _pad_missing_files(off_dir, layer, target, example_shape)
        report[layer] = {"on_count": target, "off_count": target}
    # write report
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"report": report}, f, indent=2)
    print("Padded missing files and wrote report to", out_path)
    return report

if __name__ == "__main__":
    on_dir = "experiments/realmodel_out/inject_on_attn"
    off_dir = "experiments/realmodel_out/inject_off_attn"
    compare(on_dir, off_dir)

import os
import json
import numpy as np

BASE = "experiments/realmodel_out"
ON = os.path.join(BASE, "inject_on_attn")
OFF = os.path.join(BASE, "inject_off_attn")

report = {"layers": {}}

def list_attn(path):
    out = {}
    if not os.path.isdir(path):
        return out
    for d in sorted(os.listdir(path)):
        ld = os.path.join(path, d)
        if not os.path.isdir(ld):
            continue
        files = sorted([f for f in os.listdir(ld) if f.startswith("attn_") and f.endswith(".npy")])
        out[d] = files
    return out

on_list = list_attn(ON)
off_list = list_attn(OFF)

layers = sorted(set(list(on_list.keys()) + list(off_list.keys())))

for layer in layers:
    on_files = on_list.get(layer, [])
    off_files = off_list.get(layer, [])
    common = sorted(set(on_files) & set(off_files))
    info = {"on_count": len(on_files), "off_count": len(off_files), "common_count": len(common), "mismatched_files": []}
    diffs = []
    for fn in common:
        try:
            a = np.load(os.path.join(ON, layer, fn))
            b = np.load(os.path.join(OFF, layer, fn))
            if a.shape != b.shape:
                info["mismatched_files"].append(fn)
                continue
            diffs.append(float(np.mean(np.abs(a - b))))
        except Exception as e:
            info.setdefault("errors", []).append(str(e))
    info["mean_abs_diff"] = float(np.mean(diffs)) if diffs else None
    report["layers"][layer] = info

out_path = os.path.join(BASE, "attn_pair_report.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("Wrote", out_path)
print("Summary:")
for layer, info in report["layers"].items():
    print(layer, "on", info["on_count"], "off", info["off_count"], "common", info["common_count"], "mean_diff", info["mean_abs_diff"])


