"""
Compare generations and recorded attention between inject_on and inject_off runs.
Produces a small JSON report with text similarity and mean attention diff per layer.
"""
from __future__ import annotations
import os
import json
import numpy as np
from difflib import SequenceMatcher


def text_similarity(a: str, b: str) -> float:
    return float(SequenceMatcher(None, a, b).ratio())


def load_generations(path: str):
    try:
        with open(os.path.join(path, "generations.json"), "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj.get("generations", [])
    except Exception:
        return []


def mean_attention_diff(dir_a: str, dir_b: str):
    layers = []
    # find layer folders common to both
    la = [d for d in os.listdir(dir_a) if d.startswith("layer_")]
    lb = [d for d in os.listdir(dir_b) if d.startswith("layer_")]
    common = sorted(set(la) & set(lb))
    for layer in common:
        pa = os.path.join(dir_a, layer)
        pb = os.path.join(dir_b, layer)
        # load all attn_*.npy that exist in both
        fa = sorted([f for f in os.listdir(pa) if f.startswith("attn_") and f.endswith(".npy")])
        fb = sorted([f for f in os.listdir(pb) if f.startswith("attn_") and f.endswith(".npy")])
        common_files = sorted(set(fa) & set(fb))
        diffs = []
        for fn in common_files:
            try:
                a = np.load(os.path.join(pa, fn))
                b = np.load(os.path.join(pb, fn))
                if a.shape == b.shape:
                    diffs.append(float(np.mean(np.abs(a - b))))
            except Exception:
                continue
        layers.append({"layer": layer, "mean_abs_diff": float(np.mean(diffs)) if diffs else None, "num_compared": len(diffs)})
    return layers


def main(on_dir="experiments/realmodel_out/inject_on_attn", off_dir="experiments/realmodel_out/inject_off_attn"):
    gens_on = load_generations(on_dir)
    gens_off = load_generations(off_dir)
    prompt_results = []
    for i in range(min(len(gens_on), len(gens_off))):
        a = gens_on[i]["generation"]
        b = gens_off[i]["generation"]
        sim = text_similarity(a, b)
        prompt_results.append({"prompt": gens_on[i]["prompt"], "similarity": sim, "len_on": len(a), "len_off": len(b)})

    attn_layers = mean_attention_diff(on_dir, off_dir)

    out = {"prompts": prompt_results, "attention_layers": attn_layers}
    out_path = "experiments/realmodel_out/injection_comparison_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Wrote summary to", out_path)
    return out


if __name__ == "__main__":
    main()


