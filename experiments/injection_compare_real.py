"""
Run real model generation ON/OFF injection and save generations, centroids and attention.
This requires transformers and a local model at `model_dir`.
"""
from __future__ import annotations
import os
import json
import numpy as np

def run_and_save(model_dir: str, out_dir: str, enable_injection: bool, repeat: int = 3):
    os.makedirs(out_dir, exist_ok=True)
    try:
        from transformers import LlamaForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers required for real model run") from e

    from core.integration import MPKVMManager
    from adapters.llama_adapter import attach_mpkvm_to_hf_llama
    import torch

    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype="auto", local_files_only=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    except Exception:
        tokenizer = None

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    mgr = MPKVMManager(dim=hidden_size, num_layers=num_layers, cluster_kwargs={"max_centroids": 1024, "sliding_window_size": 16384})
    # configure manager to dump attention arrays to disk for comparability
    try:
        if hasattr(mgr, "set_attn_out_dir"):
            mgr.set_attn_out_dir(out_dir)
        else:
            mgr._attn_out_dir = out_dir
    except Exception:
        try:
            print(f"[MPKVM] failed to configure attn_out_dir at {out_dir}")
        except Exception:
            pass
    attach_mpkvm_to_hf_llama(model, mgr, head_mean=True, sample_stride=1, enable_injection=enable_injection)

    prompts = [
        "In 100 words, explain the significance of manifold partitioned KV memories.",
        "Briefly summarize how centroid compression can help long-context transformers.",
    ]
    gens = []
    for _ in range(max(1, int(repeat))):
        for p in prompts:
            try:
                inputs = tokenizer(p, return_tensors="pt").to(model.device) if tokenizer is not None else {"input_ids": torch.tensor([[1,2,3,4]]).to(model.device)}
            except Exception:
                inputs = {"input_ids": torch.tensor([[1,2,3,4]]).to(model.device)}
            out = model.generate(**inputs, max_new_tokens=64)
            try:
                text = tokenizer.decode(out[0], skip_special_tokens=True) if tokenizer is not None else str(out)
            except Exception:
                text = str(out)
            gens.append({"prompt": p, "generation": text})

    # save gens
    with open(os.path.join(out_dir, "generations.json"), "w", encoding="utf-8") as f:
        json.dump({"generations": gens}, f, indent=2, ensure_ascii=False)

    # save centroids and recorded attention
    summary = {}
    for li in range(num_layers):
        try:
            centroids, counts, weights = mgr.get_layer_centroids(li)
            ld = os.path.join(out_dir, f"layer_{li}")
            os.makedirs(ld, exist_ok=True)
            np.save(os.path.join(ld, "centroids.npy"), centroids)
            np.save(os.path.join(ld, "counts.npy"), counts)
            np.save(os.path.join(ld, "weights.npy"), weights)
            summary[li] = {"centroids": int(centroids.shape[0]) if centroids is not None else 0}
        except Exception:
            summary[li] = {"error": True}

    with open(os.path.join(out_dir, "centroids_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"layers": summary}, f, indent=2)

    # save attention arrays
    for li in range(num_layers):
        attn_list = mgr.get_recorded_attention(li)
        if not attn_list:
            continue
        ld = os.path.join(out_dir, f"layer_{li}")
        os.makedirs(ld, exist_ok=True)
        for idx, arr in enumerate(attn_list):
            try:
                np.save(os.path.join(ld, f"attn_{idx}.npy"), arr)
            except Exception:
                pass

    return out_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="model/Llama-3.1-8B-Instruct")
    p.add_argument("--out", type=str, default="experiments/realmodel_out")
    p.add_argument("--repeat", type=int, default=1)
    args = p.parse_args()
    on_dir = os.path.join(args.out, "inject_on_attn")
    off_dir = os.path.join(args.out, "inject_off_attn")
    print("running ON..."); run_and_save(args.model, on_dir, True, args.repeat)
    print("running OFF..."); run_and_save(args.model, off_dir, False, args.repeat)
    print("done")


