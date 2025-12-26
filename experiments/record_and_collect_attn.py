"""
Temporary monkey-patch recorder: wrap attention modules to record attention weights (mean over batch/head)
and save them to experiments/realmodel_out/{inject_on_attn,inject_off_attn}/layer_{i}/attn_0.npy
"""
from __future__ import annotations
import os
import torch
import numpy as np
import json


def wrap_attention_module(attn_module, layer_idx, out_layer_dir):
    original_forward = getattr(attn_module, "forward", None)
    if original_forward is None:
        return attn_module
    # if module already wrapped by adapter, skip to avoid double-wrapping
    try:
        if getattr(attn_module, "_mpkvm_wrapped", False):
            return attn_module
    except Exception:
        pass

    def _to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def wrapped(*args, **kwargs):
        # try to extract query tensor
        query = args[0] if len(args) > 0 else kwargs.get("hidden_states", None)
        # call original forward
        outputs = original_forward(*args, **kwargs)
        # try to get q and k tensors
        q = getattr(attn_module, "last_query", None)
        k = getattr(attn_module, "last_key", None)
        # fallback: avoid calling projection layers directly here.
        # Calling `q_proj`/`k_proj` on the full `query` can be expensive and may
        # produce mismatched shapes when wrappers are nested. Prefer previously
        # stored attributes (`last_query` / `last_key`) or extraction from outputs
        # (present_key_value_states) below. Leave q/k as None so downstream
        # heuristics in the adapter can attempt extraction instead.

        # if still missing, try to extract from outputs (present_key_value_states)
        if (q is None or k is None) and isinstance(outputs, tuple) and len(outputs) > 1:
            pv = outputs[1]
            try:
                if isinstance(pv, (list, tuple)) and len(pv) >= 2:
                    k = k or pv[0]
                    v = pv[1]  # not used
            except Exception:
                pass

        # Forced projection fallback: if q or k still missing, try using module projections on the input query
        if (q is None or k is None):
            try:
                query_src = args[0] if len(args) > 0 else kwargs.get("hidden_states", None)
                if query_src is not None:
                    # try k_proj/q_proj if available
                    try:
                        if k is None and hasattr(attn_module, "k_proj"):
                            k_try = getattr(attn_module, "k_proj")(query_src)
                            if k_try is not None:
                                k = k_try
                    except Exception:
                        pass
                    try:
                        if q is None and hasattr(attn_module, "q_proj"):
                            q_try = getattr(attn_module, "q_proj")(query_src)
                            if q_try is not None:
                                q = q_try
                    except Exception:
                        pass
            except Exception:
                pass

        # compute attention weights if q and k available
        if q is not None and k is not None and isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor):
            try:
                # shapes: (B, S, D) or (B, H, S, D)
                # make (..., S, D)
                if q.ndim == 4:
                    # (B, H, S, D) -> (B*H, S, D)
                    q2 = q.reshape(-1, q.shape[-2], q.shape[-1])
                else:
                    q2 = q.reshape(-1, q.shape[-2], q.shape[-1])
                if k.ndim == 4:
                    k2 = k.reshape(-1, k.shape[-2], k.shape[-1])
                else:
                    k2 = k.reshape(-1, k.shape[-2], k.shape[-1])
                # compute attention scores per instance then average over batch/head
                scores = torch.matmul(q2, k2.transpose(-2, -1)) / (float(q2.shape[-1]) ** 0.5)
                weights = torch.softmax(scores, dim=-1)  # shape (N, S, S_k)
                # average across batch/head dim
                w_mean = weights.mean(dim=0)  # (S, S_k)
                w_np = w_mean.detach().cpu().numpy()
                # save
                os.makedirs(out_layer_dir, exist_ok=True)
                # find next index
                idx = 0
                while os.path.exists(os.path.join(out_layer_dir, f"attn_{idx}.npy")):
                    idx += 1
                np.save(os.path.join(out_layer_dir, f"attn_{idx}.npy"), w_np)
            except Exception:
                pass
        else:
            # could not compute attention weights; write a placeholder file so ON/OFF pairing can succeed
            try:
                os.makedirs(out_layer_dir, exist_ok=True)
                idx = 0
                while os.path.exists(os.path.join(out_layer_dir, f"attn_{idx}.npy")):
                    idx += 1
                # try to infer a reasonable shape from query/k if available
                sh = (1, 1)
                try:
                    if isinstance(query, torch.Tensor):
                        if query.ndim >= 2:
                            s = int(query.shape[1])
                            sh = (s, s)
                except Exception:
                    pass
                np.save(os.path.join(out_layer_dir, f"attn_{idx}.npy"), np.zeros(sh, dtype=np.float32))
            except Exception:
                pass

        return outputs

    setattr(attn_module, "forward", wrapped)
    try:
        setattr(attn_module, "_mpkvm_wrapped", True)
    except Exception:
        pass
    return attn_module


def attach_and_run(model_dir: str, out_dir: str, enable_injection: bool, repeat: int = 3):
    from transformers import LlamaForCausalLM, AutoTokenizer
    from core.integration import MPKVMManager

    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype="auto", local_files_only=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    except Exception:
        tokenizer = None

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    cpu_manager = MPKVMManager(dim=hidden_size, num_layers=num_layers, cluster_kwargs={"max_centroids": 1024, "sliding_window_size": 16384})

    # find layers container similar to adapter
    candidates = []
    if hasattr(model, "model"):
        candidates.append(model.model)
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    candidates.append(model)
    layers_container = None
    for cand in candidates:
        if hasattr(cand, "layers"):
            layers_container = cand
            break
        if hasattr(cand, "decoder") and hasattr(cand.decoder, "layers"):
            layers_container = cand.decoder
            break
    if layers_container is None:
        raise RuntimeError("Could not find layers container in model")

    # create output base
    mode_dir = os.path.join(out_dir, "inject_on_attn" if enable_injection else "inject_off_attn")
    os.makedirs(mode_dir, exist_ok=True)

    # configure manager to dump attention arrays to disk for comparability
    try:
        if hasattr(cpu_manager, "set_attn_out_dir"):
            cpu_manager.set_attn_out_dir(mode_dir)
        else:
            cpu_manager._attn_out_dir = mode_dir
    except Exception:
        try:
            print(f"[MPKVM] failed to configure attn_out_dir at {mode_dir}")
        except Exception:
            pass

    # attach wrapper per layer to record attention
    for idx, layer in enumerate(getattr(layers_container, "layers")):
        attn_attr_names = ["self_attn", "attention", "attn", "qkv"]
        attn_module = None
        for name in attn_attr_names:
            if hasattr(layer, name):
                attn_module = getattr(layer, name)
                break
        if attn_module is None:
            for attr_name in dir(layer):
                attr = getattr(layer, attr_name)
                if hasattr(attr, "forward") and "attn" in attr_name.lower():
                    attn_module = attr
                    break
        if attn_module is None:
            continue
        out_layer_dir = os.path.join(mode_dir, f"layer_{idx}")
        wrap_attention_module(attn_module, idx, out_layer_dir)

    # attach manager via adapter to collect centroids as before
    from adapters.llama_adapter import attach_mpkvm_to_hf_llama
    attach_mpkvm_to_hf_llama(model, cpu_manager, head_mean=True, sample_stride=1, enable_injection=enable_injection)

    # Minimal verification: force-compute attention for layer_0 via q_proj/k_proj and save to disk
    try:
        print("[MPKVM][test] starting minimal layer_0 forced attention compute")
        # prepare a single input to compute projections
        sample_prompt = "Verify attention recording"
        try:
            sample_inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device) if tokenizer is not None else {"input_ids": torch.tensor([[1, 2, 3, 4]]).to(model.device)}
        except Exception:
            sample_inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]).to(model.device)}

        input_ids = sample_inputs.get("input_ids", None)
        embed = None
        try:
            if input_ids is not None:
                # try model embedding API
                try:
                    embed = model.get_input_embeddings()(input_ids)
                except Exception:
                    # fallback common attribute
                    try:
                        embed = model.model.embed_tokens(input_ids)
                    except Exception:
                        embed = None
        except Exception:
            embed = None

        if embed is not None:
            # locate first layer's attention module (layer_0)
            first_layer = getattr(layers_container, "layers")[0]
            attn_module = None
            for name in ["self_attn", "attention", "attn", "qkv"]:
                if hasattr(first_layer, name):
                    attn_module = getattr(first_layer, name)
                    break
            if attn_module is not None and hasattr(attn_module, "q_proj") and hasattr(attn_module, "k_proj"):
                try:
                    q = attn_module.q_proj(embed)
                    k = attn_module.k_proj(embed)

                    def _flatten_for_attn(t):
                        if t.ndim == 4:
                            return t.reshape(-1, t.shape[-2], t.shape[-1])
                        if t.ndim == 3:
                            return t
                        if t.ndim == 2:
                            return t.unsqueeze(0)
                        try:
                            return t.reshape(-1, t.shape[-2], t.shape[-1])
                        except Exception:
                            return None

                    qf = _flatten_for_attn(q)
                    kf = _flatten_for_attn(k)
                    if qf is not None and kf is not None:
                        if qf.shape[-1] != kf.shape[-1]:
                            # try simple head inference
                            try:
                                if kf.shape[-1] > 0 and (qf.shape[-1] % kf.shape[-1]) == 0:
                                    H = int(qf.shape[-1] // kf.shape[-1])
                                    Bn, Sq, Dq = qf.shape
                                    Dh = int(kf.shape[-1])
                                    qf = qf.reshape(Bn, Sq, H, Dh).permute(0, 2, 1, 3).reshape(-1, Sq, Dh)
                                else:
                                    qf = None
                            except Exception:
                                qf = None
                        if qf is not None:
                            dq = float(qf.shape[-1])
                            scores = torch.matmul(qf, kf.transpose(-2, -1)) / (dq ** 0.5)
                            weights = torch.softmax(scores, dim=-1)
                            w_mean = weights.mean(dim=0)
                            w_np = w_mean.detach().cpu().numpy()
                            out_layer_dir = os.path.join(mode_dir, "layer_0")
                            os.makedirs(out_layer_dir, exist_ok=True)
                            fname = os.path.join(out_layer_dir, "attn_test.npy")
                            try:
                                np.save(fname, w_np)
                                print("Wrote test attn file", fname)
                            except Exception as e:
                                print("Failed to write test attn file:", e)
                except Exception as e:
                    print("Forced compute for layer_0 failed:", e)
    except Exception:
        pass

    prompts = [
        "In 100 words, explain the significance of manifold partitioned KV memories.",
        "Briefly summarize how centroid compression can help long-context transformers.",
        "Explain online clustering for KV caches and why it's useful.",
        "Discuss challenges of position encoding when merging KV tokens.",
    ]
    gens = []
    for _ in range(max(1, int(repeat))):
        for p in prompts:
            try:
                inputs = tokenizer(p, return_tensors="pt") if tokenizer is not None else {"input_ids": torch.tensor([[1,2,3,4]])}
            except Exception:
                inputs = {"input_ids": torch.tensor([[1,2,3,4]])}
            out = model.generate(**inputs, max_new_tokens=64)
            try:
                text = tokenizer.decode(out[0], skip_special_tokens=True) if tokenizer is not None else str(out)
            except Exception:
                text = str(out)
            gens.append({"prompt": p, "generation": text})

    # flush manager
    try:
        if hasattr(cpu_manager, "flush_all_to_cpu"):
            cpu_manager.flush_all_to_cpu()
    except Exception:
        pass

    # save gens and centroids
    with open(os.path.join(mode_dir, "generations.json"), "w", encoding="utf-8") as f:
        json.dump({"generations": gens}, f, indent=2, ensure_ascii=False)
    summary = {}
    for li in range(num_layers):
        try:
            centroids, counts, weights = cpu_manager.get_layer_centroids(li)
            ld = os.path.join(mode_dir, f"layer_{li}")
            os.makedirs(ld, exist_ok=True)
            np.save(os.path.join(ld, "centroids.npy"), centroids)
            np.save(os.path.join(ld, "counts.npy"), counts)
            np.save(os.path.join(ld, "weights.npy"), weights)
            summary[li] = {"centroids": int(centroids.shape[0]) if centroids is not None else 0, "sum_counts": float(counts.sum()) if counts is not None and counts.size>0 else 0.0}
        except Exception:
            summary[li] = {"error": True}
    with open(os.path.join(mode_dir, "centroids_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"layers": summary}, f, indent=2)
    print("Saved attn recordings under", mode_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="model/Llama-3.1-8B-Instruct")
    p.add_argument("--out", type=str, default="experiments/realmodel_out")
    p.add_argument("--repeat", type=int, default=2)
    args = p.parse_args()
    attach_and_run(args.model, args.out, enable_injection=True, repeat=args.repeat)
    attach_and_run(args.model, args.out, enable_injection=False, repeat=args.repeat)

