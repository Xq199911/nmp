"""
Adapter utilities to attach MP-KVM to HuggingFace-style Llama attention modules.

Usage:
    from core import MPKVMManager
    from adapters.llama_adapter import attach_mpkvm_to_hf_llama

    manager = MPKVMManager(dim=hidden_size, num_layers=model.config.num_hidden_layers, cluster_kwargs={...})
    attach_mpkvm_to_hf_llama(model, manager, head_mean=False, sample_stride=1)

This file provides a conservative wrapper that:
 - wraps per-layer attention `forward` methods,
 - extracts produced key/value tensors (best-effort heuristics),
 - optionally averages across heads or subsamples token dimension to reduce CPU-GPU copies,
 - converts tensors to numpy and calls manager.add_kv(layer_idx, keys, values).
"""
from __future__ import annotations
from typing import Any, Optional, Iterable, Set
import os
import numpy as np

from core.integration import MPKVMManager


def _to_numpy(tensor):
    try:
        import torch

        if isinstance(tensor, torch.Tensor):
            t = tensor.detach()
            # move to CPU
            try:
                t = t.cpu()
            except Exception:
                pass
            # cast half / bfloat16 to float32 for numpy conversion
            try:
                if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
                    t = t.to(dtype=torch.float32)
            except Exception:
                pass
            return t.numpy()
    except Exception:
        pass
    return np.asarray(tensor)


_DEBUG = os.getenv("MPKVM_DEBUG", "1") == "1"


def _process_kv_and_add(manager: MPKVMManager, layer_idx: int, k_tensor, v_tensor, head_mean: bool = False, sample_stride: int = 1):
    kn = _to_numpy(k_tensor)
    vn = _to_numpy(v_tensor)
    # expected shapes: (batch, seq, n_heads, head_dim) or (batch, seq, head_dim)
    if kn.ndim == 4:
        # (B, S, H, D_head)
        if head_mean:
            # average over heads -> (B, S, D_head)
            kn_proc = kn.mean(axis=2)
            vn_proc = vn.mean(axis=2)
        else:
            # reshape to (B*S*H, D_head)
            kn_proc = kn.reshape((-1, kn.shape[-1]))
            vn_proc = vn.reshape((-1, vn.shape[-1]))
    elif kn.ndim == 3:
        kn_proc = kn.reshape((-1, kn.shape[-1]))
        vn_proc = vn.reshape((-1, vn.shape[-1]))
    else:
        # fallback flatten last dim
        kn_proc = kn.reshape((-1, kn.shape[-1]))
        vn_proc = vn.reshape((-1, vn.shape[-1]))

    if sample_stride is not None and sample_stride > 1:
        kn_proc = kn_proc[::sample_stride]
        vn_proc = vn_proc[::sample_stride]

    # ensure float32
    kn_proc = kn_proc.astype(np.float32)
    vn_proc = vn_proc.astype(np.float32)
    try:
        # If manager supports GPU-accelerated aggregation, prefer that path
        if hasattr(manager, "add_kv_torch"):
            # attempt to pass original tensors if available
            try:
                if _DEBUG:
                    try:
                        kshape = getattr(k_tensor, "shape", None)
                        vshape = getattr(v_tensor, "shape", None)
                    except Exception:
                        kshape = None
                        vshape = None
                    print(f"[MPKVM][layer {layer_idx}] add_kv_torch called with shapes k={kshape} v={vshape}")
                manager.add_kv_torch(layer_idx, k_tensor, v_tensor)
            except Exception:
                if _DEBUG:
                    print(f"[MPKVM][layer {layer_idx}] add_kv_torch failed, falling back to add_kv (CPU path)")
                manager.add_kv(layer_idx, kn_proc, vn_proc)
        else:
            if _DEBUG:
                print(f"[MPKVM][layer {layer_idx}] add_kv (CPU) called with shapes k_proc={kn_proc.shape} v_proc={vn_proc.shape}")
            manager.add_kv(layer_idx, kn_proc, vn_proc)
    except Exception:
        # be defensive: do not raise from adapter
        if _DEBUG:
            import traceback
            print(f"[MPKVM][layer {layer_idx}] Exception in _process_kv_and_add:\n" + traceback.format_exc())
        pass


def attach_mpkvm_to_hf_llama(
    model: Any,
    manager: MPKVMManager,
    head_mean: bool = False,
    sample_stride: int = 1,
    enable_injection: Optional[bool] = None,
    max_injected_centroids: int = 256,
    per_layer_injection: Optional[Iterable[int]] = None,
    pass_centroid_weighting: bool = True,
    positionless_injection: bool = False,
):
    """
    Walk the model and attach wrappers to attention modules.
    This is heuristic and tries common HF Llama model layouts.
    """
    # resolve env vars / defaults
    if enable_injection is None:
        env_val = os.getenv("MPKVM_ENABLE_INJECTION", "1")
        enable_injection = False if env_val in ("0", "false", "False") else True
    try:
        max_injected_centroids = int(os.getenv("MPKVM_MAX_INJECTED_CENTROIDS", str(max_injected_centroids)))
    except Exception:
        pass
    per_layer_set: Optional[Set[int]] = None
    if per_layer_injection is not None:
        per_layer_set = set(int(x) for x in per_layer_injection)
    else:
        # env var like "0,1,2"
        env_layers = os.getenv("MPKVM_PER_LAYER_INJECTION", None)
        if env_layers:
            try:
                per_layer_set = set(int(x.strip()) for x in env_layers.split(",") if x.strip() != "")
            except Exception:
                per_layer_set = None
    # locate the module that holds transformer layers
    candidates = []
    if hasattr(model, "model"):
        candidates.append(model.model)
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    # fallback to model itself
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
        raise RuntimeError("Could not find transformer layers container in provided model.")

    # iterate layers and attach wrapper to common attention attribute names
    for idx, layer in enumerate(getattr(layers_container, "layers")):
        # common attr names
        attn_attr_names = ["self_attn", "attention", "attn", "qkv"]
        attn_module = None
        for name in attn_attr_names:
            if hasattr(layer, name):
                attn_module = getattr(layer, name)
                break
        if attn_module is None:
            # sometimes the attention module is nested; search attributes
            for attr_name in dir(layer):
                attr = getattr(layer, attr_name)
                if hasattr(attr, "forward") and "attn" in attr_name.lower():
                    attn_module = attr
                    break
        if attn_module is None:
            # skip if not found
            continue
        # debug: print module summary once (not per-forward) to avoid noise
        if _DEBUG:
            try:
                attrs = [a for a in dir(attn_module) if "key" in a.lower() or "value" in a.lower() or "proj" in a.lower() or "attn" in a.lower() or "present" in a.lower() or "last" in a.lower()]
                print(f"[MPKVM][layer {idx}] attaching to {type(attn_module).__name__}, candidate attrs: {attrs}")
            except Exception:
                pass

        # wrap forward
        orig_forward = getattr(attn_module, "forward")

        def make_wrapped(orig_forward, attn_module, layer_idx, enable_injection=enable_injection, max_injected_centroids=max_injected_centroids, per_layer_set=per_layer_set, pass_centroid_weighting=pass_centroid_weighting, positionless_injection=positionless_injection):
            def wrapped(*args, **kwargs):
                outputs = orig_forward(*args, **kwargs)
                # attempt to extract K/V from module attributes first
                k = getattr(attn_module, "last_key", None)
                v = getattr(attn_module, "last_value", None)
                # if not present, try to pull from outputs (present_key_value_states)
                if (k is None or v is None) and isinstance(outputs, tuple) and len(outputs) > 1:
                    pv = outputs[1]
                    try:
                        if isinstance(pv, (list, tuple)) and len(pv) >= 2:
                            k, v = pv[0], pv[1]
                    except Exception:
                        k = None
                        v = None
                # If k/v not found, attempt conservative probe: compute k/v from available projections
                if (k is None or v is None):
                    probed = False
                    # try to get a query/hidden_states from args or kwargs
                    query = None
                    if len(args) > 0:
                        query = args[0]
                    elif "hidden_states" in kwargs:
                        query = kwargs["hidden_states"]
                    # if attn_module exposes proj layers, try to project
                    try:
                        if query is not None and hasattr(attn_module, "k_proj") and hasattr(attn_module, "v_proj"):
                            try:
                                kp = getattr(attn_module, "k_proj")
                                vp = getattr(attn_module, "v_proj")
                                # compute k,v via projections (best-effort)
                                k_try = kp(query)
                                v_try = vp(query)
                                if k_try is not None and v_try is not None:
                                    k = k_try
                                    v = v_try
                                    probed = True
                                    if _DEBUG:
                                        try:
                                            print(f"[MPKVM][layer {layer_idx}] probe used k_proj/v_proj on query, shapes k={getattr(k,'shape',None)} v={getattr(v,'shape',None)}")
                                        except Exception:
                                            pass
                            except Exception:
                                # fallthrough
                                pass
                        # fallback: try applying q_proj/k_proj/v_proj where available
                        if not probed and query is not None and hasattr(attn_module, "q_proj") and hasattr(attn_module, "k_proj") and hasattr(attn_module, "v_proj"):
                            try:
                                q_try = attn_module.q_proj(query)
                                k_try = attn_module.k_proj(query)
                                v_try = attn_module.v_proj(query)
                                if k_try is not None and v_try is not None:
                                    k = k_try
                                    v = v_try
                                    probed = True
                                    if _DEBUG:
                                        print(f"[MPKVM][layer {layer_idx}] probe used q_proj/k_proj/v_proj on query")
                            except Exception:
                                pass
                    except Exception:
                        if _DEBUG:
                            import traceback
                            print(f"[MPKVM][layer {layer_idx}] probe failed:\n" + traceback.format_exc())

                # send KV to manager (prefer GPU path if available)
                if k is not None and v is not None:
                    if _DEBUG:
                        try:
                            kshape = getattr(k, "shape", None)
                            vshape = getattr(v, "shape", None)
                        except Exception:
                            kshape = None
                            vshape = None
                        print(f"[MPKVM][layer {layer_idx}] extracted k type={type(k).__name__ if k is not None else None} shape={kshape}  v type={type(v).__name__ if v is not None else None} shape={vshape}")
                    try:
                        _process_kv_and_add(manager, layer_idx, k, v, head_mean=head_mean, sample_stride=sample_stride)
                        if _DEBUG:
                            print(f"[MPKVM][layer {layer_idx}] _process_kv_and_add succeeded")
                    except Exception:
                        if _DEBUG:
                            import traceback
                            print(f"[MPKVM][layer {layer_idx}] _process_kv_and_add raised:\n" + traceback.format_exc())
                        pass

                # Attempt GPU-side centroid injection if enabled and allowed for this layer
                try:
                    if enable_injection and (per_layer_set is None or layer_idx in per_layer_set):
                        get_gpu_centroids = getattr(manager, "get_gpu_centroids", None)
                        if callable(get_gpu_centroids):
                            centroids, counts = get_gpu_centroids(layer_idx)
                            if centroids is not None:
                                try:
                                    import torch
                                    # select top-k centroids by counts if needed
                                    C = centroids.shape[0]
                                    if C > 0 and max_injected_centroids is not None and C > max_injected_centroids:
                                        if counts is not None:
                                            topk = torch.topk(counts, k=max_injected_centroids).indices
                                            centroids = centroids[topk]
                                        else:
                                            centroids = centroids[:max_injected_centroids]

                                    # try to extract query tensor (hidden_states -> q_proj) from args/attn_module
                                    query = None
                                    if len(args) > 0:
                                        query = args[0]
                                    elif "hidden_states" in kwargs:
                                        query = kwargs["hidden_states"]
                                    # attempt to get q by projecting query if q_proj exists
                                    q = None
                                    if query is not None and hasattr(attn_module, "q_proj"):
                                        try:
                                            q = attn_module.q_proj(query)
                                        except Exception:
                                            q = None
                                    # fallback: try to see if attn_module has 'last_query'
                                    if q is None:
                                        q = getattr(attn_module, "last_query", None)
                                    # if we have q, k, v as torch tensors, run ReconstructedAttentionTorch
                                    if q is not None and k is not None and v is not None:
                                        try:
                                            from core.layers import ReconstructedAttentionTorch
                                            # compute reconstructed attention output on device
                                            attn = ReconstructedAttentionTorch()
                                            # prepare centroid weighting if available and requested
                                            centroid_weighting = counts if pass_centroid_weighting and counts is not None else None
                                            new_out = attn(q, k, v, centroids_k=centroids, centroids_v=centroids, centroid_weighting=centroid_weighting, positionless=positionless_injection)
                                            # apply output projection if available
                                            out_proj = getattr(attn_module, "out_proj", None) or getattr(attn_module, "o_proj", None) or getattr(attn_module, "proj_out", None)
                                            if callable(out_proj):
                                                try:
                                                    new_out = out_proj(new_out)
                                                except Exception:
                                                    pass
                                            # place new_out into outputs (best-effort)
                                            if isinstance(outputs, tuple) and len(outputs) > 0:
                                                outputs = (new_out,) + tuple(outputs[1:])
                                            else:
                                                outputs = new_out
                                        except Exception:
                                            # if any step fails, fall back to original outputs
                                            pass
                                except Exception:
                                    # if any GPU selection step fails, skip injection
                                    pass
                except Exception:
                    pass

                return outputs
            return wrapped

        setattr(attn_module, "forward", make_wrapped(orig_forward, attn_module, idx))

    return model


