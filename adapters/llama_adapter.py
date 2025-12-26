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
from core.clustering import OnlineManifoldClustering


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
    # normalize to numpy where possible but keep original tensors for possible GPU path
    kn = _to_numpy(k_tensor)
    vn = _to_numpy(v_tensor)

    # expected shapes: (batch, seq, n_heads, head_dim) or (batch, seq, head_dim) or already flattened (N, D)
    def _reshape_proc(arr):
        if arr.ndim == 4:
            # (B, S, H, D_head)
            if head_mean:
                return arr.mean(axis=2).reshape((-1, arr.shape[-1]))
            return arr.reshape((-1, arr.shape[-1]))
        elif arr.ndim == 3:
            # (B, S, D) or (B, S, H) after head_mean handled above
            return arr.reshape((-1, arr.shape[-1]))
        elif arr.ndim == 2:
            return arr
        else:
            return arr.reshape((-1, arr.shape[-1]))

    kn_proc = _reshape_proc(kn)
    vn_proc = _reshape_proc(vn)

    if sample_stride is not None and sample_stride > 1:
        kn_proc = kn_proc[::sample_stride]
        vn_proc = vn_proc[::sample_stride]

    # ensure float32 numpy for CPU path
    try:
        kn_proc = kn_proc.astype(np.float32)
        vn_proc = vn_proc.astype(np.float32)
    except Exception:
        kn_proc = np.asarray(kn_proc, dtype=np.float32)
        vn_proc = np.asarray(vn_proc, dtype=np.float32)

    # prefer GPU aggregator if available and original tensors look like torch tensors
    try:
        if hasattr(manager, "add_kv_torch"):
            try:
                # pass through original tensors first to avoid CPU<->GPU copies
                manager.add_kv_torch(layer_idx, k_tensor, v_tensor)
                if _DEBUG:
                    print(f"[MPKVM][layer {layer_idx}] add_kv_torch succeeded")
                return
            except Exception:
                if _DEBUG:
                    print(f"[MPKVM][layer {layer_idx}] add_kv_torch failed, falling back to CPU path")
                # fallthrough to CPU path

        # CPU path
        if _DEBUG:
            try:
                print(f"[MPKVM][layer {layer_idx}] add_kv (CPU) called with shapes k_proc={kn_proc.shape} v_proc={vn_proc.shape}")
            except Exception:
                pass
        manager.add_kv(layer_idx, kn_proc, vn_proc)
    except Exception:
        if _DEBUG:
            import traceback

            print(f"[MPKVM][layer {layer_idx}] Exception in _process_kv_and_add:\n" + traceback.format_exc())
        # never raise from adapter
        return


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
    cluster_kwargs: Optional[dict] = None,
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

        # wrap forward with more robust KV extraction heuristics
        orig_forward = getattr(attn_module, "forward")

        def make_wrapped(orig_forward, attn_module, layer_idx, enable_injection=enable_injection, max_injected_centroids=max_injected_centroids, per_layer_set=per_layer_set, pass_centroid_weighting=pass_centroid_weighting, positionless_injection=positionless_injection):
            def wrapped(*args, **kwargs):
                outputs = orig_forward(*args, **kwargs)

                # 1) Try module attributes (common names)
                candidates = ["last_key", "last_value", "key", "value", "present_key", "present_value", "present_key_value_states", "present"]
                k = None
                v = None
                for name in candidates:
                    try:
                        attr = getattr(attn_module, name, None)
                        if attr is not None:
                            # some attrs may be tuples/lists (k,v)
                            if isinstance(attr, (list, tuple)) and len(attr) >= 2:
                                k, v = attr[0], attr[1]
                                break
                            # else try to infer by name
                            if "key" in name and k is None:
                                k = attr
                            if "value" in name and v is None:
                                v = attr
                            if k is not None and v is not None:
                                break
                    except Exception:
                        continue

                # 2) Try outputs (many HF models return present_key_value_states or tuple)
                if (k is None or v is None) and isinstance(outputs, tuple) and len(outputs) > 1:
                    pv = outputs[1]
                    try:
                        # present might be (k, v) or list of layers; handle several shapes
                        if isinstance(pv, (list, tuple)) and len(pv) >= 2:
                            # pv could be ((k_layer,...),(v_layer,...)) or (k,v)
                            first = pv[0]
                            second = pv[1]
                            # if first/second are tensors or arrays, assign
                            k = k or first
                            v = v or second
                        elif hasattr(pv, "present_key_values") or hasattr(pv, "past_key_values"):
                            # model-specific container; attempt attribute access
                            try:
                                p = getattr(pv, "present_key_values", None) or getattr(pv, "past_key_values", None)
                                if isinstance(p, (list, tuple)) and len(p) >= 2:
                                    k = k or p[0]
                                    v = v or p[1]
                            except Exception:
                                pass
                    except Exception:
                        pass

                # 3) Try kwargs like past_key_values
                if (k is None or v is None) and "past_key_values" in kwargs:
                    p = kwargs.get("past_key_values", None)
                    try:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            k = k or p[0]
                            v = v or p[1]
                    except Exception:
                        pass

                # 4) Conservative probe: use projection layers if exposed
                if (k is None or v is None):
                    query = None
                    if len(args) > 0:
                        query = args[0]
                    elif "hidden_states" in kwargs:
                        query = kwargs["hidden_states"]
                    try:
                        if query is not None:
                            if hasattr(attn_module, "k_proj") and hasattr(attn_module, "v_proj"):
                                try:
                                    k_try = getattr(attn_module, "k_proj")(query)
                                    v_try = getattr(attn_module, "v_proj")(query)
                                    if k_try is not None and v_try is not None:
                                        k = k or k_try
                                        v = v or v_try
                                except Exception:
                                    pass
                            if (k is None or v is None) and hasattr(attn_module, "q_proj") and hasattr(attn_module, "k_proj") and hasattr(attn_module, "v_proj"):
                                try:
                                    q_try = attn_module.q_proj(query)
                                    k_try = attn_module.k_proj(query)
                                    v_try = attn_module.v_proj(query)
                                    k = k or k_try
                                    v = v or v_try
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
                        # attempt to record attention weights if q and k are available as torch tensors
                        try:
                            import torch
                            q_tensor = None
                            # prefer explicit query variable if present
                            if 'query' in locals() and isinstance(query, torch.Tensor):
                                q_tensor = query
                            # fallback to last_query attr
                            if q_tensor is None:
                                q_tensor = getattr(attn_module, "last_query", None)
                            # if still None, try to extract from args/kwargs (hidden_states)
                            if q_tensor is None:
                                try:
                                    if len(args) > 0 and isinstance(args[0], torch.Tensor):
                                        q_tensor = args[0]
                                    elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                                        q_tensor = kwargs["hidden_states"]
                                except Exception:
                                    q_tensor = None
                            # if q_proj exists, try projecting a query to get proper shape
                            if q_tensor is not None and not hasattr(q_tensor, "shape") and hasattr(attn_module, "q_proj"):
                                try:
                                    q_tensor = attn_module.q_proj(q_tensor)
                                except Exception:
                                    pass
                            # ensure q and k are torch tensors and compute attention weights
                            if q_tensor is not None and isinstance(q_tensor, torch.Tensor) and isinstance(k, torch.Tensor):
                                try:
                                    # Normalize q/k shapes to (N, S, D) where N can be B*H if needed.
                                    def _flatten_for_attn(t):
                                        # (B, H, S, D) -> (B*H, S, D)
                                        if t.ndim == 4:
                                            return t.reshape(-1, t.shape[-2], t.shape[-1])
                                        # (B, S, D) -> (B, S, D)
                                        if t.ndim == 3:
                                            return t
                                        # (S, D) or (N, D) -> make batch dim
                                        if t.ndim == 2:
                                            return t.unsqueeze(0)
                                        # fallback: try to reshape to (..., S, D)
                                        try:
                                            return t.reshape(-1, t.shape[-2], t.shape[-1])
                                        except Exception:
                                            return None

                                    qf = _flatten_for_attn(q_tensor)
                                    kf = _flatten_for_attn(k)

                                    if qf is None or kf is None:
                                        raise RuntimeError("could not normalize q/k tensors for attention recording")

                                    # If last-dim mismatch, try to split q into heads using module attributes
                                    if qf.shape[-1] != kf.shape[-1]:
                                        # prefer num_key_value_heads or num_key_value_groups if available
                                        n_kv_heads = getattr(attn_module, "num_key_value_heads", None) or getattr(attn_module, "num_key_value_groups", None)
                                        handled = False
                                        H = None
                                        if n_kv_heads is not None and int(n_kv_heads) > 0 and qf.shape[-1] == kf.shape[-1] * int(n_kv_heads):
                                            H = int(n_kv_heads)
                                            handled = True
                                        else:
                                            # fallback: if q dim is an integer multiple of k dim, infer number of q-heads
                                            if kf.shape[-1] > 0 and (qf.shape[-1] % kf.shape[-1]) == 0:
                                                H = int(qf.shape[-1] // kf.shape[-1])
                                                handled = True

                                        if handled and H is not None:
                                            # First try to align batch-like dims by repeating k across head groups if needed
                                            try:
                                                # If kff batch axis is smaller by factor H, expand it
                                                if kf.shape[0] * H == qf.shape[0]:
                                                    try:
                                                        kf = kf.repeat_interleave(H, dim=0)
                                                        if _DEBUG:
                                                            print(f"[MPKVM][layer {layer_idx}] expanded kf by repeat_interleave, new batch dim {kf.shape[0]}")
                                                    except Exception:
                                                        # fallback to repeat
                                                        kf = kf.repeat(H, 1, 1)
                                                        if _DEBUG:
                                                            print(f"[MPKVM][layer {layer_idx}] expanded kf by repeat, new batch dim {kf.shape[0]}")
                                                # If kf already matches qf batch dim, nothing to do
                                                if qf.shape[0] == kf.shape[0] and qf.shape[-1] == kf.shape[-1]:
                                                    # perfectly aligned already
                                                    pass
                                                else:
                                                    # attempt reshaping qf into (Bn, Sq, H, Dh) as previous logic
                                                    Bn, Sq, Dq = qf.shape
                                                    Dh = int(kf.shape[-1])
                                                    try:
                                                        qf = qf.reshape(Bn, Sq, H, Dh).permute(0, 2, 1, 3).reshape(-1, Sq, Dh)
                                                    except Exception:
                                                        # if reshape fails, mark as incompatible
                                                        qf = None
                                                        if _DEBUG:
                                                            print(f"[MPKVM][layer {layer_idx}] failed to reshape qf for GQA handling qf.shape={qf} kf.shape={kf.shape}")
                                            except Exception:
                                                qf = None
                                                if _DEBUG:
                                                    import traceback

                                                    print(f"[MPKVM][layer {layer_idx}] GQA alignment failed: {traceback.format_exc()}")
                                        else:
                                            # incompatible shapes; skip recording to avoid expensive errors
                                            raise RuntimeError(f"incompatible q/k head dims q={qf.shape} k={kf.shape}")

                                    # Now compute attention scores and weights
                                    dq = float(qf.shape[-1])
                                    scores = torch.matmul(qf, kf.transpose(-2, -1)) / (dq ** 0.5)
                                    weights = torch.softmax(scores, dim=-1)
                                    # record as numpy
                                    try:
                                        manager.record_attention(layer_idx, weights.detach().cpu().numpy())
                                        if _DEBUG:
                                            print(f"[MPKVM][layer {layer_idx}] recorded attention weights shape={getattr(weights,'shape',None)}")
                                    except Exception:
                                        # best-effort: convert then record
                                        try:
                                            manager.record_attention(layer_idx, weights.detach().cpu().numpy())
                                        except Exception:
                                            if _DEBUG:
                                                print(f"[MPKVM][layer {layer_idx}] failed to record attention weights")
                                except Exception:
                                    if _DEBUG:
                                        import traceback
                                        print(f"[MPKVM][layer {layer_idx}] failed computing attention weights: {traceback.format_exc()}")
                        except Exception:
                            # ignore any failures in recording to avoid breaking forward
                            pass
                        if _DEBUG:
                            print(f"[MPKVM][layer {layer_idx}] _process_kv_and_add succeeded")
                    except Exception:
                        if _DEBUG:
                            import traceback

                            print(f"[MPKVM][layer {layer_idx}] _process_kv_and_add raised:\n" + traceback.format_exc())
                        pass

                # Forced-projection fallback: if attention weights were not recorded above,
                # try to compute q and k via projection layers (q_proj/k_proj) from hidden_states
                # and record the resulting attention. This helps when module attributes like
                # last_query/last_key are not populated.
                try:
                    if _DEBUG:
                        print(f"[MPKVM][layer {layer_idx}] attempting forced projection fallback for attention recording")
                    import torch
                    # obtain a candidate query source
                    query_src = None
                    if len(args) > 0:
                        query_src = args[0]
                    elif "hidden_states" in kwargs:
                        query_src = kwargs["hidden_states"]

                    q_proj = getattr(attn_module, "q_proj", None)
                    k_proj = getattr(attn_module, "k_proj", None)

                    qf = None
                    kf = None
                    if query_src is not None:
                        try:
                            if callable(q_proj):
                                q_try = q_proj(query_src)
                                if isinstance(q_try, torch.Tensor):
                                    qf = q_try
                            if callable(k_proj):
                                k_try = k_proj(query_src)
                                if isinstance(k_try, torch.Tensor):
                                    kf = k_try
                        except Exception:
                            qf = None
                            kf = None

                    # fallback to existing tensors if projection didn't yield tensors
                    if kf is None and isinstance(k, torch.Tensor):
                        kf = k
                    if qf is None:
                        qf = getattr(attn_module, "last_query", None)

                    if isinstance(qf, torch.Tensor) and isinstance(kf, torch.Tensor):
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

                        qff = _flatten_for_attn(qf)
                        kff = _flatten_for_attn(kf)
                        if qff is not None and kff is not None:
                            # handle possible head-dim mismatch
                            if qff.shape[-1] != kff.shape[-1]:
                                n_kv_heads = getattr(attn_module, "num_key_value_heads", None) or getattr(attn_module, "num_key_value_groups", None)
                                handled = False
                                H = None
                                if n_kv_heads is not None and int(n_kv_heads) > 0 and qff.shape[-1] == kff.shape[-1] * int(n_kv_heads):
                                    H = int(n_kv_heads)
                                    handled = True
                                else:
                                    if kff.shape[-1] > 0 and (qff.shape[-1] % kff.shape[-1]) == 0:
                                        H = int(qff.shape[-1] // kff.shape[-1])
                                        handled = True
                                if handled and H is not None:
                                    # Try to align batch-like dims by repeating kff across head groups if necessary
                                    try:
                                        if kff.shape[0] * H == qff.shape[0]:
                                            try:
                                                kff = kff.repeat_interleave(H, dim=0)
                                                if _DEBUG:
                                                    print(f"[MPKVM][layer {layer_idx}] expanded kff by repeat_interleave to match qff batch dim {kff.shape[0]}")
                                            except Exception:
                                                kff = kff.repeat(H, 1, 1)
                                                if _DEBUG:
                                                    print(f"[MPKVM][layer {layer_idx}] expanded kff by repeat fallback to match qff batch dim {kff.shape[0]}")
                                        # attempt to reshape qff into separate head dim if needed
                                        if qff.shape[-1] == kff.shape[-1] * H:
                                            Bn, Sq, Dq = qff.shape
                                            Dh = int(kff.shape[-1])
                                            try:
                                                qff = qff.reshape(Bn, Sq, H, Dh).permute(0, 2, 1, 3).reshape(-1, Sq, Dh)
                                            except Exception:
                                                qff = None
                                                if _DEBUG:
                                                    print(f"[MPKVM][layer {layer_idx}] failed to reshape qff for GQA handling; qff.shape={Bn,Sq,Dq} H={H} Dh={Dh}")
                                    except Exception:
                                        qff = None
                                        if _DEBUG:
                                            import traceback
                                            print(f"[MPKVM][layer {layer_idx}] GQA alignment failed: {traceback.format_exc()}")
                                else:
                                    qff = None
                            if qff is not None:
                                dq = float(qff.shape[-1])
                                scores = torch.matmul(qff, kff.transpose(-2, -1)) / (dq ** 0.5)
                                weights = torch.softmax(scores, dim=-1)
                                try:
                                    manager.record_attention(layer_idx, weights.detach().cpu().numpy())
                                    if _DEBUG:
                                        print(f"[MPKVM][layer {layer_idx}] forced-proj recorded attention shape={getattr(weights,'shape',None)}")
                                except Exception:
                                    if _DEBUG:
                                        print(f"[MPKVM][layer {layer_idx}] forced-proj failed to save attention via manager.record_attention, attempting direct save")
                                    # try direct save to manager._attn_out_dir as a fallback to isolate issue
                                    try:
                                        import time
                                        w_np = weights.detach().cpu().numpy()
                                        out_base = getattr(manager, "_attn_out_dir", None)
                                        if out_base:
                                            out_layer_dir = os.path.join(out_base, f"layer_{layer_idx}")
                                            os.makedirs(out_layer_dir, exist_ok=True)
                                            fname = os.path.join(out_layer_dir, f"attn_forced_{int(time.time())}.npy")
                                            try:
                                                np.save(fname, w_np)
                                                if _DEBUG:
                                                    print(f"[MPKVM][layer {layer_idx}] forced-proj direct-saved attention to {fname}")
                                            except Exception:
                                                if _DEBUG:
                                                    import traceback
                                                    print(f"[MPKVM][layer {layer_idx}] forced-proj direct save failed: {traceback.format_exc()}")
                                    except Exception:
                                        if _DEBUG:
                                            try:
                                                import traceback
                                                print(f"[MPKVM][layer {layer_idx}] forced-proj direct-save fallback failed: {traceback.format_exc()}")
                                            except Exception:
                                                pass
                except Exception:
                    if _DEBUG:
                        try:
                            import traceback

                            print(f"[MPKVM][layer {layer_idx}] forced projection fallback failed: {traceback.format_exc()}")
                        except Exception:
                            pass

                # Forced projection fallback: if previous recording attempts failed, try to
                # compute q/k via q_proj/k_proj from hidden_states (safe, guarded) and record.
                try:
                    if _DEBUG:
                        print(f"[MPKVM][layer {layer_idx}] attempting forced projection fallback for attention recording")
                    import torch

                    # attempt to locate a query source
                    query_src = None
                    if len(args) > 0:
                        query_src = args[0]
                    elif "hidden_states" in kwargs:
                        query_src = kwargs["hidden_states"]

                    qf = None
                    kf = None
                    # If projection modules exist, try to apply them to query_src
                    q_proj = getattr(attn_module, "q_proj", None)
                    k_proj = getattr(attn_module, "k_proj", None)
                    if query_src is not None:
                        try:
                            if callable(q_proj):
                                q_try = q_proj(query_src)
                                if isinstance(q_try, torch.Tensor):
                                    qf = q_try
                            if callable(k_proj):
                                k_try = k_proj(query_src)
                                if isinstance(k_try, torch.Tensor):
                                    kf = k_try
                        except Exception:
                            qf = None
                            kf = None

                    # fallback to available tensors
                    if kf is None and isinstance(k, torch.Tensor):
                        kf = k
                    # q_tensor may have been set above; try to reuse
                    q_tensor_local = None
                    try:
                        q_tensor_local = locals().get("q_tensor", None)
                    except Exception:
                        q_tensor_local = None
                    if qf is None and isinstance(q_tensor_local, torch.Tensor):
                        qf = q_tensor_local

                    # compute attention if we have tensors
                    if isinstance(qf, torch.Tensor) and isinstance(kf, torch.Tensor):
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

                        qff = _flatten_for_attn(qf)
                        kff = _flatten_for_attn(kf)
                        if qff is not None and kff is not None:
                            # handle head mismatch heuristics (same as above)
                            if qff.shape[-1] != kff.shape[-1]:
                                n_kv_heads = getattr(attn_module, "num_key_value_heads", None) or getattr(attn_module, "num_key_value_groups", None)
                                handled = False
                                if n_kv_heads is not None and int(n_kv_heads) > 0 and qff.shape[-1] == kff.shape[-1] * int(n_kv_heads):
                                    H = int(n_kv_heads)
                                    handled = True
                                else:
                                    if kff.shape[-1] > 0 and (qff.shape[-1] % kff.shape[-1]) == 0:
                                        H = int(qff.shape[-1] // kff.shape[-1])
                                        handled = True
                                if handled:
                                    Bn, Sq, Dq = qff.shape
                                    Dh = int(kff.shape[-1])
                                    qff = qff.reshape(Bn, Sq, H, Dh).permute(0, 2, 1, 3).reshape(-1, Sq, Dh)
                                else:
                                    qff = None
                            if qff is not None:
                                dq = float(qff.shape[-1])
                                scores = torch.matmul(qff, kff.transpose(-2, -1)) / (dq ** 0.5)
                                weights = torch.softmax(scores, dim=-1)
                                try:
                                    manager.record_attention(layer_idx, weights.detach().cpu().numpy())
                                    if _DEBUG:
                                        print(f"[MPKVM][layer {layer_idx}] forced-proj recorded attention shape={getattr(weights,'shape',None)}")
                                except Exception:
                                    if _DEBUG:
                                        print(f"[MPKVM][layer {layer_idx}] forced-proj failed to save attention")
                except Exception:
                    if _DEBUG:
                        try:
                            import traceback

                            print(f"[MPKVM][layer {layer_idx}] forced projection fallback failed: {traceback.format_exc()}")
                        except Exception:
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

        # If cluster_kwargs provided and manager has per-layer storage, attempt to (re)initialize the layer's cluster operator.
        try:
            if cluster_kwargs is not None and hasattr(manager, "layers"):
                try:
                    existing = None
                    try:
                        existing = manager.layers.get(idx, None)
                    except Exception:
                        existing = None
                    dim = None
                    if existing is not None:
                        try:
                            dim = int(existing.dim)
                        except Exception:
                            dim = None
                    if dim is None:
                        # try to infer from model config
                        dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None) or getattr(model.config, "n_embd", None)
                    if dim is None:
                        dim = 1024
                    try:
                        manager.layers[idx] = OnlineManifoldClustering(dim=int(dim), **cluster_kwargs)
                    except Exception:
                        # best-effort: ignore failures
                        pass
                except Exception:
                    pass
        except Exception:
            pass

        setattr(attn_module, "forward", make_wrapped(orig_forward, attn_module, idx))
        # mark module as wrapped to avoid double-wrapping by other tools/scripts
        try:
            setattr(attn_module, "_mpkvm_wrapped", True)
        except Exception:
            pass

    return model


