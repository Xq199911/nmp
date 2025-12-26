"""Clean MP-KVM integration layer: manager + monkey-patch helpers.

This module provides a lightweight MPKVMManager and utilities to
monkey-patch an attention module's forward to route KV storage
through the MP-KVM operator.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Dict

import numpy as np

from .clustering import OnlineManifoldClustering


class MPKVMManager:
    """
    Manager that holds per-layer clustering operators and exposes
    an API for attention layers to add KV vectors and retrieve centroids.
    """

    def __init__(self, dim: int, num_layers: int = 32, **cluster_kwargs):
        """
        Manager that holds per-layer clustering operators.
        Accepts `num_layers` for compatibility with older callers (also previously used `layers`).
        """
        self.layers = {}
        self.dim = dim
        self.num_layers = int(num_layers)
        # support callers passing a single `cluster_kwargs` dict (e.g., MPKVMManager(..., cluster_kwargs={...}))
        if "cluster_kwargs" in cluster_kwargs and isinstance(cluster_kwargs["cluster_kwargs"], dict):
            cluster_kwargs = cluster_kwargs["cluster_kwargs"]

        # normalize common naming differences to the clustering constructor
        # e.g., sliding_window_size -> window_size, max_centroids_per_layer -> max_centroids
        if "sliding_window_size" in cluster_kwargs:
            cluster_kwargs["window_size"] = cluster_kwargs.pop("sliding_window_size")
        if "max_centroids_per_layer" in cluster_kwargs:
            cluster_kwargs["max_centroids"] = cluster_kwargs.pop("max_centroids_per_layer")

        for l in range(self.num_layers):
            try:
                self.layers[l] = OnlineManifoldClustering(dim=dim, **cluster_kwargs)
            except TypeError:
                # defensive: if unexpected kwargs are present, filter to known params
                allowed = {"dim", "max_memory_size", "window_size", "max_centroids", "metric", "similarity_threshold"}
                filtered = {k: v for k, v in cluster_kwargs.items() if k in allowed}
                self.layers[l] = OnlineManifoldClustering(dim=dim, **filtered)
        # container to store traced attention weights per-layer (list of numpy arrays)
        self._attn_weights = {l: [] for l in range(self.num_layers)}
        # optional output directory for immediate attention dumps
        self._attn_out_dir: Optional[str] = None
        # per-layer counters for file naming
        self._attn_counters: Dict[int, int] = {}

    def set_attn_out_dir(self, path: str) -> None:
        """
        Configure a directory where attention numpy arrays will be written immediately.
        Creates per-layer subdirectories to avoid races later.
        """
        try:
            self._attn_out_dir = str(path)
            os.makedirs(self._attn_out_dir, exist_ok=True)
            for l in range(self.num_layers):
                try:
                    os.makedirs(os.path.join(self._attn_out_dir, f"layer_{l}"), exist_ok=True)
                except Exception:
                    # best-effort: continue creating remaining layers
                    pass
            # initialize counters if not present
            for l in range(self.num_layers):
                self._attn_counters.setdefault(l, 0)
        except Exception:
            try:
                print(f"[MPKVM] failed to set attn out dir to {path}")
            except Exception:
                pass

    def add_kv(self, layer_idx: int, keys: np.ndarray, values: np.ndarray, weights: Optional[np.ndarray] = None):
        # If keys provided, infer dimensionality and ensure layer cluster matches it.
        if keys is not None and hasattr(keys, "shape") and keys.ndim == 2:
            key_dim = int(keys.shape[1])
        else:
            key_dim = self.dim

        if layer_idx not in self.layers:
            # lazily create cluster operator with inferred key dim
            self.layers[layer_idx] = OnlineManifoldClustering(dim=key_dim, **{})
        else:
            # if existing cluster dim mismatches incoming keys, reinit that layer's cluster to match keys
            existing = self.layers[layer_idx]
            try:
                existing_dim = int(existing.dim)
            except Exception:
                existing_dim = self.dim
            if existing_dim != key_dim:
                # replace with a new cluster matching the incoming key dimensionality
                self.layers[layer_idx] = OnlineManifoldClustering(dim=key_dim, **{})

        # add to the layer's cluster (weights may be None)
        self.layers[layer_idx].add(keys, values, weights)

    def get_layer_centroids(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if layer_idx not in self.layers:
            return np.zeros((0, self.dim), dtype=np.float32), np.array([], dtype=int), np.array([], dtype=float)
        return self.layers[layer_idx].get_centroids()

    def record_attention(self, layer_idx: int, attn_weights_np: np.ndarray) -> None:
        """
        Record attention weights (numpy) for the given layer.
        attn_weights_np expected shape: (..., seq_q, seq_k) or (seq_q, seq_k)
        """
        # Debug entry log to help trace why recordings may be missing.
        try:
            print(f"[MPKVM][layer {layer_idx}] record_attention called; out_dir={getattr(self, '_attn_out_dir', None)}")
        except Exception:
            pass
        if layer_idx not in self._attn_weights:
            self._attn_weights[layer_idx] = []
        try:
            arr = np.asarray(attn_weights_np)
            while arr.ndim > 2:
                arr = arr.mean(axis=0)
            arr = arr.astype(np.float32)
            self._attn_weights[layer_idx].append(arr)
            # if output dir configured, write file immediately for ON/OFF pairing
            if getattr(self, "_attn_out_dir", None):
                out_dir = os.path.join(self._attn_out_dir, f"layer_{layer_idx}")
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    cnt = self._attn_counters.get(layer_idx, 0)
                    fname = os.path.join(out_dir, f"attn_{cnt:06d}.npy")
                    np.save(fname, arr)
                    self._attn_counters[layer_idx] = cnt + 1
                except Exception:
                    # log write failure for debugging but continue
                    try:
                        import traceback

                        print(f"[MPKVM][layer {layer_idx}] failed writing attn file: {traceback.format_exc()}")
                    except Exception:
                        pass
        except Exception:
            # Try a more defensive conversion and record any errors for debugging.
            try:
                arr = np.asarray(attn_weights_np, dtype=np.float32)
                while arr.ndim > 2:
                    arr = arr.mean(axis=0)
                self._attn_weights[layer_idx].append(arr)
                # attempt to write placeholder file if out_dir set to help pairing
                if getattr(self, "_attn_out_dir", None):
                    out_dir = os.path.join(self._attn_out_dir, f"layer_{layer_idx}")
                    try:
                        os.makedirs(out_dir, exist_ok=True)
                        cnt = self._attn_counters.get(layer_idx, 0)
                        fname = os.path.join(out_dir, f"attn_{cnt:06d}.npy")
                        np.save(fname, arr)
                        self._attn_counters[layer_idx] = cnt + 1
                    except Exception:
                        try:
                            import traceback

                            print(f"[MPKVM][layer {layer_idx}] failed writing fallback attn file: {traceback.format_exc()}")
                        except Exception:
                            pass
            except Exception:
                try:
                    import traceback

                    print(f"[MPKVM][layer {layer_idx}] record_attention conversion failed: {traceback.format_exc()}")
                except Exception:
                    pass

    def get_recorded_attention(self, layer_idx: int):
        """Return list of recorded attention numpy arrays for layer or empty list."""
        return list(self._attn_weights.get(layer_idx, []))


def monkey_patch_attention_forward(attn_module: Any, manager: MPKVMManager, layer_idx: int):
    """
    Monkey-patches a huggingface-style attention module instance.
    The attn_module is expected to have a `forward` method with signature
    (hidden_states, past_key_value=None, attention_mask=None, *args, **kwargs)
    and to produce `key`, `value` tensors or expose them as attributes.

    This function wraps the forward to intercept the produced KV and pass to manager.add_kv.
    It's intentionally conservative and will not modify attention math itself.
    """

    original_forward = getattr(attn_module, "forward")

    def _patched_forward(*args, **kwargs):
        # call original forward, capture outputs
        outputs = original_forward(*args, **kwargs)
        # best-effort extraction of k/v from module attributes or outputs
        try:
            # huggingface often stores k/v as attn_module.k_proj(...), attn_module.v_proj(...)
            k = getattr(attn_module, "last_key", None)
            v = getattr(attn_module, "last_value", None)
        except Exception:
            k = None
            v = None

        # If not available on module, try to extract from outputs (tuple)
        if k is None or v is None:
            if isinstance(outputs, tuple) and len(outputs) > 0:
                # heuristic: outputs[1] could be present_key_value
                pv = outputs[1] if len(outputs) > 1 else None
                if pv is not None and isinstance(pv, (list, tuple)) and len(pv) >= 2:
                    k, v = pv[0], pv[1]

        # Convert to numpy if tensors (avoid torch import at top-level)
        try:
            import torch

            def to_np(t):
                if isinstance(t, torch.Tensor):
                    return t.detach().cpu().numpy()
                return np.asarray(t)

            if k is not None and v is not None:
                kn = to_np(k.reshape(-1, k.shape[-1]))
                vn = to_np(v.reshape(-1, v.shape[-1]))
                manager.add_kv(layer_idx, kn, vn)
        except Exception:
            # silent: if torch not available or shapes unexpected, skip storage
            pass

        return outputs

    setattr(attn_module, "forward", _patched_forward)
    return attn_module


def patch_llama_attention(attn_module: Any, manager: MPKVMManager, layer_idx: int):
    """
    Compatibility wrapper (keeps previous API) that calls the generic monkey-patch.
    """
    return monkey_patch_attention_forward(attn_module, manager, layer_idx)


__all__ = ["MPKVMManager", "monkey_patch_attention_forward", "patch_llama_attention"]


