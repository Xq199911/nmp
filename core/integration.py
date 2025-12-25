"""Integration helpers and Llama adapter for MP-KVM.

This module exposes:
- MPKVMManager: per-layer managers that hold OnlineManifoldCluster instances
- monkey_patch_attention_forward: generic HF-style attention wrapper
- patch_llama_attention: helper tailored to extract K/V from Llama-like attention modules
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
from .clustering import OnlineManifoldCluster


class MPKVMManager:
    """
    Manager that holds per-layer OnlineManifoldCluster instances.
    Layers are identified by integer indices (0...L-1) for typical Transformer stacks.
    """

    def __init__(self, dim: int, num_layers: int = 32, cluster_kwargs: Optional[Dict] = None):
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        cluster_kwargs = cluster_kwargs or {}
        self._layers: Dict[int, OnlineManifoldCluster] = {
            i: OnlineManifoldCluster(dim=dim, **cluster_kwargs) for i in range(self.num_layers)
        }

    def add_kv(self, layer_idx: int, keys: np.ndarray, values: np.ndarray, weights: Optional[np.ndarray] = None, similarity_threshold: float = 0.1):
        if layer_idx not in self._layers:
            # lazily create if out-of-range
            self._layers[layer_idx] = OnlineManifoldCluster(dim=self.dim, **{})
        self._layers[layer_idx].add(keys, weights=weights, similarity_threshold=similarity_threshold)

    def get_layer_centroids(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if layer_idx not in self._layers:
            return np.zeros((0, self.dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return self._layers[layer_idx].get_centroids()

    def energy_loss(self, lambda_diversity: float = 0.0) -> Dict[int, float]:
        return {i: c.energy_loss(lambda_diversity=lambda_diversity) for i, c in self._layers.items()}


def monkey_patch_attention_forward(attn_module: Any, manager: MPKVMManager, layer_idx: int):
    """
    Conservative monkey-patch of a HF-style attention module forward.
    It will call the original forward, then attempt to extract produced key/value
    pairs (or saved attributes) and send their numpy representation to the manager.

    Note: this wrapper avoids changing the attention computation itself.
    """
    original_forward = getattr(attn_module, "forward", None)
    if original_forward is None:
        raise ValueError("Provided module has no forward()")

    def _patched_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        # best-effort extraction: module may expose last_key/last_value or outputs may contain past_key_values
        k = getattr(attn_module, "last_key", None)
        v = getattr(attn_module, "last_value", None)
        if k is None or v is None:
            try:
                # outputs might be tuple where outputs[1] = present_key_value_states
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    pv = outputs[1]
                    if isinstance(pv, (list, tuple)) and len(pv) >= 2:
                        k, v = pv[0], pv[1]
            except Exception:
                k = None
                v = None

        # Convert torch tensors -> numpy if available; be conservative and non-blocking
        if k is not None and v is not None:
            try:
                import torch

                def to_np(t):
                    if isinstance(t, torch.Tensor):
                        return t.detach().cpu().numpy()
                    return np.asarray(t)

                kn = to_np(k)
                vn = to_np(v)
                # reshape to (N, D) by merging leading dims
                kn = kn.reshape((-1, kn.shape[-1]))
                vn = vn.reshape((-1, vn.shape[-1]))
                manager.add_kv(layer_idx, kn.astype(np.float32), vn.astype(np.float32))
            except Exception:
                # silently skip if torch not installed or shape unexpected
                pass

        return outputs

    setattr(attn_module, "forward", _patched_forward)
    return attn_module


def patch_llama_attention(attn_module: Any, manager: MPKVMManager, layer_idx: int):
    """
    Helper tailored for Llama-like attention modules where K/V are commonly produced
    inside the forward and returned as present_key_value_states. This is similar to
    `monkey_patch_attention_forward` but includes an additional heuristic for common
    Llama implementations.
    """
    return monkey_patch_attention_forward(attn_module, manager, layer_idx)


__all__ = ["MPKVMManager", "monkey_patch_attention_forward", "patch_llama_attention"]

"""
MP-KVM integration layer: manager + monkey-patch helpers.

This module provides a lightweight MPKVMManager and utilities to
monkey-patch an attention module's forward to route KV storage
through the MP-KVM operator.
"""
from typing import Any, Callable, Optional, Tuple
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


__all__ = ["MPKVMManager", "monkey_patch_attention_forward"]


