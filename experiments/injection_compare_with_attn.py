"""
Compare model outputs and recorded attention weights with and without centroid injection.
Uses HF model if available and configured, else falls back to a mock attention module.
"""
from __future__ import annotations
import os
import json
import numpy as np

try:
    import torch
except Exception:
    torch = None

from core.integration import MPKVMManager
from adapters.llama_adapter import attach_mpkvm_to_hf_llama


class MockAttn:
    def __init__(self, dim=64):
        self.dim = dim
        self.q_proj = lambda x: x
        self.k_proj = lambda x: x
        self.v_proj = lambda x: x
        self.out_proj = lambda x: x

    def forward(self, hidden_states, *args, **kwargs):
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # return present tuple (k,v)
        return (hidden_states, (k, v))


class Layer:
    def __init__(self, dim=64):
        self.attn = MockAttn(dim=dim)


class DummyModel:
    def __init__(self, dim=64):
        self.model = type("M", (), {})()
        self.model.layers = [Layer(dim=dim)]


def avg_attn_diff(attn_list_a, attn_list_b):
    # attn_list: list of numpy arrays
    if len(attn_list_a) == 0 or len(attn_list_b) == 0:
        return None
    # pairwise average absolute diff over matching shapes (truncate to min len)
    L = min(len(attn_list_a), len(attn_list_b))
    diffs = []
    for i in range(L):
        a = attn_list_a[i]
        b = attn_list_b[i]
        if a.shape != b.shape:
            # try to broadcast or skip
            continue
        diffs.append(float(np.mean(np.abs(a - b))))
    return float(np.mean(diffs)) if diffs else None


def run_demo():
    device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
    # manager and model
    manager_no = MPKVMManager(dim=64, num_layers=1, cluster_kwargs={})
    model = DummyModel(dim=64)
    attach_mpkvm_to_hf_llama(model, manager_no, enable_injection=False, cluster_kwargs=None)

    # run few forwards
    hs = np.random.randn(1, 8, 64).astype(np.float32)
    # call forward
    _ = model.model.layers[0].attn.forward(hs)
    attn_no = manager_no.get_recorded_attention(0)

    # with injection: provide GPU centroids via manager.get_gpu_centroids or manager.layers
    manager_yes = MPKVMManager(dim=64, num_layers=1, cluster_kwargs={"init_preserve_first_n": 2, "similarity_threshold": 0.4})
    # create some centroids on GPU (if torch) or CPU numpy and register via get_gpu_centroids
    if torch is not None and torch.cuda.is_available():
        cent = torch.randn((4, 64), device=device)
        counts = torch.ones((4,), device=device)
        manager_yes.get_gpu_centroids = lambda layer_idx: (cent, counts)
    else:
        # leave manager.layers populated via cluster_kwargs
        pass
    attach_mpkvm_to_hf_llama(model, manager_yes, enable_injection=True, per_layer_injection=[0], cluster_kwargs={"init_preserve_first_n": 2, "similarity_threshold": 0.4})
    _ = model.model.layers[0].attn.forward(hs)
    attn_yes = manager_yes.get_recorded_attention(0)

    diff = avg_attn_diff(attn_no, attn_yes)
    out = {"attn_mean_abs_diff": diff, "attn_no_count": len(attn_no), "attn_yes_count": len(attn_yes)}
    out_dir = os.path.join("experiments", "injection_compare_out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "injection_attn_diff.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved injection attn diff to", out_dir)
    print(out)


if __name__ == "__main__":
    run_demo()
 


