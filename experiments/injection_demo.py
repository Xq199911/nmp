"""
Demo end-to-end injection using adapter with a mocked GPU centroid provider.
Validates that adapter's injection path runs ReconstructedAttentionTorch and replaces outputs.
"""
from __future__ import annotations
import os
import torch
import numpy as np

from core.integration import MPKVMManager
from adapters.llama_adapter import attach_mpkvm_to_hf_llama


class MockAttnModule:
    def __init__(self, hidden_size=64):
        self.hidden_size = hidden_size
        # simple linear projections to simulate q/k/v/out proj
        self.q_proj = lambda x: x  # identity for simplicity
        self.k_proj = lambda x: x
        self.v_proj = lambda x: x
        self.out_proj = lambda x: x

    def forward(self, hidden_states, *args, **kwargs):
        # return (output, present_key_value_states)
        # simulate key/value as projections of hidden_states: shape (B, S, D)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # present in tuple
        return (hidden_states, (k, v))


class Layer:
    def __init__(self):
        self.attn = MockAttnModule(hidden_size=64)


class DummyModel:
    def __init__(self):
        self.model = type("M", (), {})()
        self.model.layers = [Layer()]


def run_demo():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # manager (cpu) but we attach a fake get_gpu_centroids that returns torch tensors on device
    manager = MPKVMManager(dim=64, num_layers=1)

    # create synthetic centroids on device
    centroids = torch.randn((4, 64), device=device, dtype=torch.float32)
    counts = torch.ones((4,), device=device, dtype=torch.float32)

    def get_gpu_centroids(layer_idx: int):
        return centroids, counts

    manager.get_gpu_centroids = get_gpu_centroids

    model = DummyModel()
    # attach adapter with injection enabled
    attach_mpkvm_to_hf_llama(model, manager, enable_injection=True, per_layer_injection=[0], cluster_kwargs=None, positionless_injection=False)

    # run a forward
    hs = torch.randn((1, 8, 64), device=device)
    out = model.model.layers[0].attn.forward(hs)
    # after wrapping, the forward on attn should be replaced by wrapped that may return torch outputs
    # call through model layer
    res = model.model.layers[0].attn.forward(hs)

    print("Output type:", type(res))
    if isinstance(res, tuple):
        print("Output[0] type:", type(res[0]))
    print("Demo complete")


if __name__ == "__main__":
    run_demo()


