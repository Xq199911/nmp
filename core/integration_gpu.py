"""
GPU-side lightweight KV aggregator for MP-KVM.

Design:
- Maintain per-layer GPU centroids as summed key/value tensors and total weights (torch tensors on device).
- When the number of GPU centroids for a layer exceeds a threshold, flush aggregated centroids to a CPU MPKVMManager
  by converting to numpy and calling its `add_kv`.
- This minimizes frequent CPU-GPU copies by performing local merging on-device.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from typing import Tuple


class MPKVMGPUAggregator:
    def __init__(
        self,
        cpu_manager: Any,
        dim: int,
        device: Optional[str] = None,
        max_gpu_centroids_per_layer: int = 512,
        similarity_threshold: float = 0.1,
        head_mean: bool = False,
        sample_stride: int = 1,
    ):
        """
        cpu_manager: an instance of core.integration.MPKVMManager (expects numpy add_kv)
        dim: hidden dimension
        device: torch device string (e.g., 'cuda:0') or None (auto)
        """
        self.cpu_manager = cpu_manager
        self.dim = int(dim)
        self.max_gpu_centroids_per_layer = int(max_gpu_centroids_per_layer)
        self.similarity_threshold = float(similarity_threshold)
        self.head_mean = bool(head_mean)
        self.sample_stride = int(sample_stride)
        self.device = device

        # per-layer storage: maps layer_idx -> list of sum_k (torch), sum_v (torch), count (float)
        self._layers: Dict[int, Dict[str, List]] = {}

    def _ensure_layer(self, layer_idx: int):
        if layer_idx not in self._layers:
            self._layers[layer_idx] = {"sum_k": [], "sum_v": [], "count": []}

    def add_kv_torch(self, layer_idx: int, k_tensor, v_tensor, weights: Optional[Any] = None):
        """
        Accept torch tensors for K and V and perform on-device reduction.
        Expected k_tensor shape: (B, S, H, D_head) or (B, S, D)
        This function will perform head-averaging if configured and flatten tokens.
        """
        try:
            import torch
        except Exception:
            # torch not available; fall back to CPU path by converting to numpy
            kn = np.asarray(k_tensor)
            vn = np.asarray(v_tensor)
            self.cpu_manager.add_kv(layer_idx, kn.reshape((-1, kn.shape[-1])), vn.reshape((-1, vn.shape[-1])))
            return

        # move to configured device if needed
        dev = torch.device(self.device) if self.device is not None else k_tensor.device
        if k_tensor.device != dev:
            k_tensor = k_tensor.to(dev)
        if v_tensor.device != dev:
            v_tensor = v_tensor.to(dev)

        # convert shape
        if k_tensor.ndim == 4:
            # (B, S, H, D)
            if self.head_mean:
                k_proc = k_tensor.mean(dim=2)  # (B, S, D)
                v_proc = v_tensor.mean(dim=2)
            else:
                # reshape to (B*S*H, D)
                k_proc = k_tensor.reshape(-1, k_tensor.shape[-1])
                v_proc = v_tensor.reshape(-1, v_tensor.shape[-1])
        elif k_tensor.ndim == 3:
            k_proc = k_tensor.reshape(-1, k_tensor.shape[-1])
            v_proc = v_tensor.reshape(-1, v_tensor.shape[-1])
        else:
            k_proc = k_tensor.reshape(-1, k_tensor.shape[-1])
            v_proc = v_tensor.reshape(-1, v_tensor.shape[-1])

        # optional subsampling
        if self.sample_stride is not None and self.sample_stride > 1:
            k_proc = k_proc[:: self.sample_stride]
            v_proc = v_proc[:: self.sample_stride]

        self._ensure_layer(layer_idx)
        layer_storage = self._layers[layer_idx]
        sum_k_list = layer_storage["sum_k"]
        sum_v_list = layer_storage["sum_v"]
        count_list = layer_storage["count"]

        # greedy on-device merge: for each vector, either merge to nearest centroid or append
        if len(sum_k_list) == 0:
            # initialize from batch by promoting each vector as its own centroid
            for i in range(k_proc.shape[0]):
                sum_k_list.append(k_proc[i].clone())
                sum_v_list.append(v_proc[i].clone())
                count_list.append(1.0)
        else:
            # stack existing centroids for distance computation
            centroid_k = torch.stack(sum_k_list, dim=0)  # (C, D) but sums not normalized
            centroid_count = torch.tensor(count_list, device=centroid_k.device, dtype=centroid_k.dtype)
            centroid_mean = centroid_k / centroid_count[:, None]
            # normalize for cosine
            if self.similarity_threshold >= 0.0:
                k_norm = k_proc / (k_proc.norm(dim=1, keepdim=True) + 1e-12)
                c_norm = centroid_mean / (centroid_mean.norm(dim=1, keepdim=True) + 1e-12)
                sims = torch.matmul(k_norm, c_norm.T)  # (N, C)
                best_sim, best_idx = sims.max(dim=1)
                # assign or create
                for i in range(k_proc.shape[0]):
                    sim = float(best_sim[i].item())
                    idx = int(best_idx[i].item())
                    if sim >= (1.0 - self.similarity_threshold):
                        # merge into idx
                        wk = k_proc[i]
                        wv = v_proc[i]
                        sum_k_list[idx] = sum_k_list[idx] + wk
                        sum_v_list[idx] = sum_v_list[idx] + wv
                        count_list[idx] = count_list[idx] + 1.0
                    else:
                        sum_k_list.append(k_proc[i].clone())
                        sum_v_list.append(v_proc[i].clone())
                        count_list.append(1.0)
            else:
                # fallback: just append
                for i in range(k_proc.shape[0]):
                    sum_k_list.append(k_proc[i].clone())
                    sum_v_list.append(v_proc[i].clone())
                    count_list.append(1.0)

        # flush to CPU manager if too many GPU centroids
        if len(sum_k_list) >= self.max_gpu_centroids_per_layer:
            self.flush_layer_to_cpu(layer_idx)

    def flush_layer_to_cpu(self, layer_idx: int):
        """Convert GPU aggregated centroids to numpy and call cpu_manager.add_kv"""
        try:
            import torch
        except Exception:
            return
        if layer_idx not in self._layers:
            return
        storage = self._layers[layer_idx]
        sum_k_list = storage["sum_k"]
        sum_v_list = storage["sum_v"]
        count_list = storage["count"]
        if len(sum_k_list) == 0:
            return

        # compute centroids as sums / counts
        ks = torch.stack(sum_k_list, dim=0)
        vs = torch.stack(sum_v_list, dim=0)
        # ensure counts are float32 for division
        counts = torch.tensor(count_list, device=ks.device, dtype=torch.float32)
        cent_k = ks / counts[:, None]
        cent_v = vs / counts[:, None]

        # move to cpu numpy and call cpu_manager.add_kv
        cent_k_cpu_tensor = cent_k.detach()
        cent_v_cpu_tensor = cent_v.detach()
        # cast half precision / bfloat16 to float32 before numpy conversion
        try:
            if cent_k_cpu_tensor.dtype in (torch.bfloat16, torch.float16):
                cent_k_cpu_tensor = cent_k_cpu_tensor.to(dtype=torch.float32)
        except Exception:
            pass
        try:
            if cent_v_cpu_tensor.dtype in (torch.bfloat16, torch.float16):
                cent_v_cpu_tensor = cent_v_cpu_tensor.to(dtype=torch.float32)
        except Exception:
            pass
        try:
            cent_k_cpu = cent_k_cpu_tensor.cpu().numpy().astype(np.float32)
        except Exception:
            try:
                cent_k_cpu = cent_k_cpu_tensor.to(dtype=torch.float32).cpu().numpy().astype(np.float32)
            except Exception:
                cent_k_cpu = np.asarray(cent_k_cpu_tensor.cpu()).astype(np.float32)
        try:
            cent_v_cpu = cent_v_cpu_tensor.cpu().numpy().astype(np.float32)
        except Exception:
            try:
                cent_v_cpu = cent_v_cpu_tensor.to(dtype=torch.float32).cpu().numpy().astype(np.float32)
            except Exception:
                cent_v_cpu = np.asarray(cent_v_cpu_tensor.cpu()).astype(np.float32)
        try:
            # call cpu manager in one batch
            self.cpu_manager.add_kv(layer_idx, cent_k_cpu, cent_v_cpu)
        except Exception:
            # best-effort, ignore
            pass

        # clear gpu storage for this layer
        self._layers[layer_idx] = {"sum_k": [], "sum_v": [], "count": []}

    def flush_all_to_cpu(self):
        for layer_idx in list(self._layers.keys()):
            self.flush_layer_to_cpu(layer_idx)

    def get_gpu_centroids(self, layer_idx: int):
        """
        Return centroids on-device as (centroids_tensor, counts_tensor) if available.
        Centroids are computed as sums / counts and kept on the device of the stored tensors.
        Returns (None, None) if layer has no GPU aggregated centroids.
        """
        try:
            import torch
        except Exception:
            return None, None
        if layer_idx not in self._layers:
            return None, None
        storage = self._layers[layer_idx]
        sum_k_list = storage["sum_k"]
        count_list = storage["count"]
        if len(sum_k_list) == 0:
            return None, None
        ks = torch.stack(sum_k_list, dim=0)
        counts = torch.tensor(count_list, device=ks.device, dtype=torch.float32)
        centroids = ks / counts[:, None]
        return centroids, counts



__all__ = ["MPKVMGPUAggregator"]


