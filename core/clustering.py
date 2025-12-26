"""Improved online manifold clustering operator for MP-KVM.

This implementation provides:
- Weighted centroid aggregation (sums + counts)
- Sliding-window raw history (optional) for accurate energy computation
- Persistence via decay and minimum-count pruning
- Robust centroid merging when capacity exceeded (closest-pair merge using unravel_index)
- Utilities to fetch centroids and normalized centroid weights
"""
from __future__ import annotations
import time
import numpy as np
from typing import Optional, Tuple, List


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class OnlineManifoldCluster:
    """
    Online clustering tailored for KV cache compression.

    Key design:
    - store centroids as summed vectors and total weight (count) for exact weighted mean
    - ingest streaming batches and either assign to nearest centroid or create a new one
    - when capacity exceeded, merge the closest centroid pairs (weighted merge)
    - optional sliding window history for exact energy/loss diagnostics
    """

    def __init__(
        self,
        dim: int,
        max_centroids: int = 1024,
        distance: str = "cosine",
        sliding_window_size: Optional[int] = None,
        persistence_decay: float = 1.0,
        min_count_threshold: float = 1e-3,
    ):
        assert distance in ("cosine", "euclidean")
        self.dim = int(dim)
        self.max_centroids = int(max_centroids)
        self.distance = distance
        self.sliding_window_size = int(sliding_window_size) if sliding_window_size is not None else None
        self.persistence_decay = float(persistence_decay)
        self.min_count_threshold = float(min_count_threshold)

        # Internal centroid representation: summed vector and total weight
        self._sums: List[np.ndarray] = []
        self._counts: List[float] = []
        # Optional raw history used for diagnostics / energy calculations
        self._history: List[np.ndarray] = [] if self.sliding_window_size is not None else None
        # step counter for optional time-based decay (not used heavily here)
        self._step = 0

    # -------------------
    # Distance helpers
    # -------------------
    def _pairwise_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # x: (n, d), y: (m, d) -> (n, m)
        if x.size == 0 or y.size == 0:
            return np.zeros((x.shape[0], y.shape[0]), dtype=float)
        if self.distance == "cosine":
            xn = _normalize_rows(x)
            yn = _normalize_rows(y)
            return 1.0 - np.dot(xn, yn.T)
        else:
            x2 = np.sum(x * x, axis=1)[:, None]
            y2 = np.sum(y * y, axis=1)[None, :]
            xy = np.dot(x, y.T)
            d2 = x2 + y2 - 2.0 * xy
            d2[d2 < 0] = 0.0
            return np.sqrt(d2)

    def _current_centroids(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._sums:
            return np.zeros((0, self.dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        sums_stack = np.vstack(self._sums)
        counts = np.array(self._counts, dtype=np.float32)
        centroids = sums_stack / np.maximum(counts[:, None], 1e-12)
        return centroids, counts

    # -------------------
    # Public API
    # -------------------
    def add(self, vectors: np.ndarray, weights: Optional[np.ndarray] = None, similarity_threshold: float = 0.1) -> None:
        """
        Ingest a batch of vectors.
        - vectors: (n, d)
        - weights: (n,) positive floats, defaults to 1.0
        - similarity_threshold: max distance (for cosine: 1 - cos_sim) to assign to nearest centroid
        """
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        n = vectors.shape[0]
        if weights is None:
            weights = np.ones((n,), dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32)

        # optionally store history for diagnostics
        if self._history is not None:
            for v in vectors:
                self._history.append(v.copy())
            # trim oldest
            while len(self._history) > self.sliding_window_size:
                self._history.pop(0)

        centroids, counts = self._current_centroids()
        # initialize centroids if empty
        if centroids.shape[0] == 0:
            to_take = min(self.max_centroids, n)
            for i in range(to_take):
                self._sums.append(vectors[i].astype(np.float32) * float(weights[i]))
                self._counts.append(float(weights[i]))
            if n > to_take:
                self.add(vectors[to_take:], weights=weights[to_take:], similarity_threshold=similarity_threshold)
            return

        # compute distances and nearest centroid
        dists = self._pairwise_distance(vectors, centroids)  # (n, m)
        nearest = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(n), nearest]

        for i in range(n):
            d = float(nearest_dist[i])
            idx = int(nearest[i])
            w = float(weights[i])
            v = vectors[i].astype(np.float32)
            if d <= similarity_threshold:
                # weighted incremental update: sums and counts
                self._sums[idx] = self._sums[idx] + v * w
                self._counts[idx] = self._counts[idx] + w
            else:
                # create new centroid
                self._sums.append(v * w)
                self._counts.append(w)

        # optional persistence decay applied to counts (simulate forgetting)
        if self.persistence_decay < 1.0:
            self._counts = [c * self.persistence_decay for c in self._counts]

        self._step += 1
        # prune very small centroids
        self._prune_low_count_centroids()
        # ensure capacity is respected
        self._compress_if_needed()

    def get_centroids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (centroids (C,d), counts (C,))"""
        return self._current_centroids()

    def get_centroid_weights(self) -> np.ndarray:
        """Return normalized centroid weights (sum to 1)"""
        _, counts = self._current_centroids()
        if counts.size == 0:
            return counts
        w = counts.astype(float)
        s = float(w.sum()) if w.sum() != 0 else 1.0
        return w / s

    def energy_loss(self, lambda_diversity: float = 0.0) -> float:
        """Approximate reconstruction loss. If history exists compute exact SSE over history."""
        centroids, counts = self._current_centroids()
        if centroids.shape[0] == 0:
            return 0.0
        sse = 0.0
        if self._history is not None and len(self._history) > 0:
            hist = np.vstack(self._history)
            dists = self._pairwise_distance(hist, centroids)
            nearest = np.argmin(dists, axis=1)
            for idx in range(centroids.shape[0]):
                sel = hist[nearest == idx]
                if sel.size == 0:
                    continue
                dif = sel - centroids[idx : idx + 1]
                sse += float(np.sum(dif * dif))
        diversity = 0.0
        if lambda_diversity > 0.0 and centroids.shape[0] > 1:
            pd = self._pairwise_distance(centroids, centroids)
            diversity = float(np.sum(np.triu(pd, k=1)))
        return float(sse + lambda_diversity * diversity)

    # -------------------
    # Internal maintenance
    # -------------------
    def _prune_low_count_centroids(self):
        # remove centroids with tiny counts to free capacity
        keep_idxs = [i for i, c in enumerate(self._counts) if c >= self.min_count_threshold]
        if len(keep_idxs) == len(self._counts):
            return
        self._sums = [self._sums[i] for i in keep_idxs]
        self._counts = [self._counts[i] for i in keep_idxs]

    def _compress_if_needed(self) -> None:
        # Merge closest centroid pairs until under limit
        while len(self._sums) > self.max_centroids:
            centroids, counts = self._current_centroids()
            m = centroids.shape[0]
            if m <= 1:
                break
            dists = self._pairwise_distance(centroids, centroids)
            # mask diagonal
            np.fill_diagonal(dists, np.inf)
            idx = int(np.argmin(dists))
            i, j = np.unravel_index(idx, dists.shape)
            # Merge j into i (weighted sum)
            self._sums[i] = self._sums[i] + self._sums[j]
            self._counts[i] = self._counts[i] + self._counts[j]
            # remove the centroid with the larger index to avoid shifting lower indices
            # (we merged j into i, so drop the other one)
            if j > i:
                del self._sums[j]
                del self._counts[j]
            else:
                del self._sums[i]
                del self._counts[i]

    # utility for debugging / snapshot
    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_centroids()


__all__ = ["OnlineManifoldCluster"]

"""
Online manifold-partitioned clustering operator for MP-KVM.

Features:
- Online updates for streaming KV vectors
- Support for 'cosine' and 'euclidean' distance metrics
- Sliding window + persistent centroids
- Centroid counts and weighted updates
"""
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class OnlineManifoldClustering:
    """
    A lightweight online clustering operator tailored for KV cache compression.

    API:
    - add(k_vecs, v_vecs, weights=None): ingest new KV vectors (np.ndarray)
    - get_centroids(): return centroids and metadata
    - partition_and_compress(): run a local clustering pass and synthesize centroids
    """

    def __init__(
        self,
        dim: int,
        max_memory_size: int = 65536,
        window_size: int = 4096,
        max_centroids: int = 1024,
        metric: str = "cosine",
        similarity_threshold: float = 0.8,
        adaptive_threshold: bool = False,
        threshold_quantile: float = 0.9,
        min_merge_similarity: Optional[float] = None,
        init_preserve_first_n: Optional[int] = None,
    ):
        self.dim = dim
        self.metric = metric
        self.similarity_threshold = float(similarity_threshold)
        # adaptive thresholding (per-instance / per-layer)
        self.adaptive_threshold = bool(adaptive_threshold)
        self.threshold_quantile = float(threshold_quantile)
        # when trimming centroids, only merge closest pairs if their similarity exceeds this value.
        # If None, fallback to the original behavior.
        self.min_merge_similarity = float(min_merge_similarity) if min_merge_similarity is not None else None
        # when initializing from the first compressed batch, optionally preserve the first N items
        # as separate centroids to avoid early global merging (helps maintain diversity).
        self.init_preserve_first_n = int(init_preserve_first_n) if init_preserve_first_n is not None else None
        # recent similarity windows used to compute adaptive threshold quantiles
        self._recent_best_sims: List[np.ndarray] = []

        # Raw buffers for streaming tokens (kept as ring buffer slice semantics)
        self.keys_buffer: List[np.ndarray] = []
        self.values_buffer: List[np.ndarray] = []
        self.weights_buffer: List[float] = []

        # Persistent centroids: list of centroid vectors, counts, and accumulated weights
        self.centroids: List[np.ndarray] = []
        self.centroid_counts: List[int] = []
        self.centroid_weights: List[float] = []

        self.max_memory_size = int(max_memory_size)
        self.window_size = int(window_size)
        self.max_centroids = int(max_centroids)

    # ----------------------
    # Ingestion / buffering
    # ----------------------
    def add(self, keys: np.ndarray, values: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Add a batch of KV vectors.
        - keys: (N, D)
        - values: (N, Dv) (kept but not used for clustering distance)
        - weights: (N,) optional importance (e.g., attention scores)
        """
        assert keys.ndim == 2 and keys.shape[1] == self.dim
        n = keys.shape[0]
        if weights is None:
            weights = np.ones((n,), dtype=float)
        for i in range(n):
            self.keys_buffer.append(keys[i].astype(np.float32))
            self.values_buffer.append(values[i].astype(np.float32))
            self.weights_buffer.append(float(weights[i]))

        # enforce sliding window trimming
        if len(self.keys_buffer) > self.window_size:
            excess = len(self.keys_buffer) - self.window_size
            # drop oldest into centroids via local compression
            self._compress_oldest_batch(excess)

        # enforce global memory cap by merging into persistent centroids
        total_items = len(self.keys_buffer) + sum(self.centroid_counts)
        if total_items > self.max_memory_size:
            self._prune_to_budget()

    # ----------------------
    # Core local compression
    # ----------------------
    def _compress_oldest_batch(self, batch_size: int):
        """
        Compress the oldest `batch_size` items in the sliding buffer into centroids.
        This is a cheap local clustering pass (greedy agglomeration).
        """
        if batch_size <= 0:
            return
        # take slice
        keys = np.stack(self.keys_buffer[:batch_size], axis=0)
        vals = np.stack(self.values_buffer[:batch_size], axis=0)
        w = np.array(self.weights_buffer[:batch_size], dtype=float)
        # remove them from buffers
        del self.keys_buffer[:batch_size]
        del self.values_buffer[:batch_size]
        del self.weights_buffer[:batch_size]

        # greedy assign to existing centroids if sufficiently similar
        if len(self.centroids) == 0:
            # Optionally preserve first N items as separate centroids to avoid immediate collapse
            if self.init_preserve_first_n is not None and self.init_preserve_first_n > 0:
                to_preserve = min(self.init_preserve_first_n, batch_size, self.max_centroids)
                for i in range(to_preserve):
                    self.centroids.append(keys[i].copy())
                    self.centroid_counts.append(1)
                    self.centroid_weights.append(float(w[i]))
                # for any remaining items, process them normally (assign/merge)
                start_idx = to_preserve
                if start_idx >= batch_size:
                    return
                keys = keys[start_idx:]
                w = w[start_idx:]
                batch_size = keys.shape[0]
            else:
                # initialize centroids directly using weighted average groups
                centroid = self._weighted_mean(keys, w)
                self.centroids.append(centroid)
                self.centroid_counts.append(batch_size)
                self.centroid_weights.append(float(w.sum()))
                return

        # compute similarities
        if self.metric == "cosine":
            k_norm = _normalize_rows(keys)
            c_stack = np.stack(self.centroids, axis=0)
            c_norm = _normalize_rows(c_stack)
            sims = np.dot(k_norm, c_norm.T)  # (n, C)
            best_idx = np.argmax(sims, axis=1)
            best_sim = sims[np.arange(len(keys)), best_idx]
            # collect recent best similarities for adaptive thresholding
            if self.adaptive_threshold:
                try:
                    self._recent_best_sims.append(best_sim.copy())
                    # keep bounded history length
                    if len(self._recent_best_sims) > 64:
                        self._recent_best_sims.pop(0)
                    all_sims = np.concatenate(self._recent_best_sims)
                    # compute quantile-based threshold and update similarity threshold
                    new_thresh = float(np.quantile(all_sims, self.threshold_quantile))
                    # only update if numeric and finite
                    if np.isfinite(new_thresh):
                        self.similarity_threshold = float(new_thresh)
                except Exception:
                    # defensive: if anything fails, keep previous threshold
                    pass
        else:
            # euclidean: smaller distance -> larger negative sim
            c_stack = np.stack(self.centroids, axis=0)
            dists = np.linalg.norm(keys[:, None, :] - c_stack[None, :, :], axis=2)
            best_idx = np.argmin(dists, axis=1)
            best_sim = -dists[np.arange(len(keys)), best_idx]

        # assign or create new centroid
        for i, sim_score in enumerate(best_sim):
            idx = int(best_idx[i])
            if (self.metric == "cosine" and sim_score >= self.similarity_threshold) or (
                self.metric != "cosine" and -sim_score <= (1.0 - self.similarity_threshold)
            ):
                # merge into centroid idx (weighted)
                self._merge_into_centroid(idx, keys[i], w[i])
            else:
                # create new centroid
                self.centroids.append(keys[i].copy())
                self.centroid_counts.append(1)
                self.centroid_weights.append(float(w[i]))

        # cap number of centroids
        self._trim_centroids_if_needed()

    def _weighted_mean(self, xs: np.ndarray, ws: np.ndarray) -> np.ndarray:
        ws_sum = float(ws.sum()) if ws.sum() != 0 else 1.0
        return (xs * ws[:, None]).sum(axis=0) / ws_sum

    def _merge_into_centroid(self, idx: int, key_vec: np.ndarray, weight: float):
        prev_w = float(self.centroid_weights[idx])
        prev_count = int(self.centroid_counts[idx])
        new_w = prev_w + float(weight)
        # weighted incremental update of centroid
        updated = (self.centroids[idx] * prev_w + key_vec * float(weight)) / new_w
        self.centroids[idx] = updated
        self.centroid_weights[idx] = new_w
        self.centroid_counts[idx] = prev_count + 1

    def _trim_centroids_if_needed(self):
        # If too many centroids, greedily merge closest pairs until under budget
        while len(self.centroids) > self.max_centroids:
            c = np.stack(self.centroids, axis=0)
            if self.metric == "cosine":
                cn = _normalize_rows(c)
                sim_mat = np.dot(cn, cn.T)
                np.fill_diagonal(sim_mat, -np.inf)
                i, j = divmod(int(sim_mat.argmax() // sim_mat.shape[0]), sim_mat.shape[0])
                # above i,j extraction is fragile; fallback to simpler approach
            # fallback pairwise
            C = len(self.centroids)
            best_pair = (0, 1)
            best_score = -1.0
            for a in range(C):
                for b in range(a + 1, C):
                    if self.metric == "cosine":
                        a_vec = self.centroids[a] / (np.linalg.norm(self.centroids[a]) + 1e-12)
                        b_vec = self.centroids[b] / (np.linalg.norm(self.centroids[b]) + 1e-12)
                        score = float(np.dot(a_vec, b_vec))
                    else:
                        score = -float(np.linalg.norm(self.centroids[a] - self.centroids[b]))
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
            a, b = best_pair
            # If a minimum merge similarity is configured and the best pair doesn't meet it,
            # prefer merging two smallest-weight centroids (less impact) instead of the closest pair.
            if self.min_merge_similarity is not None and self.metric == "cosine" and best_score < float(self.min_merge_similarity):
                # find two smallest-weight centroids
                idx_sorted = np.argsort(self.centroid_weights)
                a = int(idx_sorted[0])
                b = int(idx_sorted[1]) if len(idx_sorted) > 1 else int(idx_sorted[0])

            # merge b into a using weighted combination
            wa = self.centroid_weights[a]
            wb = self.centroid_weights[b]
            merged = (self.centroids[a] * wa + self.centroids[b] * wb) / (wa + wb + 1e-12)
            self.centroids[a] = merged
            self.centroid_weights[a] = wa + wb
            self.centroid_counts[a] = self.centroid_counts[a] + self.centroid_counts[b]
            # remove b
            del self.centroids[b]
            del self.centroid_weights[b]
            del self.centroid_counts[b]

    # ----------------------
    # Budget pruning
    # ----------------------
    def _prune_to_budget(self):
        """
        If we exceed max_memory_size, aggressively merge oldest centroids
        or least-weighted centroids until under budget.
        """
        total = len(self.keys_buffer) + sum(self.centroid_counts)
        if total <= self.max_memory_size:
            return
        # keep merging smallest-weight centroids
        while total > self.max_memory_size and len(self.centroids) > 1:
            # find two smallest-weight centroids and merge
            idx_sorted = np.argsort(self.centroid_weights)
            a = int(idx_sorted[0])
            b = int(idx_sorted[1])
            wa = self.centroid_weights[a]
            wb = self.centroid_weights[b]
            merged = (self.centroids[a] * wa + self.centroids[b] * wb) / (wa + wb + 1e-12)
            self.centroids[a] = merged
            self.centroid_weights[a] = wa + wb
            self.centroid_counts[a] = self.centroid_counts[a] + self.centroid_counts[b]
            # remove b
            del self.centroids[b]
            del self.centroid_weights[b]
            del self.centroid_counts[b]
            total = len(self.keys_buffer) + sum(self.centroid_counts)

    # ----------------------
    # Public inspection
    # ----------------------
    def get_centroids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (centroids: (C, D), counts: (C,), weights: (C,))
        """
        if len(self.centroids) == 0:
            return np.zeros((0, self.dim), dtype=np.float32), np.array([], dtype=int), np.array([], dtype=float)
        c = np.stack(self.centroids, axis=0)
        return c, np.array(self.centroid_counts, dtype=int), np.array(self.centroid_weights, dtype=float)

    def snapshot_buffer(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current sliding window buffers as arrays (keys, values, weights)."""
        if len(self.keys_buffer) == 0:
            return (np.zeros((0, self.dim), dtype=np.float32), np.zeros((0, self.dim), dtype=np.float32), np.zeros((0,), dtype=float))
        k = np.stack(self.keys_buffer, axis=0)
        v = np.stack(self.values_buffer, axis=0)
        w = np.array(self.weights_buffer, dtype=float)
        return k, v, w


__all__ = ["OnlineManifoldClustering"]


