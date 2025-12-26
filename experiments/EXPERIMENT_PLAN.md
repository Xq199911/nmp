# MP-KVM — Experimental plan and next-task checklist

## High-level experiment outline

1. Validation & unit tests
   - Ensure clustering, torch variant, adapter and GPU aggregator unit tests pass (CI / env `py`).
2. Offline algorithmic validation
   - Synthetic cluster tests, hyperparameter sweeps (similarity, window, persistence, min-merge).
   - Energy / reconstruction loss vs random baselines; visualize UMAP/PCA of clusters.
3. GPU-side engineering
   - Implement and optimize GPU aggregator (vectorized updates, reduce temporaries).
   - Micro-benchmark baseline vs optimized aggregator (batch sizes, dims).
   - Streaming benchmark: continuous batches + periodic flush; measure CPU↔GPU bytes and latency.
4. Needles-in-a-haystack experiments
   - Small validation with `needle_near` to verify pipeline can recover near-needles.
   - Parameter sweep (init_preserve, similarity, min_merge, flush cadence) to find operating points.
   - Large-scale runs to collect recall vs compression curves.
5. Adapter / injection experiments (real model)
   - Attach MP-KVM to a real Llama model via adapter; record attention and generations ON/OFF injection.
   - Save per-layer centroids and attention arrays, compute per-layer attention diffs and text-similarity metrics.
6. End-to-end evaluation & baselines
   - Compare MP-KVM against naive deletion and other streaming baselines on recall / generation quality / latency.
7. Paper artifacts
   - Generate final figures: recall vs compression, energy loss curves, streaming throughput, injection attention diffs, example generations.
   - Prepare methods and results sections; include reproducible run commands and config files.

## Immediate next tasks (checklist)

- [ ] Generate summary plots from existing grid results (`experiments/large_grid_out/`) and Needles outputs (`experiments/needles_large_out/`).
- [ ] Compute text-quality metrics (BLEU / ROUGE / perplexity) for ON/OFF generation outputs; save CSVs.
- [ ] Improve attention recording persistence (per-layer timestamped .npy) and re-run ON/OFF if needed (ensures pairwise comparability).
- [ ] Run extended Needles experiments (selected best config) at several compression ratios and save recall-vs-compression curves.
- [ ] Optimize GPU aggregator internals (preallocated per-layer tensors, minimize Python loops) — low-level performance push.
- [ ] Run LongBench / RULER end-to-end benchmarks (scale up to target workloads).
- [ ] Collect final figures and write reproducible runbook (commands, env `py`, model paths, config files).

## Where outputs are saved
- Hyperparameter grids: `experiments/cluster_sweep_out/`, `experiments/large_grid_out/`  
- Benchmarks: `experiments/benchmark_out/` (micro + streaming)  
- Needles experiments: `experiments/needles_near_out/`, `experiments/needles_large_out/`, `experiments/needles_sweep_out/`  
- Injection / real-model: `experiments/realmodel_out/` (generations, centroids, attention, comparison summary)

## How I will proceed (if you confirm)
1. Generate plots & CSV summaries for the grid and Needles outputs.  
2. Compute text-quality metrics for ON/OFF generations.  
3. Re-run ON/OFF injection if attention files are still insufficient to compute per-layer diffs.

If this plan looks good, reply "go" and I will start step 1 (plots & CSVs).


