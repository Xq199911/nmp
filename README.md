# MP-KVM (Manifold-Partitioned Key-Value Memory) — PoC

This repository contains a proof-of-concept implementation for MP-KVM:
online manifold partitioning of KV cache to create persistent centroid tokens
for long-context Transformers.

Structure:
- `core/`: clustering operator, integration manager, and reconstructed attention helpers.
- `analysis/`: visualization and energy loss utilities.
- `experiments/`: a minimal synthetic runner to generate PoC plots and numbers.
- `data/`: placeholders for dataset loaders.
- `scripts/`: env setup helper.

Quickstart (PoC):
1. Create a virtualenv and activate it.
2. Run `scripts/setup_env.sh` to install dependencies (includes torch and transformers CPU wheels).
3. For PoC with a tiny test model (fast, no large downloads):
   - `python scripts/patch_llama_example.py --model hf-internal-testing/tiny-random-llama`
4. To run the synthetic manifold benchmark:
   - `python experiments/run_benchmark.py --out experiments/out`
   This will generate `experiments/out/manifold.png` and print energy loss numbers.

Notes:
- The code is intentionally framework-agnostic for easier prototyping. Concrete model adapters
  (for Llama/Qwen) should implement careful tensor<->numpy conversion and precise attention
  patching.
- This PoC focuses on the algorithmic operator. Full integration (efficient GPU handling,
  batching, and model hooks) is left to `core/integration.py` consumers.
 - For real Llama-3-8B integration, pass the model identifier to `scripts/patch_llama_example.py --model <your-llama-3-8b-id>`.

License: MIT (add your license info).


