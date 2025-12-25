"""
Example script showing how to attach MP-KVM to a HuggingFace Llama model.
This is an illustrative snippet — running it requires transformers and model weights.
"""
from __future__ import annotations
import os, sys, textwrap, subprocess
# ensure project root is on sys.path so `from core import ...` works when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Helper: load Hugging Face token from environment or from a repo-local token file.
# Priority: HUGGINGFACE_HUB_TOKEN env var -> repo root file ".hf_token" -> fallback (none).
def _ensure_hf_token():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    if token:
        return token
    # try repo-local token file for convenience (user opted to persist token here)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    token_paths = [os.path.join(repo_root, ".hf_token"), os.path.join(repo_root, "hf_token.txt")]
    for p in token_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    t = f.read().strip()
                    if t:
                        os.environ["HUGGINGFACE_HUB_TOKEN"] = t
                        return t
            except Exception:
                pass
    return None

# call early so downstream HF calls pick up token from env
_ensure_hf_token()
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoTokenizer, PreTrainedTokenizerFast
from core.integration import MPKVMManager
from adapters.llama_adapter import attach_mpkvm_to_hf_llama
import argparse
import yaml
import os
import json


def example_attach(
    model_name: str = "facebook/llama-3-8b",
    head_mean: bool = True,
    enable_injection: bool = True,
    max_injected_centroids: int = 128,
    device: str = "cuda:0",
    sample_stride: int = 1,
    use_gpu_agg: bool = True,
    config_path: str = None,
    local_random: bool = False,
    repeat: int = 50,
    save_dir: str | None = None,
):
    # optional config override from YAML
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_name = cfg.get("model", {}).get("name", model_name)
        head_mean = bool(cfg.get("mpkvm", {}).get("head_mean", head_mean))
        enable_injection = bool(cfg.get("mpkvm", {}).get("enable_injection", enable_injection))
        max_injected_centroids = int(cfg.get("mpkvm", {}).get("max_injected_centroids", max_injected_centroids))
        device = cfg.get("mpkvm", {}).get("device", device)
        sample_stride = int(cfg.get("mpkvm", {}).get("sample_stride", sample_stride))
        use_gpu_agg = bool(cfg.get("mpkvm", {}).get("use_gpu_aggregator", use_gpu_agg))

    # allow loading a locally-downloaded model folder (e.g., repo_root/model)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate = model_name
    # if user passed a simple folder name that exists under repo root, use that
    if not os.path.isabs(candidate) and os.path.isdir(os.path.join(repo_root, candidate)):
        candidate = os.path.join(repo_root, candidate)
    # if user left default and there is a ./model folder, prefer it
    if model_name in (None, "", "model") and os.path.isdir(os.path.join(repo_root, "model")):
        candidate = os.path.join(repo_root, "model")

    try:
        # prefer local_files_only if candidate is a folder on disk
        if os.path.isdir(candidate):
            # recursively find a directory containing config.json
            found_dir = None
            for root, dirs, files in os.walk(candidate):
                if "config.json" in files:
                    found_dir = root
                    break
            if found_dir is not None:
                candidate = found_dir
            cfg_path = os.path.join(candidate, "config.json")
            print(f"Resolved model folder: {os.path.abspath(candidate)}")

            if not os.path.exists(cfg_path):
                if local_random:
                    # build a tiny random Llama config/model for quick testing
                    tiny_cfg = LlamaConfig(
                        hidden_size=64,
                        intermediate_size=256,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        vocab_size=32000,
                    )
                    model = LlamaForCausalLM(tiny_cfg)
                    # fallback tokenizer for testing
                    try:
                        tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    except Exception:
                        tokenizer = None
                else:
                    raise FileNotFoundError(f"{candidate} does not contain config.json. Add a valid HuggingFace model folder or run with --local-random to use a tiny random model for testing.")
            else:
                model = LlamaForCausalLM.from_pretrained(candidate, torch_dtype="auto", local_files_only=True)
                # try fast tokenizer first (may not need sentencepiece if tokenizer.json exists)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=True, local_files_only=True)
                except Exception:
                    # try loading PreTrainedTokenizerFast directly from tokenizer.json if present
                    try:
                        tok_path = os.path.join(candidate, "tokenizer.json")
                        if os.path.exists(tok_path):
                            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)
                        else:
                            # fallback to LlamaTokenizer (may require sentencepiece)
                            tokenizer = LlamaTokenizer.from_pretrained(candidate, local_files_only=True)
                    except Exception:
                        # as a last resort, raise to surface the error to user
                        raise
        else:
            model = LlamaForCausalLM.from_pretrained(candidate, torch_dtype="auto")
            tokenizer = LlamaTokenizer.from_pretrained(candidate)
    except Exception:
        # fallback to normal hub download (may require token/auth)
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        tokenizer = LlamaTokenizer.from_pretrained(model_name)

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    cpu_manager = MPKVMManager(dim=hidden_size, num_layers=num_layers, cluster_kwargs={"max_centroids": 1024, "sliding_window_size": 16384})

    mgr = cpu_manager
    if use_gpu_agg:
        try:
            from core.integration_gpu import MPKVMGPUAggregator

            # lower max_gpu_centroids_per_layer to force more frequent flushes during PoC
            gpu_agg = MPKVMGPUAggregator(cpu_manager, dim=hidden_size, device=device, max_gpu_centroids_per_layer=8, head_mean=head_mean, sample_stride=sample_stride)
            mgr = gpu_agg
        except Exception:
            mgr = cpu_manager

    # enable positionless injection and pass centroid weighting so ReconstructedAttentionTorch
    # receives counts for log-biasing. This toggles the injection code path in adapter.
    attach_mpkvm_to_hf_llama(
        model,
        mgr,
        head_mean=head_mean,
        sample_stride=sample_stride,
        enable_injection=enable_injection,
        max_injected_centroids=max_injected_centroids,
        pass_centroid_weighting=True,
        positionless_injection=True,
    )

    # now run several short generations to collect more K/V; adapter will collect during forward
    prompts = [
        "In 100 words, explain the significance of manifold partitioned KV memories.",
        "Briefly summarize how centroid compression can help long-context transformers.",
        "Explain online clustering for KV caches and why it's useful.",
        "Discuss challenges of position encoding when merging KV tokens."
    ]
    import time, torch
    generations = []
    total = 0
    for r in range(max(1, int(repeat))):
        for p in prompts:
            try:
                inputs = tokenizer(p, return_tensors="pt")
            except Exception:
                # fallback: if tokenizer unavailable, create dummy input ids
                inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}
            out = model.generate(**inputs, max_new_tokens=128)
            # decode output
            try:
                if hasattr(tokenizer, "decode"):
                    gen_ids = out[0]
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                else:
                    gen_text = str(out)
            except Exception:
                try:
                    gen_ids = out[0].cpu().numpy()
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                except Exception:
                    gen_text = "<decode_failed>"
            generations.append({"prompt": p, "generation": gen_text})
            total += 1
        # small sleep to avoid busy-looping on some runtimes
        time.sleep(0.01)
    print(f"Completed {total} generation calls (repeat={repeat})")

    # ensure GPU aggregator flushed to CPU if present
    if hasattr(mgr, "flush_all_to_cpu"):
        mgr.flush_all_to_cpu()

    # ensure GPU aggregator flushed to CPU if present
    if hasattr(mgr, "flush_all_to_cpu"):
        mgr.flush_all_to_cpu()

    # print per-layer centroid diagnostics
    num_layers = num_layers if 'num_layers' in locals() else model.config.num_hidden_layers
    print("Per-layer centroid summary (centroids.shape, counts.shape, weights.shape, sum_counts):")
    for li in range(num_layers):
        try:
            centroids, counts, weights = cpu_manager.get_layer_centroids(li)
            s = float(counts.sum()) if counts.size > 0 else 0.0
            print(f"  layer {li}: centroids={centroids.shape} counts={counts.shape} weights={weights.shape} sum_counts={s:.3f}")
        except Exception:
            print(f"  layer {li}: error reading centroids")

    # if manager supports GPU centroids, print GPU-side summary as well
    try:
        get_gpu = getattr(mgr, "get_gpu_centroids", None)
        if callable(get_gpu):
            print("GPU-side centroid summary (per-layer):")
            for li in range(num_layers):
                try:
                    gcent, gcounts = get_gpu(li)
                    if gcent is None:
                        print(f"  layer {li}: none")
                    else:
                        print(f"  layer {li}: device_centroids={tuple(gcent.shape)} device_counts={tuple(gcounts.shape)}")
                except Exception:
                    print(f"  layer {li}: error")
    except Exception:
        pass

    # compute and save energy loss if possible
    try:
        losses = cpu_manager.energy_loss(lambda_diversity=0.0)
        print("Energy loss per layer (lambda_diversity=0.0):")
        for k, v in losses.items():
            print(f"  layer {k}: {v:.6f}")
    except Exception:
        losses = {}

    # save generations and losses if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        gen_path = os.path.join(save_dir, "generations.txt")
        with open(gen_path, "w", encoding="utf-8") as f:
            for g in generations:
                f.write("PROMPT:\n")
                f.write(g["prompt"] + "\n")
                f.write("GENERATION:\n")
                f.write(g["generation"] + "\n")
                f.write("=" * 80 + "\n")
        loss_path = os.path.join(save_dir, "energy_loss.json")
        try:
            with open(loss_path, "w", encoding="utf-8") as f:
                json.dump({"losses": losses}, f, indent=2)
        except Exception:
            pass
        print(f"Wrote generations to {gen_path} and losses to {loss_path}")

    return {"generations": generations, "losses": losses}


def parse_and_run():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="YAML config path", default=None)
    p.add_argument("--model", default="facebook/llama-3-8b")
    p.add_argument("--head-mean", action="store_true")
    p.add_argument("--no-injection", dest="enable_injection", action="store_false")
    p.add_argument("--max-injected-centroids", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--sample-stride", type=int, default=1)
    p.add_argument("--no-gpu-agg", dest="use_gpu_agg", action="store_false")
    p.add_argument("--local-random", dest="local_random", action="store_true", help="Use a tiny local random model if local folder lacks config.json")
    p.add_argument("--repeat", type=int, default=50, help="Number of times to repeat the prompt set in-process to collect more KV")
    p.add_argument("--save-dir", type=str, default=None, help="Directory to save generation outputs and energy loss")
    args = p.parse_args()

    example_attach(
        model_name=args.model,
        head_mean=args.head_mean,
        enable_injection=args.enable_injection,
        max_injected_centroids=args.max_injected_centroids,
        device=args.device,
        sample_stride=args.sample_stride,
        use_gpu_agg=args.use_gpu_agg,
        config_path=args.config,
        local_random=args.local_random,
        repeat=args.repeat,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    parse_and_run()


