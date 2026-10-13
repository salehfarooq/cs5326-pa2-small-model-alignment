"""Kaggle preflight smoke: version/device checks plus tiny SFT/DPO forwards.

Run on a Kaggle T4 x2 notebook (not locally). No full training or generation.
Model downloads happen only when the reviewer runs this script.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import sys
import time

import torch

import submission_adapter as adapter
from src.data import format_prompt

MODEL_ID = "Qwen/Qwen2.5-0.5B"
SEED = 42
MAX_LENGTH = 128
LOG2 = 0.6931471805599453

SMOKE_PROMPTS = [
    "Explain what gravity is in one short sentence.",
    "Name two primary colors.",
]
SMOKE_CHOSEN = ["Gravity pulls objects toward each other.", "Red and blue."]
SMOKE_REJECTED = ["Gravity is a kind of color.", "Red and green and also cats."]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Kaggle preflight smoke for PA2 adapter")
    parser.add_argument("--revision", required=True, help="immutable 40 hex char model commit")
    parser.add_argument("--output", required=True, help="path for the JSON status report")
    return parser.parse_args(argv)


def _check_revision(revision):
    if not isinstance(revision, str):
        raise ValueError("--revision must be a string")
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("--revision must be exactly 40 hex characters")


def _device_info():
    devices = []
    for idx in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(idx)
        devices.append({
            "index": idx,
            "name": prop.name,
            "total_memory_bytes": prop.total_memory,
        })
    return devices


def _torch_versions():
    import transformers
    import datasets
    import huggingface_hub
    import accelerate

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "accelerate": accelerate.__version__,
    }


def _atomic_write_json(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(argv=None):
    args = _parse_args(argv)
    _check_revision(args.revision)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    report = {
        "model": MODEL_ID,
        "revision": args.revision,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "dtype": "float32",
        "autocast_dtype": "float16",
        "synthetic_smoke_fixture": True,
        "optimizer_update_performed": False,
        "full_training": False,
        "generation_performed": False,
        "checks": {},
    }

    try:
        _run(args, report)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _atomic_write_json(args.output, report)
        raise

    report["status"] = "success"
    _atomic_write_json(args.output, report)
    print(f"Report written: {args.output}")
    print("=== PA2 Kaggle smoke preflight passed ===")


def _run(args, report):
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=== PA2 Kaggle smoke preflight ===")
    print(f"Model: {MODEL_ID}")
    print(f"Revision: {args.revision}")
    print(f"Seed: {SEED}, max_length: {MAX_LENGTH}")

    versions = _torch_versions()
    report["versions"] = versions
    print(f"Versions: {json.dumps(versions)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; two T4 GPUs are required")
    device_count = torch.cuda.device_count()
    if device_count < 2:
        raise RuntimeError(f"expected at least two CUDA devices, found {device_count}")
    devices = _device_info()
    report["devices"] = devices
    print(f"CUDA devices: {json.dumps(devices)}")
    for device in devices:
        if "T4" not in device["name"]:
            raise RuntimeError(f"device {device['index']} is not a T4: {device['name']}")
    report["checks"]["two_t4"] = True

    torch.cuda.reset_peak_memory_stats("cuda:0")
    torch.cuda.reset_peak_memory_stats("cuda:1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=args.revision,
        trust_remote_code=False,
    )

    print("Loading policy model (FP32, cuda:0)...")
    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=args.revision,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda:0")
    policy.config.use_cache = False
    policy.train()
    for param in policy.parameters():
        assert param.is_floating_point(), "policy FP32 params required"
        assert param.dtype == torch.float32, "policy must stay FP32 on T4"
    report["checks"]["fp32_policy"] = True

    print("Loading frozen reference (FP32, cuda:1)...")
    reference = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=args.revision,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda:1")
    reference.config.use_cache = False
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    report["checks"]["reference_frozen"] = True

    print("NOTE: prompts/responses are synthetic smoke fixtures, not evaluation data.")

    prompts = [format_prompt(text) for text in SMOKE_PROMPTS]

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)

    print("SFT forward/backward under FP16 autocast...")
    optimizer.zero_grad(set_to_none=True)
    _sync()
    sft_start = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        sft_loss = adapter.compute_sft_loss(policy, tokenizer, prompts, SMOKE_CHOSEN, MAX_LENGTH)
    if not torch.isfinite(sft_loss) or sft_loss.item() <= 0.0:
        raise RuntimeError(f"SFT loss must be finite positive, got {sft_loss.item()}")
    scaler.scale(sft_loss).backward()
    scaler.unscale_(optimizer)
    sft_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    _sync()
    sft_seconds = time.time() - sft_start
    if not torch.isfinite(sft_norm) or sft_norm.item() == 0.0:
        raise RuntimeError(f"SFT grad norm must be finite nonzero, got {sft_norm.item()}")
    report["checks"]["finite_sft_grad"] = True
    report["sft"] = {
        "loss": sft_loss.item(),
        "grad_norm": sft_norm.item(),
        "seconds": sft_seconds,
    }
    print(f"SFT loss: {sft_loss.item():.6f} | grad norm: {sft_norm.item():.6f}")

    optimizer.zero_grad(set_to_none=True)

    print("DPO forward/backward under FP16 autocast...")
    policy.eval()
    reference.eval()
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    _sync()
    dpo_start = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        dpo_loss = adapter.compute_dpo_loss(
            policy,
            reference,
            tokenizer,
            prompts,
            SMOKE_CHOSEN,
            SMOKE_REJECTED,
            beta=0.1,
            max_length=MAX_LENGTH,
        )
    if not torch.isfinite(dpo_loss):
        raise RuntimeError("DPO loss must be finite")
    dpo_value = dpo_loss.item()
    if abs(dpo_value - LOG2) > 0.01:
        raise RuntimeError(f"DPO loss should be close to log(2), got {dpo_value}")
    scaler.scale(dpo_loss).backward()
    scaler.unscale_(optimizer)
    dpo_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    _sync()
    dpo_seconds = time.time() - dpo_start
    if not torch.isfinite(dpo_norm) or dpo_norm.item() == 0.0:
        raise RuntimeError(f"DPO grad norm must be finite nonzero, got {dpo_norm.item()}")
    for param in reference.parameters():
        assert param.grad is None, "reference must never receive gradients"
    report["checks"]["finite_dpo_grad"] = True
    report["checks"]["reference_grads_none"] = True
    report["checks"]["initial_dpo_log2"] = True
    report["dpo"] = {
        "loss": dpo_value,
        "log2": LOG2,
        "abs_diff_from_log2": abs(dpo_value - LOG2),
        "grad_norm": dpo_norm.item(),
        "seconds": dpo_seconds,
        "reference_grads_none": True,
    }
    print(f"DPO loss: {dpo_value:.6f} | log(2)={LOG2:.6f} | grad norm: {dpo_norm.item():.6f}")

    policy.train()

    peak0 = torch.cuda.max_memory_allocated("cuda:0")
    peak1 = torch.cuda.max_memory_allocated("cuda:1")
    report["peak_allocated_memory"] = {"cuda:0": peak0, "cuda:1": peak1}
    print(f"Peak allocated: cuda:0={peak0} cuda:1={peak1}")


if __name__ == "__main__":
    sys.exit(main())
