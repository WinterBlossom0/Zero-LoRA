"""Compute precision of the model weights and autocast, chosen once per process by the environment.

ZL_DTYPE = fp16 (default), bf16 or fp32. fp16 with a dynamic loss scaler (grad_scaler) gives mask gradients closer
to fp32 than bf16 does (0.4-0.9 % vs 2.5 % relative error, Qwen3.5-2B, one batch) at the same speed. Every run
records its precision in its config so that results of different precisions stay apart."""

import os

import torch

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
NAME = os.environ.get("ZL_DTYPE", "fp16")
if NAME not in DTYPES:
    raise ValueError(f"ZL_DTYPE must be one of {sorted(DTYPES)}, not {NAME!r}")
COMPUTE_DTYPE = DTYPES[NAME]


def autocast(device_type: str) -> torch.autocast:
    """Return autocast in the run precision; off on the CPU and in fp32, where torch has no fp32 autocast."""
    return torch.autocast(device_type=device_type, dtype=COMPUTE_DTYPE,
                          enabled=device_type == "cuda" and COMPUTE_DTYPE != torch.float32)


def grad_scaler() -> torch.amp.GradScaler:
    """Return a dynamic loss scaler, active only in fp16.

    Without it, fp16 gradients of Qwen3.5-2B underflow: mask gradients differed from fp32 by 8-10 % (relative L2
    error, one batch), and by 0.4-0.9 % with a loss scale of 256 to 4096 (bf16: 2.5 %).
    """
    return torch.amp.GradScaler("cuda", enabled=COMPUTE_DTYPE == torch.float16)
