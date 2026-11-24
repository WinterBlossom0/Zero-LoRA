"""The frozen base model of every run: its name, end token, masked modules, and loading.

ZL_MODEL selects the model per process (Qwen3.5-2B-Base by default; Llama-3.2-1B for the replication). Base models
only (pretrained, no alignment training). The end token closes each document when texts are joined into windows.
"""

import os

import torch
from transformers import AutoModelForCausalLM

import planting
import precision

END_TOKENS = {"Qwen/Qwen3.5-2B-Base": "<|endoftext|>", "meta-llama/Llama-3.2-1B": "<|end_of_text|>"}
MODEL_NAME = os.environ.get("ZL_MODEL", "Qwen/Qwen3.5-2B-Base")
if MODEL_NAME not in END_TOKENS:
    raise ValueError(f"ZL_MODEL must be one of {sorted(END_TOKENS)}, not {MODEL_NAME!r}")
END_TOKEN = END_TOKENS[MODEL_NAME]
TARGET_MODULES = ["gate_proj"]  # the masks act on the gate_proj weights of every MLP layer


def load_model(planted: str | None = None) -> torch.nn.Module:
    """Load the frozen model in the run precision with gradient checkpointing.

    planted: a sparse change file of planting.py, applied to the base model (setting A); None: the base model.
    """
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=precision.COMPUTE_DTYPE)
    if planted is not None:
        planting.apply_changes(model, planted)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
