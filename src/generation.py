from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    model: object
    tokenizer: object
    device: torch.device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str = "auto"):
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dtype in {"float32", "fp32"}:
        return torch.float32
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {dtype!r}")


def load_model(model_name_or_path: str, device: str = "auto", dtype: str = "auto") -> ModelBundle:
    torch_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=resolve_dtype(dtype),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(torch_device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return ModelBundle(model=model, tokenizer=tokenizer, device=torch_device)


@torch.inference_mode()
def generate_completion(
    bundle: ModelBundle,
    prompt: str,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
) -> str:
    tokenizer = bundle.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(bundle.device)
    do_sample = temperature > 0
    autocast = torch.autocast(device_type=bundle.device.type, dtype=torch.bfloat16) if bundle.device.type == "cuda" else contextlib.nullcontext()
    with autocast:
        output_ids = bundle.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
