from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .answer_extractors import extract_answer, is_correct
from .datasets import load_dataset
from .generation import load_model, set_seed
from .hooks import get_layers
from .metrics import repetition_rate
from .prompt_steering_experiment import CONTINUE_STRINGS, FINAL_STRINGS, first_token_ids, token_mass
from .prompts import build_prompt
from .process_task_disentanglement import capture_hook_state

STRICT_FINAL_RE = re.compile(r"(?:^|\n)\s*(?:Final answer|Final)\s*(?::|is|=)\s*(-?\d+(?:\.\d+)?)", re.I)


def has_valid_final(text: str) -> bool:
    return STRICT_FINAL_RE.search(text or "") is not None


def alpha_policy(
    generated_tokens: int,
    continue_mass: float,
    final_mass: float,
    repetition: float,
    alpha_max: float,
    min_reasoning_tokens: int,
    repetition_threshold: float,
) -> float:
    if repetition >= repetition_threshold:
        return 0.0
    if generated_tokens < min_reasoning_tokens and final_mass >= continue_mass:
        return alpha_max
    if generated_tokens < min_reasoning_tokens:
        return alpha_max * 0.5
    if final_mass >= continue_mass:
        return 0.0
    return alpha_max * 0.25


@torch.inference_mode()
def mean_process_direction(bundle, train_rows: list[dict[str, str]], layer: int, low_mode: str, high_mode: str) -> torch.Tensor:
    total = None
    for problem in train_rows:
        low_prompt = build_prompt(problem["question"], low_mode)
        high_prompt = build_prompt(problem["question"], high_mode)
        low_state = capture_hook_state(bundle, low_prompt, layer)
        high_state = capture_hook_state(bundle, high_prompt, layer)
        delta = high_state - low_state
        total = delta if total is None else total + delta
    return total / len(train_rows)


@torch.inference_mode()
def forward_with_direction(
    bundle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    past_key_values=None,
):
    if alpha == 0.0:
        return bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
    target_direction = direction.to(bundle.device)

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        patched = tensor.clone()
        patched[:, -1, :] = patched[:, -1, :] + alpha * target_direction.to(dtype=patched.dtype)
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    handle = get_layers(bundle.model)[layer].register_forward_hook(hook)
    try:
        return bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
    finally:
        handle.remove()


@torch.inference_mode()
def greedy_generate_controlled(
    bundle,
    prompt: str,
    layer: int,
    direction: torch.Tensor,
    controller: str,
    max_new_tokens: int,
    alpha_max: float,
    min_reasoning_tokens: int,
    repetition_threshold: float,
    continue_ids: list[int],
    final_ids: list[int],
) -> dict[str, Any]:
    encoded = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
    current_input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_len = current_input_ids.shape[1]
    past_key_values = None
    generated_ids: list[int] = []
    alpha_trace = []
    signal_trace = []
    completion = ""
    for step in range(max_new_tokens):
        if has_valid_final(completion):
            break
        probe_outputs = forward_with_direction(
            bundle,
            current_input_ids,
            attention_mask,
            layer,
            direction,
            0.0,
            past_key_values=past_key_values,
        )
        probe_logits = probe_outputs.logits[0, -1, :]
        continue_mass = token_mass(probe_logits, continue_ids)
        final_mass = token_mass(probe_logits, final_ids)
        repetition = repetition_rate(completion)
        if controller == "baseline":
            alpha = 0.0
        elif controller == "static_mean_direction":
            alpha = alpha_max
            if repetition >= repetition_threshold:
                alpha = 0.0
        elif controller == "closed_loop_cet":
            alpha = alpha_policy(step, continue_mass, final_mass, repetition, alpha_max, min_reasoning_tokens, repetition_threshold)
        else:
            raise ValueError(f"unknown controller {controller!r}")
        if alpha == 0.0:
            logits = probe_logits
            next_past_key_values = probe_outputs.past_key_values
        else:
            steered_outputs = forward_with_direction(
                bundle,
                current_input_ids,
                attention_mask,
                layer,
                direction,
                alpha,
                past_key_values=past_key_values,
            )
            logits = steered_outputs.logits[0, -1, :]
            next_past_key_values = probe_outputs.past_key_values
        next_token = torch.argmax(logits, dim=-1).reshape(1, 1)
        generated_ids.append(int(next_token.item()))
        completion = bundle.tokenizer.decode(generated_ids, skip_special_tokens=True)
        past_key_values = next_past_key_values
        current_input_ids = next_token.to(bundle.device)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token, device=attention_mask.device)], dim=1)
        alpha_trace.append(alpha)
        signal_trace.append(
            {
                "step": step,
                "alpha": alpha,
                "continue_mass": continue_mass,
                "final_mass": final_mass,
                "repetition": repetition,
                "token": bundle.tokenizer.decode([int(next_token.item())]).replace("\n", "\\n"),
            }
        )
        if bundle.tokenizer.eos_token_id is not None and int(next_token.item()) == bundle.tokenizer.eos_token_id:
            break
    return {
        "completion": completion,
        "generated_tokens": len(alpha_trace),
        "stopped_valid_final": has_valid_final(completion),
        "mean_alpha": sum(alpha_trace) / len(alpha_trace) if alpha_trace else 0.0,
        "max_alpha": max(alpha_trace) if alpha_trace else 0.0,
        "alpha_trace": alpha_trace,
        "signal_trace": signal_trace,
    }


def strict_prompt(question: str) -> str:
    return (
        "Solve the problem. Use exactly this format:\n"
        "Reasoning: <brief reasoning>\n"
        "Final answer: <number>\n\n"
        f"Question: {question}\n"
        "Reasoning:"
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["controller"], []).append(row)
    summaries = []
    for controller, subset in groups.items():
        summaries.append(
            {
                "controller": controller,
                "n": len(subset),
                "accuracy": sum(row["correct"] for row in subset) / len(subset),
                "strict_stop_rate": sum(row["stopped_valid_final"] for row in subset) / len(subset),
                "mean_generated_tokens": sum(row["generated_tokens"] for row in subset) / len(subset),
                "mean_repetition_rate": sum(row["repetition_rate"] for row in subset) / len(subset),
                "mean_alpha": sum(row["mean_alpha"] for row in subset) / len(subset),
            }
        )
    return summaries


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    train_rows = load_dataset(args.train_dataset, args.train_n, args.seed)
    eval_rows = load_dataset(args.eval_dataset, args.eval_n, args.seed)
    direction = mean_process_direction(bundle, train_rows, args.layer, args.low_mode, args.high_mode)
    continue_ids = first_token_ids(bundle.tokenizer, CONTINUE_STRINGS)
    final_ids = first_token_ids(bundle.tokenizer, FINAL_STRINGS)
    controllers = [item.strip() for item in args.controllers.split(",") if item.strip()]
    rows = []
    for problem in eval_rows:
        prompt = strict_prompt(problem["question"])
        for controller in controllers:
            result = greedy_generate_controlled(
                bundle,
                prompt,
                args.layer,
                direction,
                controller,
                args.max_new_tokens,
                args.alpha_max,
                args.min_reasoning_tokens,
                args.repetition_threshold,
                continue_ids,
                final_ids,
            )
            extracted = extract_answer(result["completion"]).text
            rows.append(
                {
                    "problem_id": problem["problem_id"],
                    "question": problem["question"],
                    "gold_answer": problem["gold_answer"],
                    "controller": controller,
                    "layer": args.layer,
                    "completion": result["completion"],
                    "extracted_answer": extracted,
                    "correct": is_correct(extracted, problem["gold_answer"]),
                    "repetition_rate": repetition_rate(result["completion"]),
                    **result,
                }
            )
    return {
        "model": args.model,
        "train_dataset": args.train_dataset,
        "train_n": args.train_n,
        "eval_dataset": args.eval_dataset,
        "eval_n": args.eval_n,
        "layer": args.layer,
        "alpha_max": args.alpha_max,
        "min_reasoning_tokens": args.min_reasoning_tokens,
        "repetition_threshold": args.repetition_threshold,
        "summaries": summarize(rows),
        "rows": rows,
    }


def write_report(result: dict[str, Any], out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Closed-Loop CET Controller",
        "",
        "## Summary",
        "",
        "| controller | n | accuracy | strict stop rate | mean tokens | repetition | mean alpha |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['controller']} | {row['n']} | {row['accuracy']:.3f} | {row['strict_stop_rate']:.3f} | "
            f"{row['mean_generated_tokens']:.2f} | {row['mean_repetition_rate']:.4f} | {row['mean_alpha']:.3f} |"
        )
    lines.extend(["", "## Examples"])
    for row in result["rows"]:
        text = row["completion"].replace("\n", "\\n")
        if len(text) > 220:
            text = text[:220] + "..."
        lines.append(
            f"- `{row['controller']}` `{row['problem_id']}` answer={row['extracted_answer']!r} "
            f"gold={row['gold_answer']!r} correct={row['correct']} tokens={row['generated_tokens']} mean_alpha={row['mean_alpha']:.3f}: {text}"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {path.with_suffix('.md')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a token-level closed-loop CET controller using a mean process direction.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-dataset", default="synthetic_math")
    parser.add_argument("--train-n", type=int, default=8)
    parser.add_argument("--eval-dataset", default="heldout_synthetic_math")
    parser.add_argument("--eval-n", type=int, default=3)
    parser.add_argument("--low-mode", default="instant")
    parser.add_argument("--high-mode", default="verify")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--controllers", default="baseline,static_mean_direction,closed_loop_cet")
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--min-reasoning-tokens", type=int, default=16)
    parser.add_argument("--repetition-threshold", type=float, default=0.18)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    write_report(run(args), args.out)


if __name__ == "__main__":
    main()
