from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .datasets import load_dataset
from .generation import load_model, set_seed
from .hooks import get_layers
from .prompt_steering_experiment import (
    CONTINUE_STRINGS,
    FINAL_STRINGS,
    first_token_ids,
    hidden_last_by_layer,
    logit_gap,
    next_logits,
    token_mass,
)
from .prompts import build_prompt


@torch.inference_mode()
def patched_next_logits(bundle, prompt: str, layer: int, replacement: torch.Tensor) -> torch.Tensor:
    direction = replacement.to(bundle.device)

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        patched = tensor.clone()
        patched[:, -1, :] = direction.to(dtype=patched.dtype)
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    handle = get_layers(bundle.model)[layer].register_forward_hook(hook)
    try:
        return next_logits(bundle, prompt)
    finally:
        handle.remove()


def parse_layers(value: str, n_layers: int) -> list[int]:
    if value == "all":
        return list(range(n_layers))
    layers: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(item))
    return layers


def run(args: argparse.Namespace) -> dict:
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    continue_ids = first_token_ids(bundle.tokenizer, CONTINUE_STRINGS)
    final_ids = first_token_ids(bundle.tokenizer, FINAL_STRINGS)
    problems = load_dataset(args.dataset, args.n, args.seed)
    n_layers = len(get_layers(bundle.model))
    layers = parse_layers(args.layers, n_layers)

    rows = []
    for problem in problems:
        low_prompt = build_prompt(problem["question"], args.low_mode)
        high_prompt = build_prompt(problem["question"], args.high_mode)
        low_states = hidden_last_by_layer(bundle, low_prompt)
        high_states = hidden_last_by_layer(bundle, high_prompt)
        low_logits = next_logits(bundle, low_prompt)
        high_logits = next_logits(bundle, high_prompt)
        low_gap = logit_gap(low_logits, continue_ids, final_ids)
        high_gap = logit_gap(high_logits, continue_ids, final_ids)
        for layer in layers:
            low_to_high_logits = patched_next_logits(bundle, low_prompt, layer, high_states[layer])
            high_to_low_logits = patched_next_logits(bundle, high_prompt, layer, low_states[layer])
            low_to_high_gap = logit_gap(low_to_high_logits, continue_ids, final_ids)
            high_to_low_gap = logit_gap(high_to_low_logits, continue_ids, final_ids)
            rows.append(
                {
                    "problem_id": problem["problem_id"],
                    "question": problem["question"],
                    "layer": layer,
                    "low_mode": args.low_mode,
                    "high_mode": args.high_mode,
                    "low_gap": low_gap,
                    "high_gap": high_gap,
                    "low_to_high_gap": low_to_high_gap,
                    "high_to_low_gap": high_to_low_gap,
                    "low_to_high_delta_vs_low": low_to_high_gap - low_gap,
                    "high_to_low_delta_vs_high": high_to_low_gap - high_gap,
                    "low_continue_mass": token_mass(low_logits, continue_ids),
                    "high_continue_mass": token_mass(high_logits, continue_ids),
                    "low_to_high_continue_mass": token_mass(low_to_high_logits, continue_ids),
                    "high_to_low_continue_mass": token_mass(high_to_low_logits, continue_ids),
                }
            )
    summaries = []
    for layer in layers:
        subset = [row for row in rows if row["layer"] == layer]
        summaries.append(
            {
                "layer": layer,
                "mean_low_gap": sum(row["low_gap"] for row in subset) / len(subset),
                "mean_high_gap": sum(row["high_gap"] for row in subset) / len(subset),
                "mean_low_to_high_delta_vs_low": sum(row["low_to_high_delta_vs_low"] for row in subset) / len(subset),
                "mean_high_to_low_delta_vs_high": sum(row["high_to_low_delta_vs_high"] for row in subset) / len(subset),
            }
        )
    best_restore = max(summaries, key=lambda row: row["mean_low_to_high_delta_vs_low"])
    best_suppress = min(summaries, key=lambda row: row["mean_high_to_low_delta_vs_high"])
    return {
        "model": args.model,
        "dataset": args.dataset,
        "n": args.n,
        "low_mode": args.low_mode,
        "high_mode": args.high_mode,
        "layers": layers,
        "summaries": summaries,
        "best_restore": best_restore,
        "best_suppress": best_suppress,
        "rows": rows,
    }


def write_report(result: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    top_restore = sorted(result["summaries"], key=lambda row: row["mean_low_to_high_delta_vs_low"], reverse=True)[:12]
    top_suppress = sorted(result["summaries"], key=lambda row: row["mean_high_to_low_delta_vs_high"])[:12]
    lines = [
        "# Activation Patching Experiment",
        "",
        "## Conclusion",
        (
            f"Best low→high patch layer {result['best_restore']['layer']} changed gap by "
            f"{result['best_restore']['mean_low_to_high_delta_vs_low']:.6g}; "
            f"best high→low suppress layer {result['best_suppress']['layer']} changed gap by "
            f"{result['best_suppress']['mean_high_to_low_delta_vs_high']:.6g}."
        ),
        "",
        "## Setup",
        f"- n: {result['n']}",
        f"- low mode: {result['low_mode']}",
        f"- high mode: {result['high_mode']}",
        "",
        "## Low→High Patches",
        "| layer | low gap | high gap | patched delta vs low |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in top_restore:
        lines.append(
            f"| {row['layer']} | {row['mean_low_gap']:.6g} | {row['mean_high_gap']:.6g} | "
            f"{row['mean_low_to_high_delta_vs_low']:.6g} |"
        )
    lines.extend(["", "## High→Low Patches", "| layer | low gap | high gap | patched delta vs high |", "| ---: | ---: | ---: | ---: |"])
    for row in top_suppress:
        lines.append(
            f"| {row['layer']} | {row['mean_low_gap']:.6g} | {row['mean_high_gap']:.6g} | "
            f"{row['mean_high_to_low_delta_vs_high']:.6g} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {path.with_suffix('.md')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch high-effort activations into low-effort runs and reverse.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="synthetic_math")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--low-mode", default="instant", choices=["instant", "cot", "verify", "budget_forcing"])
    parser.add_argument("--high-mode", default="verify", choices=["instant", "cot", "verify", "budget_forcing"])
    parser.add_argument("--layers", default="0-10")
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
