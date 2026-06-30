from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .activation_patching_experiment import patched_next_logits
from .datasets import load_dataset
from .generation import load_model, set_seed
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


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def monotonic_non_decreasing(values: list[float]) -> bool:
    return all(right >= left for left, right in zip(values, values[1:]))


def monotonic_non_increasing(values: list[float]) -> bool:
    return all(right <= left for left, right in zip(values, values[1:]))


def run(args: argparse.Namespace) -> dict:
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    continue_ids = first_token_ids(bundle.tokenizer, CONTINUE_STRINGS)
    final_ids = first_token_ids(bundle.tokenizer, FINAL_STRINGS)
    problems = load_dataset(args.dataset, args.n, args.seed)
    layers = parse_csv_ints(args.layers)
    ts = parse_csv_floats(args.ts)

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
            low_state = low_states[layer]
            high_state = high_states[layer]
            for t in ts:
                interpolated = (1.0 - t) * low_state + t * high_state
                logits = patched_next_logits(bundle, low_prompt, layer, interpolated)
                rows.append(
                    {
                        "problem_id": problem["problem_id"],
                        "question": problem["question"],
                        "layer": layer,
                        "t": t,
                        "low_gap": low_gap,
                        "high_gap": high_gap,
                        "gap": logit_gap(logits, continue_ids, final_ids),
                        "continue_mass": token_mass(logits, continue_ids),
                        "final_mass": token_mass(logits, final_ids),
                    }
                )
    summaries = []
    for layer in layers:
        layer_rows = [row for row in rows if row["layer"] == layer]
        by_t = []
        for t in ts:
            subset = [row for row in layer_rows if row["t"] == t]
            by_t.append(
                {
                    "t": t,
                    "mean_gap": sum(row["gap"] for row in subset) / len(subset),
                    "mean_continue_mass": sum(row["continue_mass"] for row in subset) / len(subset),
                    "mean_final_mass": sum(row["final_mass"] for row in subset) / len(subset),
                }
            )
        gaps = [item["mean_gap"] for item in by_t]
        summaries.append(
            {
                "layer": layer,
                "by_t": by_t,
                "delta_t1_t0": gaps[-1] - gaps[0],
                "monotonic_non_decreasing": monotonic_non_decreasing(gaps),
                "monotonic_non_increasing": monotonic_non_increasing(gaps),
            }
        )
    return {
        "model": args.model,
        "dataset": args.dataset,
        "n": args.n,
        "low_mode": args.low_mode,
        "high_mode": args.high_mode,
        "layers": layers,
        "ts": ts,
        "summaries": summaries,
        "rows": rows,
    }


def write_report(result: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Activation Interpolation Experiment",
        "",
        "## Setup",
        f"- n: {result['n']}",
        f"- low mode: {result['low_mode']}",
        f"- high mode: {result['high_mode']}",
        f"- layers: {result['layers']}",
        f"- t values: {result['ts']}",
        "",
        "## Summary",
        "| layer | delta t=1 minus t=0 | monotonic increasing | monotonic decreasing |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for summary in sorted(result["summaries"], key=lambda row: row["delta_t1_t0"], reverse=True):
        lines.append(
            f"| {summary['layer']} | {summary['delta_t1_t0']:.6g} | "
            f"{summary['monotonic_non_decreasing']} | {summary['monotonic_non_increasing']} |"
        )
    lines.extend(["", "## Curves"])
    for summary in result["summaries"]:
        lines.extend(
            [
                "",
                f"### Layer {summary['layer']}",
                "",
                "| t | mean gap | continue mass | final mass |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for item in summary["by_t"]:
            lines.append(
                f"| {item['t']} | {item['mean_gap']:.6g} | {item['mean_continue_mass']:.6g} | "
                f"{item['mean_final_mass']:.6g} |"
            )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {path.with_suffix('.md')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interpolate low/high hidden states and measure next-token causal response.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="synthetic_math")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--low-mode", default="instant", choices=["instant", "cot", "verify", "budget_forcing"])
    parser.add_argument("--high-mode", default="verify", choices=["instant", "cot", "verify", "budget_forcing"])
    parser.add_argument("--layers", default="18,20,23,27")
    parser.add_argument("--ts", default="0,0.25,0.5,0.75,1")
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
