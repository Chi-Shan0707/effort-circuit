from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .datasets import load_dataset
from .generation import load_model, set_seed
from .prompt_steering_experiment import (
    CONTINUE_STRINGS,
    FINAL_STRINGS,
    first_token_ids,
    hidden_last_by_layer,
    logit_gap,
    next_logits,
    steered_next_logits,
    token_mass,
)
from .prompts import build_prompt


def mean_gap_for_vector(bundle, prompts: list[str], layer: int, vector: torch.Tensor, alpha: float, continue_ids: list[int], final_ids: list[int]) -> dict:
    continue_masses = []
    final_masses = []
    gaps = []
    for prompt in prompts:
        logits = steered_next_logits(bundle, prompt, layer, vector, alpha)
        continue_mass = token_mass(logits, continue_ids)
        final_mass = token_mass(logits, final_ids)
        continue_masses.append(continue_mass)
        final_masses.append(final_mass)
        gaps.append(continue_mass - final_mass)
    return {
        "mean_continue_mass": sum(continue_masses) / len(continue_masses),
        "mean_final_mass": sum(final_masses) / len(final_masses),
        "mean_gap": sum(gaps) / len(gaps),
    }


def run(args: argparse.Namespace) -> dict:
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    continue_ids = first_token_ids(bundle.tokenizer, CONTINUE_STRINGS)
    final_ids = first_token_ids(bundle.tokenizer, FINAL_STRINGS)
    problems = load_dataset(args.dataset, args.n, args.seed)
    neutral_prompts = [build_prompt(problem["question"], "cot") for problem in problems]

    low_states = []
    high_states = []
    for problem in problems:
        low_states.append(hidden_last_by_layer(bundle, build_prompt(problem["question"], "instant"))[args.layer])
        high_states.append(hidden_last_by_layer(bundle, build_prompt(problem["question"], args.high_mode))[args.layer])
    low = torch.stack(low_states)
    high = torch.stack(high_states)
    effort_vector = high.mean(dim=0) - low.mean(dim=0)
    effort_vector = effort_vector / (effort_vector.norm() + 1e-8)

    baseline_gaps = []
    baseline_continue = []
    baseline_final = []
    for prompt in neutral_prompts:
        logits = next_logits(bundle, prompt)
        baseline_continue.append(token_mass(logits, continue_ids))
        baseline_final.append(token_mass(logits, final_ids))
        baseline_gaps.append(logit_gap(logits, continue_ids, final_ids))
    baseline = {
        "mean_continue_mass": sum(baseline_continue) / len(baseline_continue),
        "mean_final_mass": sum(baseline_final) / len(baseline_final),
        "mean_gap": sum(baseline_gaps) / len(baseline_gaps),
    }

    candidate = mean_gap_for_vector(bundle, neutral_prompts, args.layer, effort_vector, args.alpha, continue_ids, final_ids)
    candidate["delta_gap_vs_baseline"] = candidate["mean_gap"] - baseline["mean_gap"]

    generator = torch.Generator().manual_seed(args.seed + 1729)
    random_rows = []
    for idx in range(args.random_directions):
        random_vector = torch.randn(effort_vector.shape, generator=generator)
        random_vector = random_vector / (random_vector.norm() + 1e-8)
        row = mean_gap_for_vector(bundle, neutral_prompts, args.layer, random_vector, args.alpha, continue_ids, final_ids)
        row["random_index"] = idx
        row["delta_gap_vs_baseline"] = row["mean_gap"] - baseline["mean_gap"]
        random_rows.append(row)

    random_deltas = [row["delta_gap_vs_baseline"] for row in random_rows]
    better_or_equal = sum(delta <= candidate["delta_gap_vs_baseline"] for delta in random_deltas)
    percentile = better_or_equal / max(1, len(random_deltas))
    result = {
        "model": args.model,
        "dataset": args.dataset,
        "n": args.n,
        "high_mode": args.high_mode,
        "layer": args.layer,
        "alpha": args.alpha,
        "random_directions": args.random_directions,
        "baseline": baseline,
        "candidate": candidate,
        "random_summary": {
            "mean_delta": sum(random_deltas) / len(random_deltas) if random_deltas else None,
            "min_delta": min(random_deltas) if random_deltas else None,
            "max_delta": max(random_deltas) if random_deltas else None,
            "candidate_percentile": percentile,
        },
        "random_rows": random_rows,
    }
    return result


def write_report(result: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    top_random = sorted(result["random_rows"], key=lambda row: row["delta_gap_vs_baseline"], reverse=True)[:10]
    lines = [
        "# Random Direction Control",
        "",
        "## Conclusion",
        (
            f"Candidate delta gap is {result['candidate']['delta_gap_vs_baseline']:.6g}; "
            f"random max is {result['random_summary']['max_delta']:.6g}; "
            f"candidate percentile is {result['random_summary']['candidate_percentile']:.3f}."
        ),
        "",
        "## Setup",
        f"- Layer: {result['layer']}",
        f"- Alpha: {result['alpha']}",
        f"- n: {result['n']}",
        f"- High mode: {result['high_mode']}",
        f"- Random directions: {result['random_directions']}",
        "",
        "## Candidate",
        f"- Continue mass: {result['candidate']['mean_continue_mass']:.6g}",
        f"- Final mass: {result['candidate']['mean_final_mass']:.6g}",
        f"- Gap: {result['candidate']['mean_gap']:.6g}",
        f"- Delta vs baseline: {result['candidate']['delta_gap_vs_baseline']:.6g}",
        "",
        "## Random Summary",
        f"- Mean random delta: {result['random_summary']['mean_delta']:.6g}",
        f"- Min random delta: {result['random_summary']['min_delta']:.6g}",
        f"- Max random delta: {result['random_summary']['max_delta']:.6g}",
        "",
        "## Top Random Directions",
        "| random index | continue | final | gap | delta |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_random:
        lines.append(
            f"| {row['random_index']} | {row['mean_continue_mass']:.6g} | {row['mean_final_mass']:.6g} | "
            f"{row['mean_gap']:.6g} | {row['delta_gap_vs_baseline']:.6g} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {path.with_suffix('.md')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare effort steering direction against random directions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="synthetic_math")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--high-mode", default="verify", choices=["cot", "verify", "budget_forcing"])
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=-3.0)
    parser.add_argument("--random-directions", type=int, default=32)
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
