from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .datasets import load_dataset
from .generation import load_model, set_seed
from .hooks import get_layers
from .prompts import build_prompt


CONTINUE_STRINGS = [
    " First",
    " Let's",
    " We",
    " Since",
    " Calculate",
    " Step",
    " Wait",
    " The",
]

FINAL_STRINGS = [
    " Final",
    " Answer",
    " Therefore",
    " Thus",
    " So",
]


def first_token_ids(tokenizer, strings: list[str]) -> list[int]:
    ids = []
    for text in strings:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            ids.append(int(encoded[0]))
    return sorted(set(ids))


def token_mass(logits: torch.Tensor, token_ids: list[int]) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    return float(probs[token_ids].sum().item())


@torch.inference_mode()
def hidden_last_by_layer(bundle, prompt: str) -> torch.Tensor:
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
    outputs = bundle.model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states[1:]
    return torch.stack([state[0, -1, :].detach().float().cpu() for state in hidden_states])


@torch.inference_mode()
def next_logits(bundle, prompt: str) -> torch.Tensor:
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
    outputs = bundle.model(**inputs, use_cache=False)
    return outputs.logits[0, -1, :].detach().float().cpu()


@torch.inference_mode()
def steered_next_logits(bundle, prompt: str, layer: int, vector: torch.Tensor, alpha: float) -> torch.Tensor:
    layers = get_layers(bundle.model)
    direction = vector.to(bundle.device)

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        steered = tensor.clone()
        steered[:, -1, :] = steered[:, -1, :] + alpha * direction.to(dtype=steered.dtype)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    handle = layers[layer].register_forward_hook(hook)
    try:
        return next_logits(bundle, prompt)
    finally:
        handle.remove()


def logit_gap(logits: torch.Tensor, continue_ids: list[int], final_ids: list[int]) -> float:
    return token_mass(logits, continue_ids) - token_mass(logits, final_ids)


def run(args: argparse.Namespace) -> dict:
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    continue_ids = first_token_ids(bundle.tokenizer, CONTINUE_STRINGS)
    final_ids = first_token_ids(bundle.tokenizer, FINAL_STRINGS)
    problems = load_dataset(args.dataset, args.n, args.seed)

    low_states = []
    high_states = []
    neutral_prompts = []
    for problem in problems:
        low_states.append(hidden_last_by_layer(bundle, build_prompt(problem["question"], "instant")))
        high_states.append(hidden_last_by_layer(bundle, build_prompt(problem["question"], args.high_mode)))
        neutral_prompts.append(build_prompt(problem["question"], "cot"))
    low = torch.stack(low_states)
    high = torch.stack(high_states)
    directions = high.mean(dim=0) - low.mean(dim=0)
    directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)

    baseline_rows = []
    for prompt in neutral_prompts:
        logits = next_logits(bundle, prompt)
        baseline_rows.append(
            {
                "continue_mass": token_mass(logits, continue_ids),
                "final_mass": token_mass(logits, final_ids),
                "gap": logit_gap(logits, continue_ids, final_ids),
            }
        )

    layer_rows = []
    layers = list(range(directions.shape[0])) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    for layer in layers:
        for alpha in alphas:
            gaps = []
            continue_masses = []
            final_masses = []
            for prompt in neutral_prompts:
                logits = steered_next_logits(bundle, prompt, layer, directions[layer], alpha)
                continue_masses.append(token_mass(logits, continue_ids))
                final_masses.append(token_mass(logits, final_ids))
                gaps.append(continue_masses[-1] - final_masses[-1])
            base_gap = sum(row["gap"] for row in baseline_rows) / len(baseline_rows)
            layer_rows.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "mean_continue_mass": sum(continue_masses) / len(continue_masses),
                    "mean_final_mass": sum(final_masses) / len(final_masses),
                    "mean_gap": sum(gaps) / len(gaps),
                    "delta_gap_vs_baseline": sum(gaps) / len(gaps) - base_gap,
                }
            )
    best = max(layer_rows, key=lambda row: row["delta_gap_vs_baseline"])
    worst = min(layer_rows, key=lambda row: row["delta_gap_vs_baseline"])
    result = {
        "model": args.model,
        "dataset": args.dataset,
        "n": args.n,
        "high_mode": args.high_mode,
        "continue_strings": CONTINUE_STRINGS,
        "final_strings": FINAL_STRINGS,
        "continue_token_ids": continue_ids,
        "final_token_ids": final_ids,
        "baseline": {
            "mean_continue_mass": sum(row["continue_mass"] for row in baseline_rows) / len(baseline_rows),
            "mean_final_mass": sum(row["final_mass"] for row in baseline_rows) / len(baseline_rows),
            "mean_gap": sum(row["gap"] for row in baseline_rows) / len(baseline_rows),
        },
        "sweeps": layer_rows,
        "best_delta": best,
        "worst_delta": worst,
    }
    return result


def write_report(result: dict, out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = path.with_suffix(".md")
    top = sorted(result["sweeps"], key=lambda row: row["delta_gap_vs_baseline"], reverse=True)[:10]
    bottom = sorted(result["sweeps"], key=lambda row: row["delta_gap_vs_baseline"])[:10]
    lines = [
        "# Prompt Steering Experiment",
        "",
        "## Conclusion",
        (
            f"Best layer/alpha increased continue-minus-final token mass by "
            f"{result['best_delta']['delta_gap_vs_baseline']:.6g}; worst changed it by "
            f"{result['worst_delta']['delta_gap_vs_baseline']:.6g}."
        ),
        "",
        "## Baseline",
        f"- Mean continue mass: {result['baseline']['mean_continue_mass']:.6g}",
        f"- Mean final mass: {result['baseline']['mean_final_mass']:.6g}",
        f"- Mean gap: {result['baseline']['mean_gap']:.6g}",
        "",
        "## Top positive interventions",
        "| layer | alpha | continue | final | gap | delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top:
        lines.append(
            f"| {row['layer']} | {row['alpha']} | {row['mean_continue_mass']:.6g} | "
            f"{row['mean_final_mass']:.6g} | {row['mean_gap']:.6g} | {row['delta_gap_vs_baseline']:.6g} |"
        )
    lines.extend(["", "## Top negative interventions", "| layer | alpha | continue | final | gap | delta |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in bottom:
        lines.append(
            f"| {row['layer']} | {row['alpha']} | {row['mean_continue_mass']:.6g} | "
            f"{row['mean_final_mass']:.6g} | {row['mean_gap']:.6g} | {row['delta_gap_vs_baseline']:.6g} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt-level causal effort steering experiment.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="synthetic_math")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--high-mode", default="verify", choices=["cot", "verify", "budget_forcing"])
    parser.add_argument("--layers", default="all")
    parser.add_argument("--alphas", default="-3,-1,0,1,3")
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
