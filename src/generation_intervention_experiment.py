from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .answer_extractors import extract_answer, is_correct
from .datasets import load_dataset
from .generation import generate_completion, load_model, set_seed
from .hooks import get_layers
from .metrics import count_reasoning_tokens, repetition_rate
from .prompt_steering_experiment import hidden_last_by_layer
from .prompts import build_prompt


def build_effort_vector(bundle, problems: list[dict[str, str]], layer: int, high_mode: str) -> torch.Tensor:
    low_states = []
    high_states = []
    for problem in problems:
        low_states.append(hidden_last_by_layer(bundle, build_prompt(problem["question"], "instant"))[layer])
        high_states.append(hidden_last_by_layer(bundle, build_prompt(problem["question"], high_mode))[layer])
    vector = torch.stack(high_states).mean(dim=0) - torch.stack(low_states).mean(dim=0)
    return vector / (vector.norm() + 1e-8)


def build_random_vector(reference: torch.Tensor, seed: int, random_index: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 1729)
    vector = None
    for _ in range(random_index + 1):
        vector = torch.randn(reference.shape, generator=generator)
        vector = vector / (vector.norm() + 1e-8)
    if vector is None:
        raise ValueError("random_index must be non-negative")
    return vector


def steered_generate(bundle, prompt: str, layer: int, vector: torch.Tensor | None, alpha: float, max_new_tokens: int, temperature: float) -> str:
    if vector is None or alpha == 0:
        return generate_completion(bundle, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    direction = vector.to(bundle.device)

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        steered = tensor.clone()
        steered[:, -1, :] = steered[:, -1, :] + alpha * direction.to(dtype=steered.dtype)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    handle = get_layers(bundle.model)[layer].register_forward_hook(hook)
    try:
        return generate_completion(bundle, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    finally:
        handle.remove()


def run(args: argparse.Namespace) -> dict:
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    problems = load_dataset(args.dataset, args.n, args.seed)
    effort_vector = build_effort_vector(bundle, problems, args.layer, args.high_mode)
    random_vector = build_random_vector(effort_vector, args.seed, args.random_index)
    conditions = [
        ("baseline", None, 0.0),
        ("effort_candidate", effort_vector, args.alpha),
        ("effort_opposite", effort_vector, -args.alpha),
        ("random_control", random_vector, args.alpha),
    ]
    rows = []
    for problem in problems:
        prompt = build_prompt(problem["question"], args.prompt_mode)
        for condition, vector, alpha in conditions:
            completion = steered_generate(bundle, prompt, args.layer, vector, alpha, args.max_new_tokens, args.temperature)
            extracted = extract_answer(completion).text
            reasoning_tokens, total_tokens = count_reasoning_tokens(bundle.tokenizer, prompt, completion)
            rows.append(
                {
                    "problem_id": problem["problem_id"],
                    "question": problem["question"],
                    "gold_answer": problem["gold_answer"],
                    "condition": condition,
                    "layer": args.layer,
                    "alpha": alpha,
                    "completion": completion,
                    "extracted_answer": extracted,
                    "correct": is_correct(extracted, problem["gold_answer"]),
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens,
                    "repetition_rate": repetition_rate(completion),
                    "contains_final": "final" in completion.lower(),
                }
            )
    summaries = []
    for condition, _, _ in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        summaries.append(
            {
                "condition": condition,
                "accuracy": sum(row["correct"] for row in subset) / len(subset),
                "mean_reasoning_tokens": sum(row["reasoning_tokens"] for row in subset) / len(subset),
                "mean_total_tokens": sum(row["total_tokens"] for row in subset) / len(subset),
                "mean_repetition_rate": sum(row["repetition_rate"] for row in subset) / len(subset),
                "final_rate": sum(row["contains_final"] for row in subset) / len(subset),
            }
        )
    return {
        "model": args.model,
        "dataset": args.dataset,
        "n": args.n,
        "high_mode": args.high_mode,
        "prompt_mode": args.prompt_mode,
        "layer": args.layer,
        "alpha": args.alpha,
        "random_index": args.random_index,
        "max_new_tokens": args.max_new_tokens,
        "summaries": summaries,
        "rows": rows,
    }


def write_report(result: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Generation Intervention Experiment",
        "",
        "## Setup",
        f"- n: {result['n']}",
        f"- layer: {result['layer']}",
        f"- alpha: {result['alpha']}",
        f"- high mode for vector: {result['high_mode']}",
        f"- prompt mode for generation: {result['prompt_mode']}",
        f"- max new tokens: {result['max_new_tokens']}",
        "",
        "## Summary",
        "| condition | accuracy | mean reasoning tokens | final rate | repetition |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['condition']} | {row['accuracy']:.3f} | {row['mean_reasoning_tokens']:.2f} | "
            f"{row['final_rate']:.3f} | {row['mean_repetition_rate']:.4f} |"
        )
    lines.extend(["", "## Examples"])
    for row in result["rows"]:
        escaped = row["completion"].replace("\n", "\\n")
        if len(escaped) > 240:
            escaped = escaped[:240] + "..."
        lines.append(
            f"- `{row['condition']}` `{row['problem_id']}` correct={row['correct']} tokens={row['reasoning_tokens']} "
            f"answer={row['extracted_answer']!r} gold={row['gold_answer']!r}: {escaped}"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {path.with_suffix('.md')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short autoregressive generation test for candidate causal effort direction.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="synthetic_math")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--high-mode", default="verify", choices=["cot", "verify", "budget_forcing"])
    parser.add_argument("--prompt-mode", default="cot", choices=["instant", "cot", "verify", "budget_forcing"])
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=-3.0)
    parser.add_argument("--random-index", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
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
