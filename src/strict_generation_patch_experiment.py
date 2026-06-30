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


def strict_prompt(question: str) -> str:
    return (
        "You must use exactly this format:\n"
        "Reasoning: <brief arithmetic steps>\n"
        "Final answer: <number>\n\n"
        f"Question: {question}\n"
        "Reasoning:"
    )


def parse_conditions(value: str) -> list[tuple[str, int | None, float | None]]:
    conditions: list[tuple[str, int | None, float | None]] = [("baseline", None, None)]
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        layer_text, t_text = item.split(":", 1)
        conditions.append((f"layer{layer_text}_t{t_text}", int(layer_text), float(t_text)))
    return conditions


def prefix_patch_generate(
    bundle,
    prompt: str,
    layer: int | None,
    replacement: torch.Tensor | None,
    max_new_tokens: int,
    temperature: float,
) -> str:
    if layer is None or replacement is None:
        return generate_completion(bundle, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    used = {"done": False}
    target = replacement.to(bundle.device)

    def hook(_module, _inputs, output):
        if used["done"]:
            return output
        tensor = output[0] if isinstance(output, tuple) else output
        patched = tensor.clone()
        patched[:, -1, :] = target.to(dtype=patched.dtype)
        used["done"] = True
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

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
    conditions = parse_conditions(args.conditions)
    rows = []
    for problem in problems:
        low_states = hidden_last_by_layer(bundle, build_prompt(problem["question"], args.low_mode))
        high_states = hidden_last_by_layer(bundle, build_prompt(problem["question"], args.high_mode))
        prompt = strict_prompt(problem["question"])
        for condition, layer, t in conditions:
            replacement = None
            if layer is not None and t is not None:
                replacement = (1.0 - t) * low_states[layer] + t * high_states[layer]
            completion = prefix_patch_generate(bundle, prompt, layer, replacement, args.max_new_tokens, args.temperature)
            extracted = extract_answer(completion).text
            reasoning_tokens, total_tokens = count_reasoning_tokens(bundle.tokenizer, prompt, completion)
            lower = completion.lower()
            rows.append(
                {
                    "problem_id": problem["problem_id"],
                    "question": problem["question"],
                    "gold_answer": problem["gold_answer"],
                    "condition": condition,
                    "layer": layer,
                    "t": t,
                    "completion": completion,
                    "extracted_answer": extracted,
                    "correct": is_correct(extracted, problem["gold_answer"]),
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens,
                    "has_final": "final answer" in lower or "final:" in lower,
                    "has_reasoning_marker": "reasoning" in lower,
                    "repetition_rate": repetition_rate(completion),
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
                "final_rate": sum(row["has_final"] for row in subset) / len(subset),
                "mean_repetition_rate": sum(row["repetition_rate"] for row in subset) / len(subset),
                "answer_rate": sum(row["extracted_answer"] is not None for row in subset) / len(subset),
            }
        )
    return {
        "model": args.model,
        "dataset": args.dataset,
        "n": args.n,
        "low_mode": args.low_mode,
        "high_mode": args.high_mode,
        "conditions": [condition for condition, _, _ in conditions],
        "max_new_tokens": args.max_new_tokens,
        "summaries": summaries,
        "rows": rows,
    }


def write_report(result: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Strict Generation Prefix-Patch Experiment",
        "",
        "## Setup",
        f"- n: {result['n']}",
        f"- low mode: {result['low_mode']}",
        f"- high mode: {result['high_mode']}",
        f"- max new tokens: {result['max_new_tokens']}",
        "",
        "## Summary",
        "| condition | accuracy | answer rate | final rate | mean reasoning tokens | repetition |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['condition']} | {row['accuracy']:.3f} | {row['answer_rate']:.3f} | "
            f"{row['final_rate']:.3f} | {row['mean_reasoning_tokens']:.2f} | {row['mean_repetition_rate']:.4f} |"
        )
    lines.extend(["", "## Examples"])
    for row in result["rows"]:
        completion = row["completion"].replace("\n", "\\n")
        if len(completion) > 260:
            completion = completion[:260] + "..."
        lines.append(
            f"- `{row['condition']}` `{row['problem_id']}` correct={row['correct']} "
            f"answer={row['extracted_answer']!r} gold={row['gold_answer']!r} tokens={row['reasoning_tokens']}: {completion}"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {path.with_suffix('.md')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict-format generation with one-shot prefix activation interpolation patches.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="synthetic_math")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--low-mode", default="instant", choices=["instant", "cot", "verify", "budget_forcing"])
    parser.add_argument("--high-mode", default="verify", choices=["instant", "cot", "verify", "budget_forcing"])
    parser.add_argument("--conditions", default="18:1,20:1,27:0.75,27:1")
    parser.add_argument("--max-new-tokens", type=int, default=96)
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
