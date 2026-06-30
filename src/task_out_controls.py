from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .generation import load_model, set_seed
from .hooks import get_layers
from .prompt_steering_experiment import CONTINUE_STRINGS, FINAL_STRINGS, first_token_ids, next_logits, token_mass

VERBOSITY_STRINGS = ["Because", "This", "In", "There", "The", "It", "To", "Let"]
FORMAT_STRINGS = ["Reasoning", "Final", "Answer", "Step"]


@dataclass(frozen=True)
class ControlTask:
    task_id: str
    family: str
    subtype: str
    question: str
    gold_answer: str


def build_control_tasks(n_per_subtype: int, seed: int) -> list[ControlTask]:
    rng = random.Random(seed)
    tasks: list[ControlTask] = []

    for index in range(n_per_subtype):
        a, b, c = rng.randint(3, 30), rng.randint(2, 20), rng.randint(2, 12)
        answer = a + b * c
        tasks.append(
            ControlTask(
                f"arithmetic-{index:06d}",
                "reasoning",
                "arithmetic",
                f"Mia has {a} tokens and gets {b} more tokens for each of {c} rounds. How many tokens does she have?",
                str(answer),
            )
        )

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for index in range(n_per_subtype):
        start = rng.randrange(len(weekdays))
        delta = rng.randint(2, 17)
        answer = weekdays[(start + delta) % 7]
        tasks.append(
            ControlTask(
                f"date-{index:06d}",
                "reasoning",
                "date_arithmetic",
                f"If today is {weekdays[start]}, what day of the week will it be in {delta} days?",
                answer,
            )
        )

    for index in range(n_per_subtype):
        a, b = rng.randint(2, 9), rng.randint(10, 30)
        answer = b - a
        tasks.append(
            ControlTask(
                f"symbolic-{index:06d}",
                "reasoning",
                "symbolic",
                f"If z + {a} = {b}, what is z?",
                str(answer),
            )
        )

    capitals = [
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Canada", "Ottawa"),
        ("Brazil", "Brasilia"),
        ("Kenya", "Nairobi"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
    ]
    for index, (country, capital) in enumerate(capitals[:n_per_subtype]):
        tasks.append(
            ControlTask(
                f"capital-{index:06d}",
                "non_reasoning",
                "capital_lookup",
                f"What is the capital of {country}?",
                capital,
            )
        )

    translations = [
        ("hello", "hola"),
        ("thank you", "gracias"),
        ("good night", "buenas noches"),
        ("water", "agua"),
        ("book", "libro"),
    ]
    for index, (english, spanish) in enumerate(translations[:n_per_subtype]):
        tasks.append(
            ControlTask(
                f"translation-{index:06d}",
                "non_reasoning",
                "translation",
                f"Translate to Spanish: {english}",
                spanish,
            )
        )

    sentiments = [
        ("The service was slow and confusing.", "positive"),
        ("The room was noisy and uncomfortable.", "positive"),
        ("The instructions were unclear and frustrating.", "positive"),
        ("The meal was cold and bland.", "positive"),
        ("The app crashed twice during setup.", "positive"),
    ]
    for index, (sentence, sentiment) in enumerate(sentiments[:n_per_subtype]):
        tasks.append(
            ControlTask(
                f"sentiment-{index:06d}",
                "non_reasoning",
                "sentiment_rewrite",
                f"Rewrite this sentence to sound {sentiment}: {sentence}",
                sentiment,
            )
        )
    return tasks


def build_prompt(question: str, mode: str) -> str:
    if mode == "low":
        return f"Answer concisely.\nQuestion: {question}\nAnswer:"
    if mode == "high_reasoning":
        return (
            "Solve carefully, verify the reasoning once, and then give the final answer.\n"
            f"Question: {question}\nReasoning:"
        )
    if mode == "high_verbosity":
        return (
            "Explain verbosely in complete sentences, but do not solve step by step.\n"
            f"Question: {question}\nExplanation:"
        )
    if mode == "high_format":
        return (
            "Use exactly this format, without extra verification unless needed:\n"
            "Reasoning: <brief note>\nFinal answer: <answer>\n\n"
            f"Question: {question}\nReasoning:"
        )
    raise ValueError(f"unknown control prompt mode {mode!r}")


@torch.inference_mode()
def capture_hook_state(bundle, prompt: str, layer: int) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["state"] = tensor[0, -1, :].detach().cpu()

    handle = get_layers(bundle.model)[layer].register_forward_hook(hook)
    try:
        inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
        _ = bundle.model(**inputs, use_cache=False)
    finally:
        handle.remove()
    return captured["state"]


@torch.inference_mode()
def patched_next_logits(bundle, prompt: str, layer: int, replacement: torch.Tensor) -> torch.Tensor:
    target = replacement.to(bundle.device)

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        patched = tensor.clone()
        patched[:, -1, :] = target.to(dtype=patched.dtype)
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    handle = get_layers(bundle.model)[layer].register_forward_hook(hook)
    try:
        return next_logits(bundle, prompt)
    finally:
        handle.remove()


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["family"], row["source_mode"], row["layer"]), []).append(row)
    summaries = []
    for (family, source_mode, layer), subset in sorted(groups.items()):
        summaries.append(
            {
                "family": family,
                "source_mode": source_mode,
                "layer": layer,
                "n": len(subset),
                "mean_low_gap": sum(row["low_gap"] for row in subset) / len(subset),
                "mean_source_gap": sum(row["source_gap"] for row in subset) / len(subset),
                "mean_patched_gap": sum(row["patched_gap"] for row in subset) / len(subset),
                "mean_delta_gap": sum(row["delta_gap"] for row in subset) / len(subset),
                "mean_delta_verbosity_mass": sum(row["delta_verbosity_mass"] for row in subset) / len(subset),
                "mean_delta_format_mass": sum(row["delta_format_mass"] for row in subset) / len(subset),
            }
        )
    return summaries


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    continue_ids = first_token_ids(bundle.tokenizer, CONTINUE_STRINGS)
    final_ids = first_token_ids(bundle.tokenizer, FINAL_STRINGS)
    verbosity_ids = first_token_ids(bundle.tokenizer, VERBOSITY_STRINGS)
    format_ids = first_token_ids(bundle.tokenizer, FORMAT_STRINGS)
    layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    source_modes = [item.strip() for item in args.source_modes.split(",") if item.strip()]
    tasks = build_control_tasks(args.n_per_subtype, args.seed)

    rows = []
    for task in tasks:
        low_prompt = build_prompt(task.question, "low")
        low_logits = next_logits(bundle, low_prompt)
        low_gap = token_mass(low_logits, continue_ids) - token_mass(low_logits, final_ids)
        low_verbosity = token_mass(low_logits, verbosity_ids)
        low_format = token_mass(low_logits, format_ids)
        for layer in layers:
            for source_mode in source_modes:
                source_prompt = build_prompt(task.question, source_mode)
                source_state = capture_hook_state(bundle, source_prompt, layer)
                source_logits = next_logits(bundle, source_prompt)
                patched_logits = patched_next_logits(bundle, low_prompt, layer, source_state)
                source_gap = token_mass(source_logits, continue_ids) - token_mass(source_logits, final_ids)
                patched_gap = token_mass(patched_logits, continue_ids) - token_mass(patched_logits, final_ids)
                patched_verbosity = token_mass(patched_logits, verbosity_ids)
                patched_format = token_mass(patched_logits, format_ids)
                rows.append(
                    {
                        **asdict(task),
                        "layer": layer,
                        "source_mode": source_mode,
                        "low_gap": low_gap,
                        "source_gap": source_gap,
                        "patched_gap": patched_gap,
                        "delta_gap": patched_gap - low_gap,
                        "low_verbosity_mass": low_verbosity,
                        "patched_verbosity_mass": patched_verbosity,
                        "delta_verbosity_mass": patched_verbosity - low_verbosity,
                        "low_format_mass": low_format,
                        "patched_format_mass": patched_format,
                        "delta_format_mass": patched_format - low_format,
                    }
                )
    return {
        "model": args.model,
        "n_per_subtype": args.n_per_subtype,
        "layers": layers,
        "source_modes": source_modes,
        "token_sets": {
            "continue": CONTINUE_STRINGS,
            "final": FINAL_STRINGS,
            "verbosity": VERBOSITY_STRINGS,
            "format": FORMAT_STRINGS,
        },
        "summaries": summarize_rows(rows),
        "rows": rows,
    }


def write_report(result: dict[str, Any], out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Task-Out and Verbosity Controls",
        "",
        "## Summary",
        "",
        "| family | source mode | layer | n | low gap | source gap | patched gap | delta gap | delta verbosity | delta format |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['family']} | {row['source_mode']} | {row['layer']} | {row['n']} | "
            f"{row['mean_low_gap']:.6f} | {row['mean_source_gap']:.6f} | {row['mean_patched_gap']:.6f} | "
            f"{row['mean_delta_gap']:+.6f} | {row['mean_delta_verbosity_mass']:+.6f} | "
            f"{row['mean_delta_format_mass']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "A useful-effort interpretation requires a larger reasoning-family gain than non-reasoning or verbosity controls. If non-reasoning controls move similarly, the intervention is better described as generic generation posture control.",
        ]
    )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {path.with_suffix('.md')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run task-out, verbosity, and format controls for hook-state patching.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-per-subtype", type=int, default=2)
    parser.add_argument("--layers", default="18,27")
    parser.add_argument("--source-modes", default="high_reasoning,high_verbosity,high_format")
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
