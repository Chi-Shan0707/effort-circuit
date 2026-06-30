from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .datasets import load_dataset
from .generation import load_model, set_seed
from .hooks import get_layers
from .prompt_steering_experiment import CONTINUE_STRINGS, FINAL_STRINGS, first_token_ids, next_logits, token_mass
from .prompts import build_prompt


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
def intervened_next_logits(bundle, prompt: str, layer: int, edit: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        patched = tensor.clone()
        patched[:, -1, :] = edit(patched[:, -1, :]).to(device=patched.device, dtype=patched.dtype)
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    handle = get_layers(bundle.model)[layer].register_forward_hook(hook)
    try:
        return next_logits(bundle, prompt)
    finally:
        handle.remove()


def first_answer_token_id(tokenizer, answer: str) -> int | None:
    candidates = [str(answer), " " + str(answer)]
    best = None
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if token_ids and (best is None or len(token_ids) < len(best)):
            best = token_ids
    return None if not best else best[0]


def logprob(logits: torch.Tensor, token_id: int | None) -> float | None:
    if token_id is None:
        return None
    return float(F.log_softmax(logits.float(), dim=-1)[token_id].item())


def rank(logits: torch.Tensor, token_id: int | None) -> int | None:
    if token_id is None:
        return None
    values = logits.float()
    return int((values > values[token_id]).sum().item() + 1)


def top_tokens(tokenizer, logits: torch.Tensor, k: int) -> list[str]:
    ids = torch.topk(logits.float(), k=k).indices.tolist()
    return [tokenizer.decode([token_id]).replace("\n", "\\n") for token_id in ids]


def mean_process_directions(bundle, rows: list[dict[str, str]], layers: list[int], low_mode: str, high_mode: str) -> dict[int, torch.Tensor]:
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    for problem in rows:
        low_prompt = build_prompt(problem["question"], low_mode)
        high_prompt = build_prompt(problem["question"], high_mode)
        for layer in layers:
            low_state = capture_hook_state(bundle, low_prompt, layer)
            high_state = capture_hook_state(bundle, high_prompt, layer)
            sums[layer] = sums.get(layer, torch.zeros_like(high_state)) + (high_state - low_state)
            counts[layer] = counts.get(layer, 0) + 1
    return {layer: sums[layer] / counts[layer] for layer in layers}


def score_row(
    bundle,
    problem: dict[str, str],
    source: dict[str, str],
    condition: str,
    layer: int,
    logits: torch.Tensor,
    continue_ids: list[int],
    final_ids: list[int],
) -> dict[str, Any]:
    target_answer_id = first_answer_token_id(bundle.tokenizer, problem["gold_answer"])
    source_answer_id = first_answer_token_id(bundle.tokenizer, source["gold_answer"])
    target_lp = logprob(logits, target_answer_id)
    source_lp = logprob(logits, source_answer_id)
    return {
        "target_problem_id": problem["problem_id"],
        "source_problem_id": source["problem_id"],
        "target_question": problem["question"],
        "source_question": source["question"],
        "target_answer": problem["gold_answer"],
        "source_answer": source["gold_answer"],
        "condition": condition,
        "layer": layer,
        "continue_mass": token_mass(logits, continue_ids),
        "final_mass": token_mass(logits, final_ids),
        "continue_final_gap": token_mass(logits, continue_ids) - token_mass(logits, final_ids),
        "target_answer_logprob": target_lp,
        "source_answer_logprob": source_lp,
        "source_minus_target_logprob": None if target_lp is None or source_lp is None else source_lp - target_lp,
        "target_answer_rank": rank(logits, target_answer_id),
        "source_answer_rank": rank(logits, source_answer_id),
        "top_tokens": top_tokens(bundle.tokenizer, logits, 8),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["condition"], row["layer"]), []).append(row)
    summaries = []
    for (condition, layer), subset in sorted(groups.items()):
        leak_values = [row["source_minus_target_logprob"] for row in subset if row["source_minus_target_logprob"] is not None]
        source_better = [value for value in leak_values if value > 0]
        summaries.append(
            {
                "condition": condition,
                "layer": layer,
                "n": len(subset),
                "mean_continue_final_gap": sum(row["continue_final_gap"] for row in subset) / len(subset),
                "mean_target_answer_logprob": sum(row["target_answer_logprob"] for row in subset) / len(subset),
                "mean_source_answer_logprob": sum(row["source_answer_logprob"] for row in subset) / len(subset),
                "mean_source_minus_target_logprob": sum(leak_values) / len(leak_values) if leak_values else math.nan,
                "source_answer_beats_target_rate": len(source_better) / len(leak_values) if leak_values else math.nan,
                "mean_target_rank": sum(row["target_answer_rank"] for row in subset) / len(subset),
                "mean_source_rank": sum(row["source_answer_rank"] for row in subset) / len(subset),
            }
        )
    return summaries


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    train_rows = load_dataset(args.train_dataset, args.train_n, args.seed)
    eval_rows = load_dataset(args.eval_dataset, args.eval_n + 1, args.seed)
    directions = mean_process_directions(bundle, train_rows, layers, args.low_mode, args.high_mode)
    continue_ids = first_token_ids(bundle.tokenizer, CONTINUE_STRINGS)
    final_ids = first_token_ids(bundle.tokenizer, FINAL_STRINGS)

    rows = []
    pairs = list(zip(eval_rows[:-1], eval_rows[1:]))
    for target, source in pairs:
        target_low_prompt = build_prompt(target["question"], args.low_mode)
        target_high_prompt = build_prompt(target["question"], args.high_mode)
        source_high_prompt = build_prompt(source["question"], args.high_mode)
        low_logits = next_logits(bundle, target_low_prompt)
        for layer in layers:
            target_high_state = capture_hook_state(bundle, target_high_prompt, layer)
            source_high_state = capture_hook_state(bundle, source_high_prompt, layer)
            rows.append(score_row(bundle, target, source, "low", layer, low_logits, continue_ids, final_ids))
            same_logits = intervened_next_logits(bundle, target_low_prompt, layer, lambda _state, replacement=target_high_state: replacement)
            rows.append(score_row(bundle, target, source, "same_question_high_replace", layer, same_logits, continue_ids, final_ids))
            cross_logits = intervened_next_logits(bundle, target_low_prompt, layer, lambda _state, replacement=source_high_state: replacement)
            rows.append(score_row(bundle, target, source, "cross_question_high_replace", layer, cross_logits, continue_ids, final_ids))
            direction_logits = intervened_next_logits(
                bundle,
                target_low_prompt,
                layer,
                lambda state, direction=directions[layer]: state + args.alpha * direction.to(device=state.device, dtype=state.dtype),
            )
            rows.append(score_row(bundle, target, source, "mean_process_direction_add", layer, direction_logits, continue_ids, final_ids))
    return {
        "model": args.model,
        "train_dataset": args.train_dataset,
        "train_n": args.train_n,
        "eval_dataset": args.eval_dataset,
        "eval_n": args.eval_n,
        "layers": layers,
        "low_mode": args.low_mode,
        "high_mode": args.high_mode,
        "alpha": args.alpha,
        "summaries": summarize(rows),
        "rows": rows,
    }


def write_report(result: dict[str, Any], out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Process vs Task Disentanglement",
        "",
        "## Summary",
        "",
        "| condition | layer | n | gap | target lp | source lp | source-target lp | source beats target | target rank | source rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['condition']} | {row['layer']} | {row['n']} | {row['mean_continue_final_gap']:.6f} | "
            f"{row['mean_target_answer_logprob']:.6f} | {row['mean_source_answer_logprob']:.6f} | "
            f"{row['mean_source_minus_target_logprob']:+.6f} | {row['source_answer_beats_target_rate']:.3f} | "
            f"{row['mean_target_rank']:.1f} | {row['mean_source_rank']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "Cross-question replacement should transfer process posture without making the source answer more likely than the target answer. Mean process-direction addition is safer if it preserves posture effects while reducing source-answer leakage.",
        ]
    )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {path.with_suffix('.md')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Separate process posture from task content via cross-question patching and mean-direction addition.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-dataset", default="synthetic_math")
    parser.add_argument("--train-n", type=int, default=8)
    parser.add_argument("--eval-dataset", default="heldout_synthetic_math")
    parser.add_argument("--eval-n", type=int, default=5)
    parser.add_argument("--low-mode", default="instant")
    parser.add_argument("--high-mode", default="verify")
    parser.add_argument("--layers", default="18,27")
    parser.add_argument("--alpha", type=float, default=1.0)
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
