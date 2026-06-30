from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .datasets import load_dataset
from .generation import load_model, set_seed
from .hooks import get_layers
from .prompt_steering_experiment import first_token_ids, next_logits, token_mass
from .prompts import build_prompt

FINAL_STRINGS = ["Final", "Answer", "Therefore", "Thus", "So"]


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


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id]).replace("\n", "\\n")


def usable_token(tokenizer, token_id: int) -> bool:
    text = tokenizer.decode([token_id])
    if not text:
        return False
    if token_id in set(filter(lambda value: value is not None, [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id])):
        return False
    return not text.isspace()


def select_clusters(
    bundle,
    train_rows: list[dict[str, str]],
    low_mode: str,
    high_mode: str,
    top_k: int,
    min_mean_prob: float,
) -> dict[str, Any]:
    low_logprobs = []
    high_logprobs = []
    for problem in train_rows:
        low_logits = next_logits(bundle, build_prompt(problem["question"], low_mode))
        high_logits = next_logits(bundle, build_prompt(problem["question"], high_mode))
        low_logprobs.append(F.log_softmax(low_logits.float(), dim=-1).cpu())
        high_logprobs.append(F.log_softmax(high_logits.float(), dim=-1).cpu())
    mean_low = torch.stack(low_logprobs).mean(dim=0)
    mean_high = torch.stack(high_logprobs).mean(dim=0)
    mean_low_prob = mean_low.exp()
    mean_high_prob = mean_high.exp()
    score = mean_high - mean_low
    eligible = torch.maximum(mean_low_prob, mean_high_prob) >= min_mean_prob

    positive = torch.argsort(score, descending=True).tolist()
    negative = torch.argsort(score, descending=False).tolist()
    continue_ids = [
        token_id
        for token_id in positive
        if score[token_id] > 0 and bool(eligible[token_id]) and usable_token(bundle.tokenizer, token_id)
    ][:top_k]
    final_ids = [
        token_id
        for token_id in negative
        if score[token_id] < 0 and bool(eligible[token_id]) and usable_token(bundle.tokenizer, token_id)
    ][:top_k]
    return {
        "continue_ids": continue_ids,
        "final_ids": final_ids,
        "continue_tokens": [token_text(bundle.tokenizer, token_id) for token_id in continue_ids],
        "final_tokens": [token_text(bundle.tokenizer, token_id) for token_id in final_ids],
        "continue_scores": [float(score[token_id]) for token_id in continue_ids],
        "final_scores": [float(score[token_id]) for token_id in final_ids],
        "continue_mean_high_prob": [float(mean_high_prob[token_id]) for token_id in continue_ids],
        "continue_mean_low_prob": [float(mean_low_prob[token_id]) for token_id in continue_ids],
        "final_mean_high_prob": [float(mean_high_prob[token_id]) for token_id in final_ids],
        "final_mean_low_prob": [float(mean_low_prob[token_id]) for token_id in final_ids],
        "min_mean_prob": min_mean_prob,
    }


def parse_conditions(value: str) -> list[tuple[str, int | None, float | None]]:
    conditions: list[tuple[str, int | None, float | None]] = [("low", None, None), ("high", None, None)]
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        layer_text, t_text = item.split(":", 1)
        conditions.append((f"layer{layer_text}_t{t_text}", int(layer_text), float(t_text)))
    return conditions


def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    return float((p * (log_p - log_q)).sum().item())


def first_answer_token_id(tokenizer, answer: str) -> int | None:
    candidates = [str(answer), " " + str(answer)]
    best: list[int] | None = None
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if token_ids and (best is None or len(token_ids) < len(best)):
            best = token_ids
    return None if not best else best[0]


def answer_margin(logits: torch.Tensor, answer_token_id: int | None) -> float | None:
    if answer_token_id is None:
        return None
    logprobs = F.log_softmax(logits.float(), dim=-1)
    answer_value = logprobs[answer_token_id]
    masked = logprobs.clone()
    masked[answer_token_id] = -math.inf
    return float((answer_value - masked.max()).item())


def eos_final_margin(bundle, logits: torch.Tensor, final_ids: list[int], continue_ids: list[int]) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    eos_mass = float(probs[bundle.tokenizer.eos_token_id].item()) if bundle.tokenizer.eos_token_id is not None else 0.0
    return eos_mass + token_mass(logits, final_ids) - token_mass(logits, continue_ids)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    summaries = []
    for condition, subset in groups.items():
        answer_values = [row["answer_margin"] for row in subset if row["answer_margin"] is not None]
        summaries.append(
            {
                "condition": condition,
                "n": len(subset),
                "mean_cluster_gap": sum(row["cluster_gap"] for row in subset) / len(subset),
                "mean_continue_mass": sum(row["continue_mass"] for row in subset) / len(subset),
                "mean_final_mass": sum(row["final_mass"] for row in subset) / len(subset),
                "mean_kl_vs_low": sum(row["kl_vs_low"] for row in subset) / len(subset),
                "mean_answer_margin": sum(answer_values) / len(answer_values) if answer_values else None,
                "mean_eos_final_margin": sum(row["eos_final_margin"] for row in subset) / len(subset),
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
    clusters = select_clusters(bundle, train_rows, args.low_mode, args.high_mode, args.top_k, args.min_mean_prob)
    conditions = parse_conditions(args.conditions)
    manual_final_ids = first_token_ids(bundle.tokenizer, FINAL_STRINGS)

    rows = []
    for problem in eval_rows:
        low_prompt = build_prompt(problem["question"], args.low_mode)
        high_prompt = build_prompt(problem["question"], args.high_mode)
        low_logits = next_logits(bundle, low_prompt)
        high_logits = next_logits(bundle, high_prompt)
        states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        answer_id = first_answer_token_id(bundle.tokenizer, problem["gold_answer"])
        for _name, layer, _t in conditions:
            if layer is not None and layer not in states:
                states[layer] = (
                    capture_hook_state(bundle, low_prompt, layer),
                    capture_hook_state(bundle, high_prompt, layer),
                )
        for condition, layer, t in conditions:
            if condition == "low":
                logits = low_logits
            elif condition == "high":
                logits = high_logits
            else:
                low_state, high_state = states[layer]
                replacement = (1.0 - t) * low_state + t * high_state
                logits = patched_next_logits(bundle, low_prompt, layer, replacement)
            continue_mass = token_mass(logits, clusters["continue_ids"])
            final_mass = token_mass(logits, clusters["final_ids"])
            rows.append(
                {
                    "problem_id": problem["problem_id"],
                    "question": problem["question"],
                    "gold_answer": problem["gold_answer"],
                    "condition": condition,
                    "layer": layer,
                    "t": t,
                    "continue_mass": continue_mass,
                    "final_mass": final_mass,
                    "cluster_gap": continue_mass - final_mass,
                    "kl_vs_low": kl_divergence(logits, low_logits),
                    "kl_high_vs_low": kl_divergence(high_logits, low_logits),
                    "answer_token_id": answer_id,
                    "answer_margin": answer_margin(logits, answer_id),
                    "eos_final_margin": eos_final_margin(bundle, logits, manual_final_ids, clusters["continue_ids"]),
                }
            )
    return {
        "model": args.model,
        "train_dataset": args.train_dataset,
        "train_n": args.train_n,
        "eval_dataset": args.eval_dataset,
        "eval_n": args.eval_n,
        "low_mode": args.low_mode,
        "high_mode": args.high_mode,
        "clusters": clusters,
        "conditions": [condition for condition, _, _ in conditions],
        "summaries": summarize(rows),
        "rows": rows,
    }


def write_report(result: dict[str, Any], out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Data-Driven Token Metric",
        "",
        "## Learned Token Clusters",
        "",
        f"- Continue/high cluster: `{result['clusters']['continue_tokens']}`",
        f"- Final/low cluster: `{result['clusters']['final_tokens']}`",
        "",
        "## Heldout Summary",
        "",
        "| condition | n | cluster gap | continue mass | final mass | KL vs low | answer margin | EOS/Final margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summaries"]:
        answer = "nan" if row["mean_answer_margin"] is None else f"{row['mean_answer_margin']:.6f}"
        lines.append(
            f"| {row['condition']} | {row['n']} | {row['mean_cluster_gap']:.6f} | "
            f"{row['mean_continue_mass']:.6f} | {row['mean_final_mass']:.6f} | {row['mean_kl_vs_low']:.6f} | "
            f"{answer} | {row['mean_eos_final_margin']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "Clusters are selected only on the train split and evaluated on heldout rows. A useful metric should remain interpretable when combined with KL, answer-token margin, and EOS/Final margin; a high cluster gap alone is not a task-utility result.",
        ]
    )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {path.with_suffix('.md')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learn high-vs-low discriminative token clusters on train and evaluate on heldout interventions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-dataset", default="synthetic_math")
    parser.add_argument("--train-n", type=int, default=8)
    parser.add_argument("--eval-dataset", default="heldout_synthetic_math")
    parser.add_argument("--eval-n", type=int, default=5)
    parser.add_argument("--low-mode", default="instant")
    parser.add_argument("--high-mode", default="verify")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--min-mean-prob", type=float, default=1e-4)
    parser.add_argument("--conditions", default="18:1,27:0.75,27:1")
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
