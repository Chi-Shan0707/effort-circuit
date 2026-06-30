from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .answer_extractors import extract_answer, is_correct
from .datasets import load_dataset
from .generation import generate_completion, load_model, set_seed
from .intervene import load_vector, residual_steering
from .io_utils import write_table
from .metrics import count_reasoning_tokens, repetition_rate
from .prompts import build_prompt


def run(args: argparse.Namespace) -> pd.DataFrame:
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    effort_vector, layer = load_vector(args.effort_vector)
    probe_vector, _ = load_vector(args.probe)
    effort_norm = float(effort_vector.norm().item() + 1e-8)
    target = np.percentile([effort_norm], args.target_percentile)
    rows = []
    for problem in load_dataset(args.dataset, args.n, args.seed):
        for controller in ["instant", "cot", "budget_forcing", "static_steering", "cet"]:
            prompt_mode = "instant" if controller == "instant" else ("budget_forcing" if controller == "budget_forcing" else "cot")
            prompt = build_prompt(problem["question"], prompt_mode)
            alpha = 0.0
            if controller == "static_steering":
                alpha = args.alpha_max / 2
            elif controller == "cet":
                alpha = args.alpha_max if effort_norm < target or args.soft_budget > 0 else 0.0
            if alpha:
                with residual_steering(bundle.model, layer, effort_vector, alpha):
                    completion = generate_completion(bundle, prompt, args.temperature, min(args.hard_budget, args.max_new_tokens))
            else:
                completion = generate_completion(bundle, prompt, args.temperature, min(args.hard_budget, args.max_new_tokens))
            extracted = extract_answer(completion).text
            reasoning_tokens, total_tokens = count_reasoning_tokens(bundle.tokenizer, prompt, completion)
            if controller == "cet" and reasoning_tokens > args.soft_budget and extracted is not None:
                alpha = 0.0
            rows.append(
                {
                    "problem_id": problem["problem_id"],
                    "controller": controller,
                    "alpha": alpha,
                    "layer": layer,
                    "effort_score": effort_norm,
                    "probe_score": float(probe_vector.norm().item()),
                    "question": problem["question"],
                    "gold_answer": problem["gold_answer"],
                    "completion": completion,
                    "extracted_answer": extracted,
                    "correct": is_correct(extracted, problem["gold_answer"]),
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens,
                    "repetition_rate": repetition_rate(completion),
                    "malformed_rate": float(extracted is None),
                }
            )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CET controller evaluation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="heldout_synthetic_math")
    parser.add_argument("--effort-vector", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--target-percentile", type=float, default=75)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--soft-budget", type=int, default=256)
    parser.add_argument("--hard-budget", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    df = run(args)
    write_table(df, args.out)
    print(f"wrote {len(df)} CET rows to {args.out}")


if __name__ == "__main__":
    main()
