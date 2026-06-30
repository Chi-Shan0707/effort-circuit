from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch

from .answer_extractors import extract_answer, is_correct
from .datasets import load_dataset
from .generation import generate_completion, load_model, set_seed
from .intervene import load_vector, residual_steering
from .io_utils import write_table
from .metrics import cost_adjusted_score, count_reasoning_tokens, repetition_rate
from .prompts import build_prompt


def parse_csv_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def first_vector_path(root: str) -> Path:
    preferred = Path(root) / "best_probe.pt"
    if preferred.exists():
        return preferred
    candidates = sorted(Path(root).glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No vector artifacts found in {root}")
    return candidates[0]


def run(args: argparse.Namespace) -> pd.DataFrame:
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    vector, vector_layer = load_vector(str(first_vector_path(args.vectors)))
    problems = load_dataset(args.dataset, args.n, args.seed)
    alphas = parse_csv_floats(args.alphas)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    rows = []
    baseline_correct: dict[str, bool] = {}
    for problem in problems:
        prompt = build_prompt(problem["question"], "cot")
        base_completion = generate_completion(bundle, prompt, args.temperature, args.max_new_tokens)
        base_extracted = extract_answer(base_completion).text
        baseline_correct[problem["problem_id"]] = is_correct(base_extracted, problem["gold_answer"])
    for mode in modes:
        for alpha in alphas:
            for problem in problems:
                prompt = build_prompt(problem["question"], "cot")
                active_vector = vector
                layer = vector_layer
                if mode == "random_direction":
                    generator = torch.Generator().manual_seed(args.seed + int((alpha + 10) * 1000))
                    active_vector = torch.randn(vector.shape, generator=generator)
                    active_vector = active_vector / (active_vector.norm() + 1e-8) * (vector.norm() + 1e-8)
                if mode in {"residual_vector", "random_direction"} and alpha != 0:
                    with residual_steering(bundle.model, layer, active_vector, alpha):
                        completion = generate_completion(bundle, prompt, args.temperature, args.max_new_tokens)
                else:
                    completion = generate_completion(bundle, prompt, args.temperature, args.max_new_tokens)
                extracted = extract_answer(completion).text
                correct = is_correct(extracted, problem["gold_answer"])
                reasoning_tokens, total_tokens = count_reasoning_tokens(bundle.tokenizer, prompt, completion)
                rows.append(
                    {
                        "problem_id": problem["problem_id"],
                        "mode": mode,
                        "alpha": alpha,
                        "layer": layer,
                        "question": problem["question"],
                        "gold_answer": problem["gold_answer"],
                        "completion": completion,
                        "extracted_answer": extracted,
                        "correct": correct,
                        "reasoning_tokens": reasoning_tokens,
                        "total_tokens": total_tokens,
                        "positive_flips": int(correct and not baseline_correct[problem["problem_id"]]),
                        "negative_flips": int((not correct) and baseline_correct[problem["problem_id"]]),
                        "cost_adjusted_score": cost_adjusted_score(float(correct), reasoning_tokens),
                        "repetition_rate": repetition_rate(completion),
                        "malformed_rate": float(extracted is None),
                    }
                )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run intervention sweeps.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="heldout_synthetic_math")
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--neurons", required=True)
    parser.add_argument("--alphas", default="-3,-2,-1,-0.5,0,0.5,1,2,3")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--modes", default="residual_vector,neuron_scale,random_direction")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    df = run(args)
    write_table(df, args.out)
    print(f"wrote {len(df)} sweep rows to {args.out}")


if __name__ == "__main__":
    main()
