from __future__ import annotations

import argparse

import pandas as pd

from .answer_extractors import extract_answer, is_correct
from .datasets import iter_dataset
from .generation import generate_completion, load_model, set_seed
from .io_utils import write_table
from .metrics import count_reasoning_tokens, whitespace_token_count
from .prompts import build_prompt, parse_modes


def fallback_completion(question: str, gold_answer: str, mode: str) -> str:
    if mode == "instant":
        return f" {gold_answer}"
    if mode == "cot":
        return f" Compute the requested arithmetic carefully. Final answer: {gold_answer}"
    if mode == "verify":
        return f" Compute once, verify the same result, and conclude. Final answer: {gold_answer}"
    return f" Plan, compute, check, then answer. Final answer: {gold_answer}"


def collect(args: argparse.Namespace) -> pd.DataFrame:
    set_seed(args.seed)
    bundle = None if args.no_model else load_model(args.model, args.device, args.dtype)
    rows = []
    for problem in iter_dataset(args.dataset, args.n, args.seed):
        for mode in parse_modes(args.modes):
            prompt = build_prompt(problem["question"], mode)
            completion = (
                fallback_completion(problem["question"], problem["gold_answer"], mode)
                if bundle is None
                else generate_completion(bundle, prompt, args.temperature, args.max_new_tokens)
            )
            extracted = extract_answer(completion)
            if bundle is None:
                reasoning_tokens = whitespace_token_count(completion)
                total_tokens = whitespace_token_count(prompt + " " + completion)
            else:
                reasoning_tokens, total_tokens = count_reasoning_tokens(bundle.tokenizer, prompt, completion)
            rows.append(
                {
                    "problem_id": problem["problem_id"],
                    "question": problem["question"],
                    "gold_answer": problem["gold_answer"],
                    "mode": mode,
                    "prompt": prompt,
                    "completion": completion,
                    "extracted_answer": extracted.text,
                    "correct": is_correct(extracted.text, problem["gold_answer"]),
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens,
                    "first_final_position": extracted.first_final_position,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                }
            )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect baseline reasoning traces.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="synthetic_math")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--modes", default="instant,cot,verify,budget_forcing")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--no-model", action="store_true", help="Use deterministic synthetic completions for smoke tests.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    df = collect(args)
    write_table(df, args.out)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
