from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .answer_extractors import extract_answer, is_correct
from .datasets import load_dataset
from .generation import load_model, set_seed
from .hooks import get_layers
from .metrics import repetition_rate
from .prompts import build_prompt


STRICT_FINAL_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:Final answer|Final)\s*(?::|is|=)\s*(?P<answer>-?\d+(?:\.\d+)?)",
    re.I,
)
RELAXED_FINAL_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:Final answer|Final|Answer)\s*(?::|is|=)\s*(?P<answer>-?\d+(?:\.\d+)?)",
    re.I,
)


def strict_prompt(question: str) -> str:
    return (
        "Solve the arithmetic problem. Use this exact format once:\n"
        "Reasoning: <brief arithmetic>\n"
        "Final answer: <number>\n\n"
        f"Question: {question}\n"
        "Reasoning:"
    )


class StopAfterFinalAnswer(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int, final_line_re: re.Pattern[str]):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.final_line_re = final_line_re
        self.matched_text: str | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[0, self.prompt_len :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        match = self.final_line_re.search(text)
        if match:
            self.matched_text = text[: match.end()]
            return True
        return False


@torch.inference_mode()
def capture_hook_state(bundle, prompt: str, layer: int) -> torch.Tensor:
    captured = {}

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


def final_line_regex(relaxed: bool) -> re.Pattern[str]:
    return RELAXED_FINAL_LINE_RE if relaxed else STRICT_FINAL_LINE_RE


@torch.inference_mode()
def generate_stop_after_final(
    bundle,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    layer: int | None = None,
    replacement: torch.Tensor | None = None,
    relaxed_final_markers: bool = False,
):
    tokenizer = bundle.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(bundle.device)
    prompt_len = inputs["input_ids"].shape[1]
    stopper = StopAfterFinalAnswer(tokenizer, prompt_len, final_line_regex(relaxed_final_markers))
    handles = []
    used = {"done": False}
    if layer is not None and replacement is not None:
        target = replacement.to(bundle.device)

        def patch_hook(_module, _inputs, output):
            if used["done"]:
                return output
            tensor = output[0] if isinstance(output, tuple) else output
            patched = tensor.clone()
            patched[:, -1, :] = target.to(dtype=patched.dtype)
            used["done"] = True
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handles.append(get_layers(bundle.model)[layer].register_forward_hook(patch_hook))
    try:
        do_sample = temperature > 0
        output_ids = bundle.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            max_new_tokens=max_new_tokens,
            stopping_criteria=StoppingCriteriaList([stopper]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    finally:
        for handle in handles:
            handle.remove()
    generated_ids = output_ids[0, prompt_len:]
    full_completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    stopped_completion = stopper.matched_text or full_completion
    return stopped_completion, full_completion, stopper.matched_text is not None


def parse_conditions(value: str) -> list[tuple[str, int | None, float | None]]:
    parsed: list[tuple[str, int | None, float | None]] = [("baseline", None, None)]
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        layer_text, t_text = item.split(":", 1)
        parsed.append((f"layer{layer_text}_t{t_text}", int(layer_text), float(t_text)))
    return parsed


def answer_changes(text: str) -> int:
    answers = []
    for match in re.finditer(r"(?:Final answer|Final)\s*(?::|is|=)\s*([^\n]+)", text, re.I):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        if numbers:
            answers.append(numbers[-1])
    return max(0, len(set(answers)) - 1)


def extract_first_final(text: str, relaxed: bool = False) -> str | None:
    match = final_line_regex(relaxed).search(text)
    if not match:
        return None
    return match.group("answer")


def extract_first_strict_final(text: str) -> str | None:
    return extract_first_final(text, relaxed=False)


def run(args: argparse.Namespace) -> dict:
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    problems = load_dataset(args.dataset, args.n, args.seed)
    conditions = parse_conditions(args.conditions)
    rows = []
    for problem in problems:
        prompt = strict_prompt(problem["question"])
        hook_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for _condition, layer, _t in conditions:
            if layer is not None and layer not in hook_states:
                low_state = capture_hook_state(bundle, build_prompt(problem["question"], args.low_mode), layer)
                high_state = capture_hook_state(bundle, build_prompt(problem["question"], args.high_mode), layer)
                hook_states[layer] = (low_state, high_state)
        for condition, layer, t in conditions:
            replacement = None
            if layer is not None and t is not None:
                low_state, high_state = hook_states[layer]
                replacement = (1.0 - t) * low_state + t * high_state
            stopped_completion, full_completion, stopped = generate_stop_after_final(
                bundle,
                prompt,
                args.max_new_tokens,
                args.temperature,
                layer,
                replacement,
                relaxed_final_markers=args.relaxed_final_markers,
            )
            first_answer = extract_first_final(stopped_completion, relaxed=args.relaxed_final_markers)
            last_answer = extract_answer(full_completion).text
            rows.append(
                {
                    "problem_id": problem["problem_id"],
                    "question": problem["question"],
                    "gold_answer": problem["gold_answer"],
                    "condition": condition,
                    "layer": layer,
                    "t": t,
                    "stopped": stopped,
                    "completion": stopped_completion,
                    "full_completion": full_completion,
                    "first_final_answer": first_answer,
                    "last_final_answer": last_answer,
                    "first_final_correct": is_correct(first_answer, problem["gold_answer"]),
                    "last_final_correct": is_correct(last_answer, problem["gold_answer"]),
                    "generated_tokens": len(bundle.tokenizer.encode(stopped_completion, add_special_tokens=False)),
                    "full_generated_tokens": len(bundle.tokenizer.encode(full_completion, add_special_tokens=False)),
                    "answer_changes": answer_changes(full_completion),
                    "post_final_continuation": int(stopped and stopped_completion != full_completion),
                    "repetition_rate": repetition_rate(stopped_completion),
                }
            )
    summaries = []
    for condition, _, _ in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        summaries.append(
            {
                "condition": condition,
                "first_final_accuracy": sum(row["first_final_correct"] for row in subset) / len(subset),
                "last_final_accuracy": sum(row["last_final_correct"] for row in subset) / len(subset),
                "stop_rate": sum(row["stopped"] for row in subset) / len(subset),
                "malformed_rate": sum(row["first_final_answer"] in (None, "") for row in subset) / len(subset),
                "mean_generated_tokens": sum(row["generated_tokens"] for row in subset) / len(subset),
                "mean_answer_changes": sum(row["answer_changes"] for row in subset) / len(subset),
                "mean_repetition_rate": sum(row["repetition_rate"] for row in subset) / len(subset),
            }
        )
    return {
        "model": args.model,
        "dataset": args.dataset,
        "n": args.n,
        "low_mode": args.low_mode,
        "high_mode": args.high_mode,
        "conditions": [condition for condition, _, _ in conditions],
        "final_marker_protocol": "relaxed" if args.relaxed_final_markers else "strict",
        "summaries": summaries,
        "rows": rows,
    }


def write_report(result: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Stop-After-First-Final Experiment",
        "",
        f"Final marker protocol: `{result.get('final_marker_protocol', 'strict')}`",
        "",
        "## Summary",
        "",
        "| condition | first-final acc | last-detectable acc | strict/relaxed stop rate | malformed rate | mean tokens | answer changes | repetition |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['condition']} | {row['first_final_accuracy']:.3f} | {row['last_final_accuracy']:.3f} | "
            f"{row['stop_rate']:.3f} | {row['malformed_rate']:.3f} | {row['mean_generated_tokens']:.2f} | {row['mean_answer_changes']:.2f} | "
            f"{row['mean_repetition_rate']:.4f} |"
        )
    lines.extend(["", "## Examples"])
    for row in result["rows"]:
        text = row["completion"].replace("\n", "\\n")
        if len(text) > 220:
            text = text[:220] + "..."
        lines.append(
            f"- `{row['condition']}` `{row['problem_id']}` first={row['first_final_answer']!r} "
            f"gold={row['gold_answer']!r} correct={row['first_final_correct']} tokens={row['generated_tokens']}: {text}"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {path.with_suffix('.md')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate until first valid final answer, optionally prefix-patching hook-captured states.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="heldout_synthetic_math")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--low-mode", default="instant")
    parser.add_argument("--high-mode", default="verify")
    parser.add_argument("--conditions", default="18:1,27:0.75,27:1")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--relaxed-final-markers",
        action="store_true",
        help="Treat generic 'Answer:' as a final marker. Default is strict: only 'Final answer:' or 'Final:' count.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    write_report(run(args), args.out)


if __name__ == "__main__":
    main()
