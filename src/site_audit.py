from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .datasets import load_dataset
from .generation import load_model, set_seed
from .hooks import get_layers
from .prompts import build_prompt


def max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


@torch.inference_mode()
def logits_and_hidden_states(bundle, prompt: str):
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
    outputs = bundle.model(**inputs, output_hidden_states=True, use_cache=False)
    return outputs.logits[0, -1, :].detach().cpu(), [state[0, -1, :].detach().cpu() for state in outputs.hidden_states[1:]]


@torch.inference_mode()
def capture_hook_output(bundle, prompt: str, layer: int):
    captured = {}

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["state"] = tensor[0, -1, :].detach().cpu()

    handle = get_layers(bundle.model)[layer].register_forward_hook(hook)
    try:
        logits = run_logits(bundle, prompt)
    finally:
        handle.remove()
    return logits, captured["state"]


@torch.inference_mode()
def run_logits(bundle, prompt: str) -> torch.Tensor:
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
    outputs = bundle.model(**inputs, use_cache=False)
    return outputs.logits[0, -1, :].detach().cpu()


@torch.inference_mode()
def patched_logits(bundle, prompt: str, layer: int, replacement: torch.Tensor) -> torch.Tensor:
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
        return run_logits(bundle, prompt)
    finally:
        handle.remove()


def run(args: argparse.Namespace) -> dict:
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    bundle = load_model(args.model, args.device, args.dtype)
    problems = load_dataset(args.dataset, args.n, args.seed)
    layers = [int(item) for item in args.layers.split(",") if item.strip()]
    rows = []
    for problem in problems:
        for mode in [args.low_mode, args.high_mode]:
            prompt = build_prompt(problem["question"], mode)
            base_logits, hidden_states = logits_and_hidden_states(bundle, prompt)
            for layer in layers:
                hook_base_logits, hook_state = capture_hook_output(bundle, prompt, layer)
                self_patch_logits = patched_logits(bundle, prompt, layer, hook_state)
                hidden_patch_logits = patched_logits(bundle, prompt, layer, hidden_states[layer])
                rows.append(
                    {
                        "problem_id": problem["problem_id"],
                        "mode": mode,
                        "layer": layer,
                        "base_vs_hook_run_logit_max_abs_delta": max_abs_delta(base_logits, hook_base_logits),
                        "self_hook_patch_logit_max_abs_delta": max_abs_delta(base_logits, self_patch_logits),
                        "hidden_state_patch_logit_max_abs_delta": max_abs_delta(base_logits, hidden_patch_logits),
                        "hidden_state_vs_hook_state_max_abs_delta": max_abs_delta(hidden_states[layer], hook_state),
                    }
                )
    summaries = []
    for layer in layers:
        subset = [row for row in rows if row["layer"] == layer]
        summaries.append(
            {
                "layer": layer,
                "mean_passive_hook_logit_delta": sum(row["base_vs_hook_run_logit_max_abs_delta"] for row in subset) / len(subset),
                "max_passive_hook_logit_delta": max(row["base_vs_hook_run_logit_max_abs_delta"] for row in subset),
                "mean_self_hook_patch_logit_delta": sum(row["self_hook_patch_logit_max_abs_delta"] for row in subset) / len(subset),
                "max_self_hook_patch_logit_delta": max(row["self_hook_patch_logit_max_abs_delta"] for row in subset),
                "mean_hidden_state_patch_logit_delta": sum(row["hidden_state_patch_logit_max_abs_delta"] for row in subset) / len(subset),
                "max_hidden_state_patch_logit_delta": max(row["hidden_state_patch_logit_max_abs_delta"] for row in subset),
                "mean_hidden_state_vs_hook_state_delta": sum(row["hidden_state_vs_hook_state_max_abs_delta"] for row in subset) / len(subset),
                "max_hidden_state_vs_hook_state_delta": max(row["hidden_state_vs_hook_state_max_abs_delta"] for row in subset),
            }
        )
    return {
        "model": args.model,
        "dataset": args.dataset,
        "n": args.n,
        "layers": layers,
        "low_mode": args.low_mode,
        "high_mode": args.high_mode,
        "summaries": summaries,
        "rows": rows,
    }


def write_report(result: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Activation Site Audit",
        "",
        "## Purpose",
        "",
        "Audit whether `outputs.hidden_states[layer+1]` can be safely patched into decoder block hook output at the same layer.",
        "",
        "## Summary",
        "",
        "| layer | passive hook logit Δ max | self-hook patch logit Δ max | hidden-states→hook logit Δ max | hidden-states vs hook-state Δ max |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['layer']} | {row['max_passive_hook_logit_delta']:.6g} | {row['max_self_hook_patch_logit_delta']:.6g} | "
            f"{row['max_hidden_state_patch_logit_delta']:.6g} | {row['max_hidden_state_vs_hook_state_delta']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Checks Reported",
            "",
            "- Passive hook identity check: `base_vs_hook_run_logit_max_abs_delta`.",
            "- Self-hook patch identity check: `self_hook_patch_logit_max_abs_delta`.",
            "- Hidden-states-to-hook equivalence check: `hidden_state_patch_logit_max_abs_delta` and `hidden_state_vs_hook_state_max_abs_delta`.",
            "- Per-layer max logit deltas and max state deltas are summarized above and retained per row in JSON.",
            "",
            "## Current Run Note",
            "",
            "In the current archived audit (`outputs/site_audit_n2_layers18_27.*`), layer 18 passed site-equivalence while layer 27 failed site-equivalence. Therefore old layer27 hidden-state patch conclusions are downgraded to site-mismatch/off-manifold/readout-boundary interventions unless rerun with hook-captured decoder block states.",
            "",
            "## Interpretation Guide",
            "",
            "- Passive hook logit delta should be near zero; otherwise merely observing the hook changes the run.",
            "- Self hook patch logit delta should be near zero; otherwise the hook/patch machinery is not identity-preserving.",
            "- Hidden-state patch delta near zero means HuggingFace hidden state and hook output are equivalent for that layer/site.",
            "- Large hidden-state-vs-hook delta means prior patching mixed incompatible activation sites and must be interpreted as a site-mismatch intervention.",
        ]
    )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and {path.with_suffix('.md')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit hidden_states vs decoder block hook output patch equivalence.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="synthetic_math")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--layers", default="18,27")
    parser.add_argument("--low-mode", default="instant")
    parser.add_argument("--high-mode", default="verify")
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
