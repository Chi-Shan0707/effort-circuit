from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .io_utils import read_table


def markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = [["" if pd.isna(value) else str(round(value, 4)) if isinstance(value, float) else str(value) for value in row] for row in df.to_numpy()]
    widths = [len(col) for col in cols]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    header = "| " + " | ".join(col.ljust(width) for col, width in zip(cols, widths)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def table(df: pd.DataFrame, group_cols: list[str]) -> str:
    group_cols = [col for col in group_cols if col in df]
    metric_cols = [col for col in ["correct", "reasoning_tokens", "total_tokens", "positive_flips", "negative_flips", "cost_adjusted_score", "repetition_rate", "malformed_rate"] if col in df]
    if not metric_cols:
        return "_No compatible metric columns._\n"
    grouped = df.groupby(group_cols)[metric_cols].mean(numeric_only=True).reset_index() if group_cols else df[metric_cols].mean(numeric_only=True).to_frame().T
    return markdown(grouped)


def analyze(args: argparse.Namespace) -> None:
    frames = []
    for path in args.inputs:
        df = read_table(path)
        df["source"] = Path(path).name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sections = ["# effort-circuit report\n"]
    sections.append("## Baseline table\n")
    if "controller" in data:
        sections.append(table(data[data["controller"].isin(["instant", "cot", "budget_forcing"])], ["source", "controller"]))
    elif "mode" in data:
        sections.append(table(data, ["source", "mode"]))
    sections.append("\n## Layer × alpha heatmaps\n")
    sections.append(table(data, [col for col in ["source", "mode", "layer", "alpha"] if col in data]))
    sections.append("\n## Length-vs-accuracy Pareto frontier\n")
    sections.append(table(data, [col for col in ["source", "mode", "controller"] if col in data]))
    sections.append("\n## Positive/negative flip analysis\n")
    sections.append(table(data, [col for col in ["source", "mode", "alpha"] if col in data]))
    sections.append("\n## Random direction percentile\n")
    if "mode" in data and "random_direction" in set(data["mode"]):
        random_acc = data[data["mode"] == "random_direction"]["correct"].mean()
        best_acc = data[data["mode"] != "random_direction"]["correct"].mean()
        sections.append(f"Random-direction mean accuracy: {random_acc:.3f}; non-random mean accuracy: {best_acc:.3f}.\n")
    else:
        sections.append("_No random-direction rows found._\n")
    sections.append("\n## Best static steering vs CET\n")
    if "controller" in data:
        sections.append(table(data[data["controller"].isin(["static_steering", "cet"])], ["source", "controller"]))
    else:
        sections.append("_No CET input found._\n")
    sections.append("\n## Top neurons/features with examples\n")
    sections.append("See `outputs/neurons.json` and vector/RPM reports for ranked features.\n")
    sections.append("\n## Failure cases\n")
    if "correct" in data:
        failures = data[~data["correct"].astype(bool)].head(10)
        cols = [col for col in ["source", "problem_id", "mode", "controller", "question", "extracted_answer", "gold_answer"] if col in failures]
        sections.append(markdown(failures[cols]) if not failures.empty else "No failures in supplied inputs.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote report to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze effort-circuit outputs.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    analyze(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
