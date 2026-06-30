from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, float]:
    if n <= 0:
        return {"center": float("nan"), "low": float("nan"), "high": float("nan")}
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * n)) / n) / denom
    return {"center": center, "low": max(0.0, center - half), "high": min(1.0, center + half)}


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap_ci(
    pairs: list[tuple[bool, bool]],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    if not pairs:
        return {"mean_diff": float("nan"), "low": float("nan"), "high": float("nan")}
    observed = sum(int(candidate) - int(baseline) for baseline, candidate in pairs) / len(pairs)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        diffs.append(sum(int(candidate) - int(baseline) for baseline, candidate in sample) / len(sample))
    return {"mean_diff": observed, "low": quantile(diffs, 0.025), "high": quantile(diffs, 0.975)}


def exact_mcnemar_p(discordant_positive: int, discordant_negative: int) -> float:
    total = discordant_positive + discordant_negative
    if total == 0:
        return 1.0
    observed = min(discordant_positive, discordant_negative)
    lower_tail = sum(math.comb(total, k) for k in range(observed + 1)) / (2**total)
    return min(1.0, 2.0 * lower_tail)


def group_rows(rows: list[dict[str, Any]], metric: str) -> dict[str, dict[str, bool]]:
    grouped: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        grouped[row["condition"]][row["problem_id"]] = bool(row[metric])
    return grouped


def analyze_metric(rows: list[dict[str, Any]], metric: str, baseline: str, bootstrap: int, seed: int) -> dict[str, Any]:
    grouped = group_rows(rows, metric)
    if baseline not in grouped:
        raise ValueError(f"baseline condition {baseline!r} not found")
    baseline_map = grouped[baseline]
    conditions = sorted(grouped.keys(), key=lambda value: (value != baseline, value))
    result: dict[str, Any] = {"metric": metric, "baseline": baseline, "conditions": []}
    for condition in conditions:
        current = grouped[condition]
        common_ids = sorted(set(baseline_map) & set(current))
        successes = sum(current[problem_id] for problem_id in common_ids)
        n = len(common_ids)
        pairs = [(baseline_map[problem_id], current[problem_id]) for problem_id in common_ids]
        b_wrong_c_right = sum((not base) and cand for base, cand in pairs)
        b_right_c_wrong = sum(base and not cand for base, cand in pairs)
        result["conditions"].append(
            {
                "condition": condition,
                "n": n,
                "successes": successes,
                "accuracy": successes / n if n else float("nan"),
                "wilson95": wilson_ci(successes, n),
                "paired_diff_vs_baseline": paired_bootstrap_ci(pairs, bootstrap, seed),
                "mcnemar": {
                    "baseline_wrong_candidate_right": b_wrong_c_right,
                    "baseline_right_candidate_wrong": b_right_c_wrong,
                    "discordant_total": b_wrong_c_right + b_right_c_wrong,
                    "exact_two_sided_p": exact_mcnemar_p(b_wrong_c_right, b_right_c_wrong),
                },
            }
        )
    return result


def analyze_auxiliary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    summaries = []
    for condition in sorted(by_condition):
        subset = by_condition[condition]
        n = len(subset)
        summaries.append(
            {
                "condition": condition,
                "n": n,
                "strict_stop_rate": sum(bool(row.get("stopped")) for row in subset) / n,
                "strict_malformed_rate": sum(row.get("first_final_answer") in (None, "") for row in subset) / n,
                "mean_generated_tokens": sum(float(row.get("generated_tokens", 0.0)) for row in subset) / n,
                "mean_repetition_rate": sum(float(row.get("repetition_rate", 0.0)) for row in subset) / n,
            }
        )
    return {"conditions": summaries}


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Paired Statistical Analysis",
        "",
        f"Input: `{result['input']}`",
        f"Bootstrap iterations: `{result['bootstrap_iterations']}`",
        "",
    ]
    for metric_result in result["metrics"]:
        lines.extend(
            [
                f"## Metric: `{metric_result['metric']}`",
                "",
                "| condition | n | acc | Wilson 95% CI | diff vs baseline | bootstrap 95% CI | b-wrong/c-right | b-right/c-wrong | McNemar p |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in metric_result["conditions"]:
            ci = row["wilson95"]
            diff = row["paired_diff_vs_baseline"]
            mcnemar = row["mcnemar"]
            lines.append(
                f"| {row['condition']} | {row['n']} | {row['accuracy']:.3f} | "
                f"[{ci['low']:.3f}, {ci['high']:.3f}] | {diff['mean_diff']:+.3f} | "
                f"[{diff['low']:+.3f}, {diff['high']:+.3f}] | "
                f"{mcnemar['baseline_wrong_candidate_right']} | {mcnemar['baseline_right_candidate_wrong']} | "
                f"{mcnemar['exact_two_sided_p']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Auxiliary Protocol Metrics",
            "",
            "| condition | n | strict stop rate | strict malformed rate | mean tokens | repetition |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["auxiliary"]["conditions"]:
        lines.append(
            f"| {row['condition']} | {row['n']} | {row['strict_stop_rate']:.3f} | "
            f"{row['strict_malformed_rate']:.3f} | {row['mean_generated_tokens']:.2f} | "
            f"{row['mean_repetition_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "For small n, Wilson intervals and exact paired tests are intentionally wide. Treat these tables as a falsification and measurement-quality audit, not as evidence of a settled effect size.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = data["rows"]
    result = {
        "input": args.input,
        "baseline": args.baseline,
        "bootstrap_iterations": args.bootstrap,
        "metrics": [
            analyze_metric(rows, metric, args.baseline, args.bootstrap, args.seed + index)
            for index, metric in enumerate(args.metrics.split(","))
        ],
        "auxiliary": analyze_auxiliary(rows),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Wilson, paired bootstrap, and exact McNemar statistics from experiment JSON rows.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--metrics", default="first_final_correct,last_final_correct")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(render_report(result), encoding="utf-8")
    print(f"wrote {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
