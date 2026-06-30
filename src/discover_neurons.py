from __future__ import annotations

import argparse

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from .activation_store import load_activation_rows, stack_for
from .io_utils import read_table, write_json


def cohens_d(good: np.ndarray, bad: np.ndarray) -> np.ndarray:
    var = (good.var(axis=0) + bad.var(axis=0)) / 2
    return (good.mean(axis=0) - bad.mean(axis=0)) / np.sqrt(var + 1e-8)


def discover(args: argparse.Namespace) -> None:
    rows = load_activation_rows(args.activations)
    traces = read_table(args.traces)
    results = []
    for layer in sorted({row["layer"] for row in rows if row["site"] == args.site}):
        x, meta = stack_for(rows, site=args.site, position="prompt_last", layer=layer)
        joined = meta.merge(traces.reset_index(names="trace_index"), on="trace_index", how="left")
        joined["correct"] = joined["correct"].fillna(False)
        joined["reasoning_tokens"] = joined["reasoning_tokens"].fillna(joined["reasoning_tokens"].median()).fillna(0)
        correct = joined["correct"].astype(int).to_numpy()
        effort = joined["reasoning_tokens"].to_numpy()
        good = (correct == 1) & (effort >= np.median(effort))
        bad = ~good
        if good.sum() == 0 or bad.sum() == 0:
            continue
        values = x.numpy()
        d = cohens_d(values[good], values[bad])
        ratio = (np.abs(values[good]).mean(axis=0) + 1e-6) / (np.abs(values[bad]).mean(axis=0) + 1e-6)
        mi = mutual_info_classif(values, correct, discrete_features=False, random_state=0) if len(set(correct)) == 2 else np.zeros(values.shape[1])
        stability = np.zeros(values.shape[1])
        rng = np.random.default_rng(0)
        for _ in range(10):
            mask = rng.random(len(values)) > 0.5
            if mask.sum() > 2 and (~mask).sum() > 2:
                stability += np.sign(values[mask].mean(axis=0) - values[~mask].mean(axis=0)) == np.sign(d)
        stability /= 10
        score = np.abs(d) + np.log(ratio + 1e-6) + mi + stability
        for neuron in np.argsort(score)[-50:][::-1]:
            results.append(
                {
                    "layer": int(layer),
                    "neuron": int(neuron),
                    "cohens_d": float(d[neuron]),
                    "ratio_score": float(ratio[neuron]),
                    "mutual_information": float(mi[neuron]),
                    "stability": float(stability[neuron]),
                    "score": float(score[neuron]),
                }
            )
    write_json({"site": args.site, "neurons": sorted(results, key=lambda x: x["score"], reverse=True)}, args.out)
    print(f"wrote neuron scores to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover reasoning-relevant neurons.")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--site", default="mlp_act")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    discover(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
