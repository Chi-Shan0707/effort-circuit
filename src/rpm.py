from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .activation_store import load_activation_rows, stack_for
from .io_utils import read_table


def fit(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_activation_rows(args.activations)
    traces = read_table(args.traces)
    layers = sorted({row["layer"] for row in rows if row["site"] == "resid_post"})
    if args.layers != "selected_by_probe":
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    report = []
    best = None
    for layer in layers:
        x, meta = stack_for(rows, site="resid_post", position="reasoning_all", layer=layer)
        dim = min(args.pca_dim, x.shape[0], x.shape[1])
        centered = x - x.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:dim].T
        z = centered @ basis
        if len(z) < 2:
            continue
        z0, z1 = z[:-1], z[1:]
        a = torch.linalg.lstsq(z0, z1).solution.T
        eigvals, eigvecs = torch.linalg.eig(a)
        joined = meta.merge(traces.reset_index(names="trace_index"), on="trace_index", how="left")
        joined["correct"] = joined["correct"].fillna(False)
        joined["reasoning_tokens"] = joined["reasoning_tokens"].fillna(joined["reasoning_tokens"].median()).fillna(0)
        lengths = joined["reasoning_tokens"].to_numpy()[: len(z)]
        correctness = joined["correct"].astype(int).to_numpy()[: len(z)]
        for idx, eig in enumerate(eigvals):
            mode_z = eigvecs[:, idx].real.float()
            vector = (basis @ mode_z).float()
            score_series = (centered @ vector).numpy()
            length_corr = float(np.corrcoef(score_series, lengths)[0, 1]) if len(score_series) > 1 else float("nan")
            correct_corr = float(np.corrcoef(score_series, correctness)[0, 1]) if len(set(correctness)) > 1 else float("nan")
            item = {
                "layer": int(layer),
                "mode": int(idx),
                "eigenvalue_abs": float(torch.abs(eig).item()),
                "remaining_length_corr": length_corr,
                "correctness_corr": correct_corr,
                "dev_causal_score": float(np.nan_to_num(length_corr) + np.nan_to_num(correct_corr)),
            }
            path = out / f"layer{layer}_mode{idx}.pt"
            torch.save({"vector": vector, **item}, path)
            report.append(item)
            if best is None or item["dev_causal_score"] > best["dev_causal_score"]:
                best = item | {"path": path.name}
    if best:
        torch.save(torch.load(out / best["path"], map_location="cpu", weights_only=False), out / "best.pt")
    (out / "report.json").write_text(json.dumps({"modes": report, "best": best}, indent=2), encoding="utf-8")
    print(f"wrote RPM report to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit Reasoning Persistence Modes.")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--layers", default="selected_by_probe")
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    fit(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
