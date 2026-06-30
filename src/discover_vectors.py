from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .activation_store import load_activation_rows, stack_for
from .io_utils import read_table


def safe_auc(y_true, score) -> float:
    try:
        if len(set(map(int, y_true))) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, score))
    except Exception:
        return float("nan")


def label_values(meta, traces, label: str) -> np.ndarray:
    joined = meta.merge(traces.reset_index(names="trace_index"), on="trace_index", how="left", suffixes=("", "_trace"))
    joined["reasoning_tokens"] = joined["reasoning_tokens"].fillna(joined["reasoning_tokens"].median()).fillna(0)
    joined["correct"] = joined["correct"].fillna(False)
    if label == "high_effort":
        return (joined["reasoning_tokens"] >= joined["reasoning_tokens"].median()).astype(int).to_numpy()
    if label == "correct":
        return joined["correct"].astype(int).to_numpy()
    if label == "positive_flip":
        per_problem = joined.groupby("problem_id")["correct"].transform("max")
        return ((per_problem.astype(int) == 1) & (joined["correct"].astype(int) == 1)).astype(int).to_numpy()
    raise ValueError(f"Unsupported label {label!r}")


def discover(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_activation_rows(args.activations)
    traces = read_table(args.traces)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    layers = sorted({row["layer"] for row in rows if row["site"] == "resid_post" and row["position"] == "prompt_last"})
    report = []
    best = None
    for layer in layers:
        x, meta = stack_for(rows, site="resid_post", position="prompt_last", layer=layer)
        x_np = x.numpy()
        for label in labels:
            y = label_values(meta, traces, label)
            if "paired_diff" in methods:
                high = x_np[y == 1]
                low = x_np[y == 0]
                if len(high) and len(low):
                    vector = torch.from_numpy(high.mean(axis=0) - low.mean(axis=0)).float()
                    score = x_np @ vector.numpy()
                    item = {"layer": layer, "method": "paired_diff", "label": label, "auroc": safe_auc(y, score)}
                    torch.save({"vector": vector, **item}, out / f"layer{layer}_{label}_paired_diff.pt")
                    report.append(item)
            if "svd" in methods and len(x_np) > 1:
                centered = x - x.mean(dim=0, keepdim=True)
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                vector = vh[0].float()
                score = x_np @ vector.numpy()
                item = {"layer": layer, "method": "svd", "label": label, "auroc": safe_auc(y, score)}
                torch.save({"vector": vector, **item}, out / f"layer{layer}_{label}_svd.pt")
                report.append(item)
            if "probe" in methods and len(set(map(int, y))) == 2 and len(y) >= 6:
                train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.33, random_state=0, stratify=y)
                clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_np[train_idx], y[train_idx])
                prob = clf.predict_proba(x_np)[:, 1]
                vector = torch.from_numpy(clf.coef_[0]).float()
                item = {
                    "layer": layer,
                    "method": "probe",
                    "label": label,
                    "auroc": safe_auc(y, prob),
                    "train_auroc": safe_auc(y[train_idx], prob[train_idx]),
                    "test_auroc": safe_auc(y[test_idx], prob[test_idx]),
                }
                torch.save({"vector": vector, "intercept": float(clf.intercept_[0]), **item}, out / f"layer{layer}_{label}_probe.pt")
                report.append(item)
                if best is None or np.nan_to_num(item["test_auroc"], nan=-1) > np.nan_to_num(best["test_auroc"], nan=-1):
                    best = item | {"path": f"layer{layer}_{label}_probe.pt"}
    for item in report:
        x, meta = stack_for(rows, site="resid_post", position="prompt_last", layer=item["layer"])
        joined = meta.merge(traces.reset_index(names="trace_index"), on="trace_index", how="left")
        joined["reasoning_tokens"] = joined["reasoning_tokens"].fillna(joined["reasoning_tokens"].median()).fillna(0)
        vec = torch.load(out / f"layer{item['layer']}_{item['label']}_{item['method']}.pt", map_location="cpu", weights_only=False)["vector"]
        scores = (x @ vec).numpy()
        item["correlation_reasoning_tokens"] = float(np.corrcoef(scores, joined["reasoning_tokens"].to_numpy())[0, 1]) if len(scores) > 1 else float("nan")
    if best:
        torch.save(torch.load(out / best["path"], map_location="cpu", weights_only=False), out / "best_probe.pt")
    (out / "report.json").write_text(json.dumps({"vectors": report, "best_probe": best}, indent=2), encoding="utf-8")
    print(f"wrote {len(report)} vector artifacts to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover steering vectors.")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--methods", default="paired_diff,probe,svd")
    parser.add_argument("--labels", default="high_effort,correct,positive_flip")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    discover(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
