from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch


def load_activation_rows(path: str | Path) -> list[dict]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    for shard in manifest["shards"]:
        rows.extend(torch.load(root / shard, map_location="cpu", weights_only=False))
    return rows


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    serializable = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "activation"}
        item["activation_norm"] = float(row["activation"].norm().item())
        serializable.append(item)
    return pd.DataFrame(serializable)


def stack_for(rows: list[dict], site: str | None = None, position: str | None = None, layer: int | None = None):
    selected = [
        row
        for row in rows
        if (site is None or row["site"] == site)
        and (position is None or row["position"] == position)
        and (layer is None or row["layer"] == layer)
    ]
    if not selected:
        raise ValueError("No activations matched selection")
    x = torch.stack([row["activation"].float() for row in selected])
    meta = pd.DataFrame([{key: value for key, value in row.items() if key != "activation"} for row in selected])
    return x, meta
