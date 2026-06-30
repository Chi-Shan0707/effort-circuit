from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    out = ensure_parent(path)
    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def write_json(data: object, path: str | Path) -> None:
    out = ensure_parent(path)
    out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
