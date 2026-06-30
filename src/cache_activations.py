from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .generation import load_model
from .hooks import ActivationCache, parse_layers, register_activation_hooks
from .io_utils import read_table


def select_position(tensor: torch.Tensor, position: str) -> torch.Tensor:
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if position == "prompt_last":
        return tensor[:, -1, :].mean(dim=0)
    if position == "pre_final":
        return tensor[:, max(0, tensor.shape[1] - 2), :].mean(dim=0)
    if position == "reasoning_all":
        return tensor.mean(dim=(0, 1))
    raise ValueError(f"Unsupported position {position!r}")


@torch.inference_mode()
def cache(args: argparse.Namespace) -> None:
    traces = read_table(args.traces)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_model(args.model, args.device, args.dtype)
    layers = parse_layers(args.layers, bundle.model)
    sites = [site.strip() for site in args.activation_sites.split(",") if site.strip()]
    positions = [pos.strip() for pos in args.positions.split(",") if pos.strip()]
    cache_store = ActivationCache()
    handles = register_activation_hooks(bundle.model, layers, sites, cache_store)
    manifest = {"layers": layers, "sites": sites, "positions": positions, "shards": []}
    shard_rows: list[dict] = []
    shard_idx = 0
    try:
        for row_idx, row in traces.iterrows():
            cache_store.clear()
            inputs = bundle.tokenizer(str(row["prompt"]) + str(row.get("completion", "")), return_tensors="pt").to(bundle.device)
            _ = bundle.model(**inputs)
            for name, tensors in cache_store.records.items():
                latest = tensors[-1]
                for position in positions:
                    shard_rows.append(
                        {
                            "trace_index": int(row_idx),
                            "problem_id": row["problem_id"],
                            "mode": row["mode"],
                            "name": name,
                            "layer": int(name.split(".")[0].replace("layer", "")),
                            "site": name.split(".")[1],
                            "position": position,
                            "activation": select_position(latest, position),
                        }
                    )
            if len(shard_rows) >= args.shard_size:
                shard_path = out / f"shard_{shard_idx:05d}.pt"
                torch.save(shard_rows, shard_path)
                manifest["shards"].append(shard_path.name)
                shard_rows = []
                shard_idx += 1
        if shard_rows:
            shard_path = out / f"shard_{shard_idx:05d}.pt"
            torch.save(shard_rows, shard_path)
            manifest["shards"].append(shard_path.name)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        for handle in handles:
            handle.remove()
    print(f"wrote activation manifest with {len(manifest['shards'])} shards to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache transformer activations.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--activation-sites", default="resid_post,mlp_act,mlp_out")
    parser.add_argument("--positions", default="prompt_last,reasoning_all,pre_final")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--shard-size", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> None:
    cache(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
