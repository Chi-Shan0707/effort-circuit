from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .activation_store import load_activation_rows, stack_for


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        features = torch.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return reconstruction, features


def train(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_activation_rows(args.activations)
    x, _ = stack_for(rows, site=args.site, position=args.position, layer=args.layer)
    x = x.float()
    model = SparseAutoencoder(x.shape[1], args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(x), batch_size=args.batch_size, shuffle=True)
    history = []
    for epoch in range(args.epochs):
        total = 0.0
        for (batch,) in loader:
            reconstruction, features = model(batch)
            mse = torch.mean((reconstruction - batch) ** 2)
            sparsity = torch.mean(torch.abs(features))
            loss = mse + args.l1 * sparsity
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        history.append(total / max(1, len(x)))
    with torch.no_grad():
        reconstruction, features = model(x)
        mse = torch.mean((reconstruction - x) ** 2)
        variance = torch.var(x)
        explained_variance = 1.0 - float(mse.item() / (variance.item() + 1e-8))
        l0 = float((features > 0).float().sum(dim=1).mean().item())
        dead_feature_rate = float(((features > 0).sum(dim=0) == 0).float().mean().item())
    metrics = {
        "input_dim": x.shape[1],
        "hidden_dim": args.hidden_dim,
        "site": args.site,
        "position": args.position,
        "layer": args.layer,
        "loss": history,
        "final_loss": history[-1] if history else None,
        "mse": float(mse.item()),
        "explained_variance": explained_variance,
        "l0": l0,
        "dead_feature_rate": dead_feature_rate,
        "num_examples": int(x.shape[0]),
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": x.shape[1],
            "hidden_dim": args.hidden_dim,
            "site": args.site,
            "position": args.position,
            "layer": args.layer,
            "loss": history,
            "metrics": metrics,
        },
        out / "sae.pt",
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote SAE to {out / 'sae.pt'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a sparse autoencoder over cached activations.")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--site", default="mlp_act")
    parser.add_argument("--position", default="prompt_last")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
