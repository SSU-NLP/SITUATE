"""Train the LAM on all adjacent utterance pairs of superseg_train (labels UNUSED).

Val = superseg_validation reconstruction MSE. Saves model + config to results/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dts_common import embed_split  # noqa: E402
from lam import LAM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"


def make_pairs(data: list[dict]) -> np.ndarray:
    """Stack all adjacent (z_t, z_{t+1}) pairs -> (P, 2, 768)."""
    zt, ztp1 = [], []
    for d in data:
        e = d["emb"]
        zt.append(e[:-1])
        ztp1.append(e[1:])
    return np.concatenate(zt), np.concatenate(ztp1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook-size", type=int, default=16)
    ap.add_argument("--code-dim", type=int, default=32)
    ap.add_argument("--commitment", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="K16")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"

    tr = embed_split("superseg_train")
    va = embed_split("superseg_validation")
    ztr, ztr1 = make_pairs(tr)
    zva, zva1 = make_pairs(va)
    print(f"train pairs {len(ztr)}, val pairs {len(zva)}")

    ztr_t = torch.tensor(ztr, device=dev)
    ztr1_t = torch.tensor(ztr1, device=dev)
    zva_t = torch.tensor(zva, device=dev)
    zva1_t = torch.tensor(zva1, device=dev)

    model = LAM(
        code_dim=args.code_dim,
        codebook_size=args.codebook_size,
        commitment_weight=args.commitment,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = len(ztr_t)
    best_val = 1e9
    hist = []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            out = model(ztr_t[idx], ztr1_t[idx])
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            tot += out["recon"].item() * len(idx)
        model.eval()
        with torch.no_grad():
            vo = model(zva_t, zva1_t)
            vrec = vo["recon"].item()
            used = len(torch.unique(vo["idx"]))
        hist.append({"ep": ep, "train_recon": tot / n, "val_recon": vrec, "codes_used": used})
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep{ep:3d} train {tot/n:.4f} val {vrec:.4f} codes_used {used}/{args.codebook_size}")
        if vrec < best_val:
            best_val = vrec

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / f"lam_{args.tag}.pt"
    torch.save({"state": model.state_dict(), "args": vars(args)}, ckpt)
    (OUT / f"train_hist_{args.tag}.json").write_text(json.dumps(hist, indent=2))
    # baseline: MSE of z_{t+1} vs z_t (predict-no-change) for reference
    with torch.no_grad():
        naive = ((zva1_t - zva_t) ** 2).sum(-1).mean().item()
    print(f"DONE tag={args.tag} best_val_recon={best_val:.4f} naive(z_t=z_t+1)_mse={naive:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
