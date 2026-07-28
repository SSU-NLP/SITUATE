"""Train LAM (IDM+VQ+FDM) UNSUPERVISED on the large diverse corpus (build_corpus.py output).

Embeddings kept on CPU (fp16), pair indices gathered per-batch to GPU. --frac enables the
data-scale ablation (does leakage/perf improve with more data?). Eval sets never seen here.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lam import LAM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path("/tmp/claude-0/-root/d13a28d4-7d56-4cb2-9443-281710e6c27a/scratchpad")
CORPUS = SCRATCH / "idm_corpus"
OUT = ROOT / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"


def load_corpus():
    shards = sorted(glob.glob(str(CORPUS / "emb_shard_*.npy")))
    emb = np.concatenate([np.load(s) for s in shards], axis=0)  # (U,768) fp16
    lens = np.load(CORPUS / "dlg_lens.npy")
    # pair_left = global indices i where (i, i+1) are within the same dialogue
    offs = np.concatenate([[0], np.cumsum(lens)])
    pair_left = np.concatenate([np.arange(offs[d], offs[d + 1] - 1) for d in range(len(lens))])
    return emb, pair_left.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook-size", type=int, default=16)
    ap.add_argument("--code-dim", type=int, default=32)
    ap.add_argument("--commitment", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="corpus_K16")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"

    emb, pair_left = load_corpus()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pair_left)
    if args.frac < 1.0:
        pair_left = pair_left[: int(len(pair_left) * args.frac)]
    n_val = min(200_000, len(pair_left) // 20)
    val_left = pair_left[:n_val]
    tr_left = pair_left[n_val:]
    print(f"corpus {len(emb)} utts, train pairs {len(tr_left)}, val pairs {len(val_left)}", flush=True)

    emb_t = torch.from_numpy(emb)  # CPU fp16
    def gather(idx):
        zt = emb_t[idx].to(dev, dtype=torch.float32, non_blocking=True)
        z1 = emb_t[idx + 1].to(dev, dtype=torch.float32, non_blocking=True)
        return zt, z1

    model = LAM(code_dim=args.code_dim, codebook_size=args.codebook_size,
                commitment_weight=args.commitment).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist = []
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr_left))
        tot = 0.0
        for i in range(0, len(tr_left), args.batch):
            idx = tr_left[perm[i : i + args.batch]]
            zt, z1 = gather(idx)
            out = model(zt, z1)
            opt.zero_grad(); out["loss"].backward(); opt.step()
            tot += out["recon"].item() * len(idx)
        model.eval()
        with torch.no_grad():
            vt, v1 = gather(val_left)
            vo = model(vt, v1)
            vrec = vo["recon"].item(); used = len(torch.unique(vo["idx"]))
        hist.append({"ep": ep, "train_recon": tot / len(tr_left), "val_recon": vrec, "codes_used": used})
        print(f"ep{ep:2d} train {tot/len(tr_left):.4f} val {vrec:.4f} codes {used}/{args.codebook_size}", flush=True)
        best = min(best, vrec)

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(args)}, OUT / f"lam_{args.tag}.pt")
    (OUT / f"train_hist_{args.tag}.json").write_text(json.dumps(hist, indent=2))
    print(f"DONE {args.tag} best_val {best:.4f} npairs {len(tr_left)} -> lam_{args.tag}.pt", flush=True)


if __name__ == "__main__":
    main()
