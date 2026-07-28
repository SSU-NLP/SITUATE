"""Confirm the noise-bottleneck (β) promising result: full corpus (frac 1.0) β=10 × 2 seeds
+ β-robustness (5/20 at frac 0.3). Corpus loaded ONCE. Reports Gate1 leakage + Gate2 (A vs Bp vs random).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dts_common import embed_split  # noqa: E402
from train_corpus import load_corpus  # noqa: E402
from lam_bottleneck import LAMBottleneck  # noqa: E402
from run_bottleneck import gate1, gate2  # noqa: E402

DEV = "cuda"


def train_seed(emb, pair_left, frac, epochs, beta, latent_dim, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed); pl = pair_left.copy(); rng.shuffle(pl)
    pl = pl[: int(len(pl) * frac)]
    nval = min(150_000, len(pl) // 20); val, tr = pl[:nval], pl[nval:]
    embt = torch.from_numpy(emb)
    m = LAMBottleneck(latent_dim=latent_dim, bottleneck="noise", beta=beta).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4); B = 8192
    def batch(idx):
        return embt[idx].to(DEV, torch.float32), embt[idx + 1].to(DEV, torch.float32)
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(len(tr))
        for i in range(0, len(tr), B):
            idx = tr[perm[i:i + B]]; zt, z1 = batch(idx)
            out = m(zt, z1); opt.zero_grad(); out["loss"].backward(); opt.step()
    m.eval()
    with torch.no_grad():
        zt, z1 = batch(val); recon = m(zt, z1)["recon"].item()
    return m, recon


def main():
    emb, pair_left = load_corpus()
    va = embed_split("superseg_validation")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cuda")
    dsets = [("tiage_test", None), ("superseg_test", None), ("dialseg711_test", None), ("ami_eval", "ami")]

    configs = [
        ("beta10 frac1.0 seed0", 10.0, 1.0, 0),
        ("beta10 frac1.0 seed1", 10.0, 1.0, 1),
        ("beta5  frac0.3 seed0", 5.0, 0.3, 0),
        ("beta20 frac0.3 seed0", 20.0, 0.3, 0),
    ]
    for name, beta, frac, seed in configs:
        m, recon = train_seed(emb, pair_left, frac, 8, beta, 64, seed)
        g1 = gate1(m, va)
        g2 = gate2(m, dsets, enc)
        print(f"\n######## {name} (recon={recon:.4f}) ########", flush=True)
        print(f"  [Gate1] R2(ā→내용)={g1['r2_content']:.3f}  AMI(clust;topic)={g1['ami_cluster_topic']:.3f}")
        for ds in g2:
            r = g2[ds]
            print(f"  [Gate2] {ds:16s} A={r['A']:.3f}(F1 {r['A_F1']:.3f}) Bp={r['Bp']:.3f} rand={r['rand']:.3f}  "
                  f"p(A-Bp)={r['p_A_Bp']:.1e} p(A-rand)={r['p_A_rand']:.1e}", flush=True)
    print("\nCONFIRM DONE", flush=True)


if __name__ == "__main__":
    main()
