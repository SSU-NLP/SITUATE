"""AUDIT: is the *training method* the reason the code leaks content? Or is it inherent?

Trains several LAM variants on the current corpus and reports, for each:
  - val recon (MSE) and FDM-ONLY recon (no code) => root cause: how well does z_t predict z_{t+1}?
  - code contribution = mse(code=mean) - mse(code=real)  (does the code help at all?)
  - content leakage NMI(code; topic-cluster) vs surprise delta-bin  (held-out corpus slice)
  - TIAGE matched-F1: A_bigram (unsup) vs Bp (surprise)

If leakage stays high & TIAGE A<=Bp across ALL training configs -> not a training bug, inherent.
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lam import LAM, mlp  # noqa: E402
from dts_common import embed_split  # noqa: E402
from gate1_diagnostics import mutual_info, entropy  # noqa: E402
from gate2_segmentation import code_seq  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-root/d13a28d4-7d56-4cb2-9443-281710e6c27a/scratchpad")
CORPUS = SCRATCH / "idm_corpus"
DEV = "cuda"


def load_pairs(cap=1_200_000):
    shards = sorted(glob.glob(str(CORPUS / "emb_shard_*.npy")))
    emb = np.concatenate([np.load(s) for s in shards], 0)
    lens = np.load(CORPUS / "dlg_lens.npy")
    offs = np.concatenate([[0], np.cumsum(lens)])
    pl = np.concatenate([np.arange(offs[d], offs[d + 1] - 1) for d in range(len(lens))]).astype(np.int64)
    rng = np.random.default_rng(0); rng.shuffle(pl)
    pl = pl[:cap]
    return emb, pl


class FDMOnly(nn.Module):
    def __init__(self, z=768):
        super().__init__()
        self.f = mlp([z, 512, z])

    def forward(self, zt, z1):
        zhat = self.f(zt)
        return ((zhat - z1) ** 2).sum(-1).mean()


def train_generic(model, emb_t, tr, va, epochs, lr, cosine, loss_kind, is_lam):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs) if cosine else None
    B = 4096

    def batch(idx):
        return emb_t[idx].to(DEV, torch.float32), emb_t[idx + 1].to(DEV, torch.float32)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        for i in range(0, len(tr), B):
            idx = tr[perm[i:i + B]]
            zt, z1 = batch(idx)
            if is_lam:
                out = model(zt, z1)
                if loss_kind == "cos":
                    zhat = out["zhat"]
                    rec = (1 - torch.cosine_similarity(zhat, z1, dim=-1)).mean()
                else:
                    rec = out["recon"]
                loss = rec + out["commit"]
            else:
                loss = model(zt, z1)
            opt.zero_grad(); loss.backward(); opt.step()
        if sched:
            sched.step()
    # eval recon
    model.eval()
    with torch.no_grad():
        zt, z1 = batch(va)
        if is_lam:
            o = model(zt, z1)
            return o["recon"].item()
        return model(zt, z1).item()


def audit_lam(model, emb_t, va):
    model.eval()
    with torch.no_grad():
        zt = emb_t[va].to(DEV, torch.float32); z1 = emb_t[va + 1].to(DEV, torch.float32)
        _, q, idx, _ = model.encode_transition(zt, z1)
        mse_real = ((model.fdm(torch.cat([zt, q], -1)) - z1) ** 2).sum(-1).mean().item()
        qmean = q.mean(0, keepdim=True).expand_as(q)
        mse_mean = ((model.fdm(torch.cat([zt, qmean], -1)) - z1) ** 2).sum(-1).mean().item()
        idx = idx.cpu().numpy()
    Z1 = z1.cpu().numpy(); Zt = zt.cpu().numpy()
    from sklearn.cluster import KMeans
    topic = KMeans(30, n_init=3, random_state=0).fit_predict(Z1[:20000])
    delta = 1 - (Zt[:20000] * Z1[:20000]).sum(1)
    qs = np.quantile(delta, np.linspace(0, 1, len(np.unique(idx)) + 1)[1:-1])
    dbin = np.digitize(delta, qs)
    _, nmi_ct = mutual_info(idx[:20000], topic)
    _, nmi_dt = mutual_info(dbin, topic)
    counts = np.bincount(idx, minlength=model.codebook_size)
    return {"mse_real": mse_real, "mse_mean": mse_mean, "contrib": mse_mean - mse_real,
            "nmi_code_topic": nmi_ct, "nmi_delta_topic": nmi_dt,
            "ppl": float(np.exp(entropy(counts)))}


def tiage_quick(model):
    from gate2_segmentation import code_seq as cs
    data = embed_split("tiage_test")
    # A_bigram needs code-bigram stats: build from tiage itself is circular; use unigram-free
    # simple unsupervised: boundary = code change rarity via global bigram from this set
    K = model.codebook_size
    P = np.ones((K, K)); uni = np.ones(K)
    seqs = []
    for d in data:
        c = cs(model, d["emb"], DEV); seqs.append(c)
        for t in range(len(c)):
            uni[c[t]] += 1
            if t > 0:
                P[c[t - 1], c[t]] += 1
    P /= P.sum(1, keepdims=True); uni /= uni.sum()
    fa, fb = [], []
    for d, c in zip(data, seqs):
        n = len(d["turns"]); emb = d["emb"]
        gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}; N = len(gold)
        delta = 1 - (emb[:-1] * emb[1:]).sum(1)
        sbig = np.array([-math.log(P[c[t - 1], c[t]]) if t > 0 else -math.log(uni[c[t]]) for t in range(len(c))])
        selA = set(np.argsort(-sbig)[:N].tolist()); selB = set(np.argsort(-delta)[:N].tolist())
        fa.append(len(selA & gold) / N if N else 0.0)
        fb.append(len(selB & gold) / N if N else 0.0)
    return float(np.mean(fa)), float(np.mean(fb))


def main():
    emb, pl = load_pairs()
    emb_t = torch.from_numpy(emb)
    va = pl[:60000]; tr = pl[60000:]
    print(f"audit corpus: {len(emb)} utts, train {len(tr)} val {len(va)} pairs", flush=True)

    # root cause: FDM-only
    torch.manual_seed(0)
    fo = FDMOnly().to(DEV)
    r0 = train_generic(fo, emb_t, tr, va, epochs=12, lr=1e-3, cosine=True, loss_kind="mse", is_lam=False)
    print(f"\n[FDM-only, no code]  val recon = {r0:.4f}  (how well z_t predicts z_{{t+1}})", flush=True)

    configs = [
        ("current(lr3e-4,mse,10ep)", dict(epochs=10, lr=3e-4, cosine=False, loss_kind="mse")),
        ("better(lr1e-4,cos,25ep)", dict(epochs=25, lr=1e-4, cosine=True, loss_kind="mse")),
        ("cosloss(lr1e-4,cos,25ep)", dict(epochs=25, lr=1e-4, cosine=True, loss_kind="cos")),
    ]
    print(f"\n{'config':28s}{'recon':>7s}{'contrib':>8s}{'ppl':>6s}{'NMIc;top':>9s}{'NMId;top':>9s}"
          f"{'TIAGE_Abig':>11s}{'TIAGE_Bp':>9s}", flush=True)
    for name, cfg in configs:
        torch.manual_seed(0); np.random.seed(0)
        m = LAM(codebook_size=16).to(DEV)
        rec = train_generic(m, emb_t, tr, va, is_lam=True, **cfg)
        a = audit_lam(m, emb_t, va)
        fa, fb = tiage_quick(m)
        print(f"{name:28s}{rec:>7.4f}{a['contrib']:>8.4f}{a['ppl']:>6.1f}"
              f"{a['nmi_code_topic']:>9.3f}{a['nmi_delta_topic']:>9.3f}{fa:>11.3f}{fb:>9.3f}", flush=True)


if __name__ == "__main__":
    main()
