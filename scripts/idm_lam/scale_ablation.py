"""Data-scale ablation: does leakage(Gate1) / TIAGE perf(Gate2) improve with more data?

Answers doc §9's caveat "small corpus -> can't tell method-failure from data-starvation".
Train K=16 on {10,25,50,100}% of superseg_train pairs (seed-fixed subsample), then measure:
  Gate1: NMI(code;topic)  vs  cond-NMI(code;boundary|topic)  vs surprise delta-bin baseline.
  Gate2: TIAGE matched-F1 for A_typed / A_bigram / Bp(surprise) / random.
If leakage flat/rising and TIAGE-A flat/below-Bp across scale -> NO-GO is structural, not data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dts_common import embed_split  # noqa: E402
from lam import LAM  # noqa: E402
from gate1_diagnostics import mutual_info  # noqa: E402
from gate2_segmentation import code_seq, calibrate  # noqa: E402
from gate2_significance import topn_set, dlg_f1  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"
K = 16


def make_pairs(data):
    zt, z1 = [], []
    for d in data:
        e = d["emb"]; zt.append(e[:-1]); z1.append(e[1:])
    return np.concatenate(zt), np.concatenate(z1)


def train_frac(ztr, z1, frac, epochs=60):
    torch.manual_seed(0); np.random.seed(0)
    n = len(ztr)
    idx = np.random.default_rng(0).permutation(n)[: int(n * frac)]
    zt = torch.tensor(ztr[idx], device=DEV); z1t = torch.tensor(z1[idx], device=DEV)
    m = LAM(code_dim=32, codebook_size=K, commitment_weight=0.25).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    nn = len(zt)
    for ep in range(epochs):
        perm = torch.randperm(nn, device=DEV)
        for i in range(0, nn, 512):
            b = perm[i : i + 512]
            out = m(zt[b], z1t[b]); opt.zero_grad(); out["loss"].backward(); opt.step()
    m.eval()
    return m


def gate1_quick(m, va):
    Zt, Z1, bl, topicsrc = [], [], [], []
    for d in va:
        e, bnd = d["emb"], d["boundaries"]
        for t in range(len(d["turns"]) - 1):
            Zt.append(e[t]); Z1.append(e[t + 1]); bl.append(1 if (t + 1) in bnd else 0)
    Zt = np.stack(Zt); Z1 = np.stack(Z1); bl = np.array(bl)
    with torch.no_grad():
        _, _, idx, _ = m.encode_transition(torch.tensor(Zt, device=DEV), torch.tensor(Z1, device=DEV))
    idx = idx.cpu().numpy()
    from sklearn.cluster import KMeans
    topic = KMeans(30, n_init=4, random_state=0).fit_predict(Z1)
    delta = 1 - (Zt * Z1).sum(1)
    qs = np.quantile(delta, np.linspace(0, 1, K + 1)[1:-1]); dbin = np.digitize(delta, qs)
    _, nmi_topic = mutual_info(idx, topic)

    def cond(code):
        v, tot = [], 0
        for c in np.unique(topic):
            msk = topic == c
            if msk.sum() < 30 or len(np.unique(bl[msk])) < 2 or len(np.unique(code[msk])) < 2:
                continue
            _, nm = mutual_info(code[msk].astype(int), bl[msk].astype(int))
            v.append(nm * msk.sum()); tot += msk.sum()
        return sum(v) / tot if tot else 0.0
    return {"nmi_code_topic": float(nmi_topic), "cond_code_bnd": float(cond(idx)),
            "cond_delta_bnd": float(cond(dbin))}


def gate2_tiage(m, train, tiage):
    sh, P, uni = calibrate(m, train, DEV, K)
    f = {"A_typed": [], "A_bigram": [], "Bp": [], "rand": []}
    for d in tiage:
        e = d["emb"]; n = len(d["turns"])
        gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}; N = len(gold)
        c = code_seq(m, e, DEV); delta = 1 - (e[:-1] * e[1:]).sum(1)
        sb = np.array([-np.log(P[c[t - 1], c[t]]) if t > 0 else -np.log(uni[c[t]]) for t in range(len(c))])
        f["A_typed"].append(dlg_f1(topn_set(sh[c], N), gold))
        f["A_bigram"].append(dlg_f1(topn_set(sb, N), gold))
        f["Bp"].append(dlg_f1(topn_set(delta, N), gold))
        rng = np.random.default_rng(7)
        f["rand"].append(dlg_f1(set(rng.permutation(len(delta))[:N].tolist()), gold))
    return {k: float(np.mean(v)) for k, v in f.items()}


if __name__ == "__main__":
    train = embed_split("superseg_train")
    va = embed_split("superseg_validation")
    tiage = embed_split("tiage_test")
    ztr, z1 = make_pairs(train)
    rows = []
    for frac in [0.1, 0.25, 0.5, 1.0]:
        m = train_frac(ztr, z1, frac)
        g1 = gate1_quick(m, va)
        g2 = gate2_tiage(m, train, tiage)
        npairs = int(len(ztr) * frac)
        rows.append({"frac": frac, "npairs": npairs, **g1, **g2})
        print(f"frac={frac:.2f} ({npairs:6d} pairs) | "
              f"NMI(code;topic)={g1['nmi_code_topic']:.3f} cond(code;bnd)={g1['cond_code_bnd']:.3f} "
              f"cond(delta;bnd)={g1['cond_delta_bnd']:.3f} || "
              f"TIAGE F1: A_typed={g2['A_typed']:.3f} A_bigram={g2['A_bigram']:.3f} "
              f"Bp={g2['Bp']:.3f} rand={g2['rand']:.3f}")
    (OUT / "scale_ablation.json").write_text(json.dumps(rows, indent=2))
    print("saved -> scale_ablation.json")
