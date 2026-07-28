"""No-parameter, noise-ADAPTIVE multi-scale boundary signal.

Compute block-cosine at scales w in {1,2,3,5,8}; z-score each (per dialogue). Combine WITHOUT
any tuned parameter, several ways, and pick the one robust across CLEAN (tiage) and NOISY (ami):
  ms_mean : mean of z-scored scales
  ms_max  : max over scales
  ms_kurt : SELECT the single scale with highest kurtosis (self-adaptive: clean->small w, noisy->large w)
  ms_snr  : sum of scales weighted by their excess-kurtosis (>0)  [soft version of ms_kurt]
Reference: per-dataset best single window (cos_w2 clean / cos_w5 noisy) = the oracle a fixed
window can't reach with ONE setting. A good adaptive combo matches both.
Density-matched, official scorers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.stats import kurtosis

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dts_common import embed_split, ENCODER_NAME  # noqa: E402
from gate2_heldout import load_ami  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402

DEV = "cuda"
WSET = [1, 2, 3, 5, 8]


def block_cos(emb, w):
    n = len(emb); out = np.zeros(n - 1)
    for t in range(n - 1):
        L = emb[max(0, t - w + 1):t + 1].mean(0); R = emb[t + 1:min(n, t + 1 + w)].mean(0)
        L = L / (np.linalg.norm(L) + 1e-9); R = R / (np.linalg.norm(R) + 1e-9)
        out[t] = 1 - float(L @ R)
    return out


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def signals(emb):
    S = {w: block_cos(emb, w) for w in WSET}
    Z = {w: z(S[w]) for w in WSET}
    M = np.stack([Z[w] for w in WSET], 0)          # (nscale, T)
    kurts = np.array([max(0.0, float(kurtosis(S[w]))) for w in WSET])  # peakiness per scale
    out = {f"cos_w{w}": S[w] for w in WSET}
    out["ms_mean"] = M.mean(0)
    out["ms_max"] = M.max(0)
    out["ms_kurt"] = Z[WSET[int(np.argmax(kurts))]]                    # self-selected scale
    wsum = kurts.sum() + 1e-9
    out["ms_snr"] = (M * (kurts / wsum)[:, None]).sum(0)
    return out


def topn(s, N):
    return set(np.argsort(-s)[:N].tolist()) if N > 0 else set()


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")
    METH = ["cos_w2", "cos_w5", "ms_mean", "ms_max", "ms_kurt", "ms_snr"]
    for name, kind in [("tiage_test", None), ("ami_eval", "ami"),
                       ("superseg_test", None), ("dialseg711_test", None)]:
        data = load_ami("eval", enc) if kind == "ami" else embed_split(name)
        golds = []; preds = {m: [] for m in METH}
        for d in data:
            n = len(d["turns"]); emb = d["emb"]
            gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}; N = len(gold)
            sig = signals(emb)
            if kind == "ami":
                bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt)
            else:
                g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0; golds.append(g)
            for m in METH:
                sel = topn(sig[m], N)
                if kind == "ami":
                    p = [0] * n
                    for t in sel:
                        if t + 1 < n:
                            p[t + 1] = 1
                    p[-1] = 0
                else:
                    p = [1 if t in sel else 0 for t in range(n)]; p[-1] = 0
                preds[m].append(p)
        scorer = score_meetings if kind == "ami" else score_dialogues
        print(f"\n=== {name} ({'NOISY' if kind=='ami' else 'clean'}) ===")
        print(f"{'method':10s}{'F1':>8s}{'Score':>8s}")
        for m in METH:
            r = scorer(golds, preds[m])
            print(f"{m:10s}{r['f1']:>8.3f}{r['score']:>8.3f}")


if __name__ == "__main__":
    main()
