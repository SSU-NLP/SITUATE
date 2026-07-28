"""TEXT-ONLY smart signals for AMI (no timing, no speaker, no LLM). Beyond adjacent-turn distance:
  block_sym(w)  : sym block-cos (baseline aggregation)
  asym(WL,WR)   : left topic-so-far (large WL) vs incoming (small WR)
  proto_ewma(r) : distance of z_{t+1} to EWMA topic prototype of the past (rho r)
  ms_mean       : multi-scale (prev best text signal)
  texttiling    : Hearst-style depth over block cosine
Density-matched ±2 tolerant ceiling check. Can text-only reach 0.5?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dts_common import ENCODER_NAME  # noqa: E402
from ami_structure import load_raw_eval, z  # noqa: E402
from capstone_segment import ms_mean  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402

DEV = "cuda"


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def block_sym(emb, w):
    n = len(emb); out = np.zeros(n - 1)
    for t in range(n - 1):
        L = emb[max(0, t - w + 1):t + 1].mean(0); R = emb[t + 1:min(n, t + 1 + w)].mean(0)
        out[t] = 1 - _cos(L, R)
    return out


def asym(emb, WL, WR):
    n = len(emb); out = np.zeros(n - 1)
    for t in range(n - 1):
        L = emb[max(0, t - WL + 1):t + 1].mean(0); R = emb[t + 1:min(n, t + 1 + WR)].mean(0)
        out[t] = 1 - _cos(L, R)
    return out


def proto_ewma(emb, rho):
    n = len(emb); out = np.zeros(n - 1); mu = emb[0].copy()
    for t in range(n - 1):
        out[t] = 1 - _cos(emb[t + 1], mu)
        mu = rho * mu + (1 - rho) * emb[t]
    return out


def texttiling(emb, w=6):
    """Hearst depth: gap score g(t)=1-cos(L,R); depth = (peak_left-g)+(peak_right-g)."""
    n = len(emb); g = block_sym(emb, w)
    dep = np.zeros(len(g))
    for t in range(len(g)):
        lp = g[t]
        i = t
        while i > 0 and g[i - 1] >= g[i]:
            i -= 1; lp = max(lp, g[i])
        rp = g[t]
        j = t
        while j + 1 < len(g) and g[j + 1] >= g[j]:
            j += 1; rp = max(rp, g[j])
        dep[t] = (lp - g[t]) + (rp - g[t])
    return dep


def topn(s, N):
    return set(np.argsort(-s)[:N].tolist()) if N > 0 else set()


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")
    data = load_raw_eval(enc)

    def build(emb):
        return {
            "ms_mean": ms_mean(emb),
            "block_w5": block_sym(emb, 5),
            "asym8_2": asym(emb, 8, 2),
            "asym12_3": asym(emb, 12, 3),
            "proto_ewma.8": proto_ewma(emb, 0.8),
            "proto_ewma.9": proto_ewma(emb, 0.9),
            "texttiling": texttiling(emb, 6),
        }
    METH = ["ms_mean", "block_w5", "asym8_2", "asym12_3", "proto_ewma.8", "proto_ewma.9",
            "texttiling", "ORACLE", "random"]
    golds = []; preds = {m: [] for m in METH}; rng = np.random.default_rng(0)
    for d in data:
        n = len(d["turns"]); emb = d["emb"]
        gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}; N = len(gold)
        sig = build(emb)
        sel = {m: topn(sig[m], N) for m in sig}
        sel["ORACLE"] = set(gold); sel["random"] = set(rng.permutation(n - 1)[:N].tolist())
        bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt)
        for m in METH:
            p = [0] * n
            for t in sel[m]:
                if t + 1 < n:
                    p[t + 1] = 1
            p[-1] = 0; preds[m].append(p)
    print("AMI eval, TEXT-ONLY, density-matched, ±2 tolerant.")
    print(f"{'signal':14s}{'F1':>8s}{'Score':>8s}")
    for m in METH:
        r = score_meetings(golds, preds[m])
        print(f"{m:14s}{r['f1']:>8.3f}{r['score']:>8.3f}")


if __name__ == "__main__":
    main()
