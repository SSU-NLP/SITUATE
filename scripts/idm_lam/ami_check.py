"""Sanity: AMI density-matched, ±2 TOLERANT F1 (official ami_scoring), incl. ORACLE upper bound.
If ORACLE ~1.0 -> scorer+alignment correct; then compare cosine/FDM_best/C_lex vs random honestly.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dts_common import ENCODER_NAME  # noqa: E402
from lam import LAM  # noqa: E402
from gate2_heldout import load_ami  # noqa: E402
from sticky_sem import error_matrix  # noqa: E402
from hi_ontop.ami_scoring import score_meetings, tol_f1  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"
WORD = re.compile(r"[a-zA-Z']+")


def lex(turns):
    toks = [Counter(w.lower() for w in WORD.findall(s)) for s in turns]
    n = len(turns); w = 2; out = []
    for t in range(n - 1):
        L = Counter(); R = Counter()
        for i in range(max(0, t - w + 1), t + 1):
            L.update(toks[i])
        for i in range(t + 1, min(n, t + 1 + w)):
            R.update(toks[i])
        ks = set(L) | set(R)
        if not ks:
            out.append(0.0); continue
        a = np.array([L[k] for k in ks], float); b = np.array([R[k] for k in ks], float)
        out.append(1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    return np.array(out)


def topn(s, N):
    return set(np.argsort(-s)[:N].tolist()) if N > 0 else set()


def main():
    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]
    model = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"],
               commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")
    data = load_ami("eval", enc)

    methods = ["ORACLE", "cosine", "FDM_best", "C_lex", "random"]
    golds = []; preds = {m: [] for m in methods}
    # also direct tol_f1 on boundary-index lists (independent of per-turn scorer)
    directf1 = {m: [] for m in methods}
    rng = np.random.default_rng(0)
    ntot = 0
    for d in data:
        n = len(d["turns"]); emb = d["emb"]
        gold_tr = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}
        N = len(gold_tr); ntot += N
        E, delta = error_matrix(model, emb)
        sel = {
            "ORACLE": set(gold_tr),
            "cosine": topn(delta, N),
            "FDM_best": topn(E.min(1), N),
            "C_lex": topn(lex(d["turns"]), N),
            "random": set(rng.permutation(n - 1)[:N].tolist()),
        }
        bt = list(d["bnd_top"]); bt[-1] = 0
        golds.append(bt)
        gold_turns = [i for i in range(n) if bt[i] == 1]
        for m in methods:
            p = [0] * n
            for t in sel[m]:
                if t + 1 < n:
                    p[t + 1] = 1
            p[-1] = 0
            preds[m].append(p)
            pred_turns = [t + 1 for t in sel[m] if 0 < t + 1 < n - 1]
            directf1[m].append(tol_f1(gold_turns, pred_turns, tol=2))

    print(f"AMI eval: {len(data)} meetings, {ntot} gold boundaries. Density-matched, ±2 TOLERANT F1.")
    print(f"{'method':10s}{'F1(±2 official)':>16s}{'F1(±2 direct)':>15s}{'Score':>8s}{'Pk':>7s}{'WD':>7s}")
    for m in methods:
        r = score_meetings(golds, preds[m])
        print(f"{m:10s}{r['f1']:>16.3f}{np.mean(directf1[m]):>15.3f}{r['score']:>8.3f}{r['pk']:>7.3f}{r['wd']:>7.3f}")


if __name__ == "__main__":
    main()
