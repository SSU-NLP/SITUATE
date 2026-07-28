"""Performance push #1: late-fusion of cosine-delta (raw surprise) + FDM_best (learned-dynamics
surprise). Per-dialogue z-score each, combine score = w*z(delta) + (1-w)*z(FDM_best), sweep w.
Density-matched, official scorers, held-out. Goal: beat cosine-delta on human TIAGE.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dts_common import embed_split, ENCODER_NAME  # noqa: E402
from lam import LAM  # noqa: E402
from gate2_heldout import load_ami  # noqa: E402
from sticky_sem import error_matrix  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"
WS = [1.0, 0.75, 0.5, 0.25, 0.0]  # 1.0=pure cosine, 0.0=pure FDM_best


def z(x):
    s = x.std()
    return (x - x.mean()) / (s + 1e-9)


def topn(sig, N):
    return set(np.argsort(-sig)[:N].tolist()) if N > 0 else set()


def eval_set(model, name, kind, enc):
    data = load_ami("eval", enc) if kind == "ami" else embed_split(name)
    methods = [f"w{w}" for w in WS]
    golds, preds = [], {m: [] for m in methods}
    f1 = {m: [] for m in methods}
    for d in data:
        n = len(d["turns"]); emb = d["emb"]
        gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}
        E, delta = error_matrix(model, emb)
        fdm_best = E.min(1)
        zd, zf = z(delta), z(fdm_best)
        if kind == "ami":
            bt = list(d["bnd_top"]); bt[-1] = 0; N = sum(bt); golds.append(bt)
        else:
            N = len(gold); g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0
            golds.append(g)
        for w in WS:
            sel = topn(w * zd + (1 - w) * zf, N)
            if kind == "ami":
                p = [0] * n
                for t in sel:
                    if t + 1 < n:
                        p[t + 1] = 1
                p[-1] = 0
            else:
                p = [1 if t in sel else 0 for t in range(n)]; p[-1] = 0
            preds[f"w{w}"].append(p)
            f1[f"w{w}"].append(len(sel & gold) / len(gold) if gold else 0.0)
    scorer = score_meetings if kind == "ami" else score_dialogues
    out = {m: scorer(golds, preds[m]) for m in methods}
    base = np.array(f1["w1.0"])  # pure cosine
    for m in methods:
        arr = np.array(f1[m]); dd = arr - base
        out[m]["dF1"] = float(dd.mean())
        out[m]["p"] = 1.0 if np.all(dd == 0) else float(wilcoxon(arr, base).pvalue)
    return out, methods


def main():
    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]
    model = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"],
               commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")

    for name, kind in [("tiage_test", None), ("ami_eval", "ami"),
                       ("superseg_test", None), ("dialseg711_test", None)]:
        out, methods = eval_set(model, name, kind, enc)
        print(f"\n=== {name} ===  (w1.0=pure cosine-δ, w0.0=pure FDM_best)")
        print(f"{'weight':10s}{'F1':>8s}{'Score':>8s}   vs cosine(ΔF1,p)")
        for m in methods:
            r = out[m]; tag = "" if m == "w1.0" else f"   ΔF1={r['dF1']:+.3f} p={r['p']:.1e}"
            print(f"{m:10s}{r['f1']:>8.3f}{r['score']:>8.3f}{tag}")


if __name__ == "__main__":
    main()
