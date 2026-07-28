"""Proper density-matched official scoring of the SEM-style learned-dynamics signals.

Methods: Bp_cos (delta_eff), FDM_err, FDM_best (learned forward-dynamics surprise), D_random.
Scorers: DTS official for tiage/dialseg/superseg (end-turn, -1); AMI official for ami (start-turn, +2 tol).
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
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"
METHODS = ["Bp_cos", "FDM_err", "FDM_best", "D_random"]


def signals(model, emb):
    z = torch.tensor(emb, device=DEV); zt, z1 = z[:-1], z[1:]
    with torch.no_grad():
        cb = model.vq._codebook.embed.reshape(-1, model.code_dim)
        qmean = cb.mean(0, keepdim=True).expand(len(zt), -1)
        err_mean = ((model.fdm(torch.cat([zt, qmean], -1)) - z1) ** 2).sum(-1).cpu().numpy()
        errs = [((model.fdm(torch.cat([zt, cb[k:k+1].expand(len(zt), -1)], -1)) - z1) ** 2).sum(-1)
                for k in range(cb.shape[0])]
        err_best = torch.stack(errs, 0).min(0).values.cpu().numpy()
    delta = 1.0 - (emb[:-1] * emb[1:]).sum(1)
    return {"Bp_cos": delta, "FDM_err": err_mean, "FDM_best": err_best}


def topn(sig, N, rng=None):
    if N <= 0:
        return set()
    if rng is not None:
        return set(rng.permutation(len(sig))[:N].tolist())
    return set(np.argsort(-sig)[:N].tolist())


def main():
    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]
    model = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"],
               commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")

    def eval_dts(ds):
        data = embed_split(ds); golds = []; preds = {m: [] for m in METHODS}
        f1 = {m: [] for m in METHODS}
        for d in data:
            n = len(d["turns"]); gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}; N = len(gold)
            g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0; golds.append(g)
            sig = signals(model, d["emb"])
            sel = {m: topn(sig[m], N) for m in ["Bp_cos", "FDM_err", "FDM_best"]}
            sel["D_random"] = topn(sig["Bp_cos"], N, rng=np.random.default_rng(7))
            for m in METHODS:
                p = [1 if t in sel[m] else 0 for t in range(n)]; p[-1] = 0; preds[m].append(p)
                f1[m].append(len(sel[m] & gold) / N if N else 0.0)
        r = {m: score_dialogues(golds, preds[m]) for m in METHODS}
        r["_f1arr"] = f1
        return r

    def eval_ami():
        data = load_ami("eval", enc); golds = []; preds = {m: [] for m in METHODS}
        f1 = {m: [] for m in METHODS}
        for d in data:
            n = len(d["turns"]); bt = list(d["bnd_top"]); bt[-1] = 0; N = sum(bt)
            gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}; golds.append(bt)
            sig = signals(model, d["emb"])
            sel = {m: topn(sig[m], N) for m in ["Bp_cos", "FDM_err", "FDM_best"]}
            sel["D_random"] = topn(sig["Bp_cos"], N, rng=np.random.default_rng(7))
            for m in METHODS:
                p = [0] * n
                for t in sel[m]:
                    if t + 1 < n:
                        p[t + 1] = 1
                p[-1] = 0; preds[m].append(p)
                f1[m].append(len(sel[m] & gold) / len(gold) if gold else 0.0)
        r = {m: score_meetings(golds, preds[m]) for m in METHODS}
        r["_f1arr"] = f1
        return r

    print(f"{'set':16s}{'method':10s}{'F1':>8s}{'Pk':>8s}{'WD':>8s}{'Score':>8s}  vs Bp_cos(ΔF1,p)")
    tasks = [("tiage_test", None), ("superseg_test", None), ("dialseg711_test", None), ("ami_eval", "ami")]
    for name, kind in tasks:
        r = eval_ami() if kind == "ami" else eval_dts(name)
        base = np.array(r["_f1arr"]["Bp_cos"])
        for m in METHODS:
            arr = np.array(r["_f1arr"][m]); d = arr - base
            p = 1.0 if np.all(d == 0) else float(wilcoxon(arr, base).pvalue)
            tag = "" if m == "Bp_cos" else f"  ΔF1={d.mean():+.3f} p={p:.1e}"
            print(f"{name if m==METHODS[0] else '':16s}{m:10s}{r[m]['f1']:>8.3f}{r[m]['pk']:>8.3f}"
                  f"{r[m]['wd']:>8.3f}{r[m]['score']:>8.3f}{tag}")


if __name__ == "__main__":
    main()
