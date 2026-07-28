"""CAPSTONE: fully no-parameter, noise-robust, DEPLOYABLE segmenter.
Signal = ms_mean (mean of z-scored multi-scale block-cosine, w in {1,2,3,5,8}).
Threshold = Otsu (parameter-free bimodal) + non-max suppression (self-limits count).
NO density-matching, NO per-dataset tuning. Compared to naive (raw cosine + mu+1sigma).
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
from adaptive_robust import otsu, _nms_from_thresh, rule_mu1sig  # noqa: E402
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


def ms_mean(emb):
    zs = []
    for w in WSET:
        s = block_cos(emb, w)
        zs.append((s - s.mean()) / (s.std() + 1e-9))
    return np.stack(zs, 0).mean(0)


def raw_cos(emb):
    return 1.0 - (emb[:-1] * emb[1:]).sum(1)


def otsu_nms(s):
    return _nms_from_thresh(s, np.full(len(s), otsu(s)))


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")
    # method = (signal_fn, decision_fn)
    METH = {
        "naive(cosraw+mu1sig)": (raw_cos, rule_mu1sig),
        "capstone(ms_mean+otsuNMS)": (ms_mean, otsu_nms),
    }
    for name, kind in [("tiage_test", None), ("superseg_test", None),
                       ("dialseg711_test", None), ("ami_eval", "ami")]:
        data = load_ami("eval", enc) if kind == "ami" else embed_split(name)
        golds = []; preds = {m: [] for m in METH}; npred = {m: 0 for m in METH}; ng = 0
        for d in data:
            n = len(d["turns"]); emb = d["emb"]
            if kind == "ami":
                bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt); ng += sum(bt)
            else:
                g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0
                golds.append(g); ng += sum(g)
            for m, (sigfn, decfn) in METH.items():
                sel = decfn(sigfn(emb))
                if kind == "ami":
                    p = [0] * n
                    for t in sel:
                        if t + 1 < n:
                            p[t + 1] = 1
                    p[-1] = 0
                else:
                    p = [1 if t in sel else 0 for t in range(n)]; p[-1] = 0
                preds[m].append(p); npred[m] += sum(p)
        scorer = score_meetings if kind == "ami" else score_dialogues
        print(f"\n=== {name} ({'NOISY' if kind=='ami' else 'clean'}) === gold={ng}, no density-match, no tuning")
        print(f"{'method':28s}{'F1':>8s}{'Pk':>7s}{'WD':>7s}{'Score':>8s}{'n_pred':>8s}")
        for m in METH:
            r = scorer(golds, preds[m])
            print(f"{m:28s}{r['f1']:>8.3f}{r['pk']:>7.3f}{r['wd']:>7.3f}{r['score']:>8.3f}{npred[m]:>8d}")


if __name__ == "__main__":
    main()
